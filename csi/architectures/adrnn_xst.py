import warnings
from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch import einsum

import numbers

from . import BaseModel
from csi.data import gen_meas_torch_batch


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# S-MSA
class SpectralMultiheadSelfAttention(nn.Module):
    def __init__(self, 
                 cfg,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape
        x = self.norm(x)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :,  :]
        .expand(bs, n_kv_heads, n_rep, slen,  head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


# S-GQA
class SpectralGroupedQueryAttention(nn.Module):
    def __init__(self, 
                 cfg,
                 dim, 
                 num_heads,  # 1, 2, 4
                 kv_num_heads, # 1, 1, 2,
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.rep = num_heads // kv_num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, (dim * 2) // self.rep, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d((dim * 2) // self.rep, (dim * 2) // self.rep, kernel_size=3, stride=1, padding=1, groups=(dim * 2) // self.rep, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape
        x = self.norm(x)
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.kv_num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.kv_num_heads)

        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out
    

# X-MSA
class CrossMultiheadSelfAttention(nn.Module):
    def __init__(self, 
                 cfg,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm_q = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.norm_k = LayerNorm(dim, LayerNorm_type=LayerNorm_type)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, q, k):
        b,c,h,w = q.shape
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = self.q_dwconv(self.q(q))
        kv = self.kv_dwconv(self.kv(k))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out
    

# X-GQA
class CrossGroupedQueryAttention(nn.Module):
    def __init__(self, 
                 cfg,
                 dim, 
                 num_heads, 
                 kv_num_heads,
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.rep = num_heads // kv_num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm_q = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.norm_k = LayerNorm(dim, LayerNorm_type=LayerNorm_type)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, (dim*2) // self.rep, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d((dim*2) // self.rep, (dim*2) // self.rep, kernel_size=3, stride=1, padding=1, groups=(dim*2) // self.rep, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, q, k):
        b,c,h,w = q.shape
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = self.q_dwconv(self.q(q))
        kv = self.kv_dwconv(self.kv(k))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.kv_num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.kv_num_heads)

        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out
    


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b, h, w, c]
        return out: [b, h, w, c]
        """
        out = self.net(x)
        return out
    

## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66
    ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x
    

def FFN_FN(
    cfg,
    ffn_name,
    dim
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim, 
                ffn_expansion_factor=cfg.MODEL.ADRNN_XST.FFN_EXPAND, 
            )
    elif ffn_name == "FeedForward":
        return FeedForward(dim = dim)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)
    
# S-GQAB
class SpectralGroupedQueryAttentionBlock(nn.Module):
    def __init__(self, 
                 cfg,
                 dim, 
                 num_heads,
                 layernorm_type,
                 num_blocks,
                 ):
        super().__init__()
        self.cfg = cfg

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SpectralGroupedQueryAttention(
                    cfg,
                    dim = dim, 
                    num_heads = num_heads,
                    kv_num_heads = num_heads // 2 if num_heads!=1 else 1,
                    LayerNorm_type = layernorm_type
                ),
                PreNorm(dim, FFN_FN(
                    cfg,
                    ffn_name = cfg.MODEL.ADRNN_XST.FFN_NAME,
                    dim = dim
                ),
                layernorm_type = layernorm_type)
            ]))


    def forward(self, x):
        for (s_msa, ffn) in self.blocks:
            x = x + s_msa(x) 
            x = x + ffn(x)

        return x



# X-GQAB
class CrossGroupedQueryAttentionBlock(nn.Module):
    def __init__(self, 
                 cfg,
                 dim, 
                 num_heads,
                 layernorm_type,
                 num_blocks,
                 ):
        super().__init__()
        self.cfg = cfg


        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                CrossGroupedQueryAttention(
                    cfg,
                    dim = dim, 
                    num_heads = num_heads,
                    kv_num_heads = num_heads // 2 if num_heads!=1 else 1,
                    LayerNorm_type = layernorm_type
                ),
                PreNorm(dim, FFN_FN(
                    cfg,
                    ffn_name = cfg.MODEL.ADRNN_XST.FFN_NAME,
                    dim = dim
                ),
                layernorm_type = layernorm_type)
            ]))


    def forward(self, q, k):
        for (x_msa, ffn) in self.blocks:
            q = q + x_msa(q, k) 
            q = q + ffn(q)

        return q


# XSAB
class CrossSpectralAttentionBlock(nn.Module):
    def __init__(self,
                 cfg,
                 dim, 
                 num_heads,
                 layernorm_type,
                 s_msab_num_blocks,
                 x_msab_num_blocks,
                 num_blocks,
                 ):
        super().__init__()
        self.cfg = cfg

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SpectralGroupedQueryAttentionBlock(
                    cfg,
                    dim = dim, 
                    num_heads = num_heads,
                    layernorm_type = layernorm_type,
                    num_blocks = s_msab_num_blocks,
                ),
                CrossGroupedQueryAttentionBlock(
                    cfg,
                    dim = dim, 
                    num_heads = num_heads,
                    layernorm_type = layernorm_type,
                    num_blocks = x_msab_num_blocks,
                ) if self.cfg.DATASETS.WITH_PAN else nn.Identity()
            ]))

    def forward(self, inputs):
        # q = inputs
        q = inputs['q']
        for (s_msab, x_msab) in self.blocks:
            q = q + s_msab(q)
            if self.cfg.DATASETS.WITH_PAN:
                k = inputs['k']
                q = q + x_msab(q, k) 

        return q

class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class XST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embeddingC = nn.Conv2d(cfg.MODEL.ADRNN_XST.IN_DIM, cfg.MODEL.ADRNN_XST.DIM, kernel_size=3, stride=1, padding=1, bias=False)

        if self.cfg.DATASETS.WITH_PAN:
            self.embeddingP = nn.Conv2d(cfg.MODEL.ADRNN_XST.IN_DIM, cfg.MODEL.ADRNN_XST.DIM, kernel_size=3, stride=1, padding=1, bias=False)

        self.EncoderC = nn.ModuleList([
            CrossSpectralAttentionBlock(
                cfg = cfg, 
                dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 0, 
                num_heads = 2 ** 0, 
                layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[0],
                s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
            ),
            CrossSpectralAttentionBlock(
                cfg = cfg, 
                dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1, 
                num_heads = 2 ** 1, 
                layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[1],
                s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
            ),
        ])

        if self.cfg.DATASETS.WITH_PAN:
            self.EncoderP = nn.ModuleList([
                CrossSpectralAttentionBlock(
                    cfg = cfg, 
                    dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 0, 
                    num_heads = 2 ** 0, 
                    layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                    num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[0],
                    s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                    x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
                ),
                CrossSpectralAttentionBlock(
                    cfg = cfg, 
                    dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1, 
                    num_heads = 2 ** 1, 
                    layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                    num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[1],
                    s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                    x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
                ),
            ])

        self.BottleNeckC = CrossSpectralAttentionBlock(
                cfg = cfg, 
                dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 2, 
                num_heads = 2 ** 2, 
                layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[2],
                s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
            )
        
        if self.cfg.DATASETS.WITH_PAN:
            self.BottleNeckP = CrossSpectralAttentionBlock(
                    cfg = cfg, 
                    dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 2, 
                    num_heads = 2 ** 2, 
                    layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                    num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[2],
                    s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                    x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
                )

        self.DecoderC = nn.ModuleList([
            CrossSpectralAttentionBlock(
                cfg = cfg, 
                dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1, 
                num_heads = 2 ** 1, 
                layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[3],
                s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
            ),
            CrossSpectralAttentionBlock(
                cfg = cfg, 
                dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 0, 
                num_heads = 2 ** 0, 
                layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[4],
                s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
            )
        ])

        if self.cfg.DATASETS.WITH_PAN:
            self.DecoderP = nn.ModuleList([
                CrossSpectralAttentionBlock(
                    cfg = cfg, 
                    dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1, 
                    num_heads = 2 ** 1, 
                    layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                    num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[3],
                    s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                    x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
                ),
                CrossSpectralAttentionBlock(
                    cfg = cfg, 
                    dim = cfg.MODEL.ADRNN_XST.DIM * 2 ** 0, 
                    num_heads = 2 ** 0, 
                    layernorm_type = cfg.MODEL.ADRNN_XST.LAYERNORM_TYPE,
                    num_blocks = cfg.MODEL.ADRNN_XST.NUM_BLOCKS[4],
                    s_msab_num_blocks = cfg.MODEL.ADRNN_XST.S_MSAB_NUM_BLOCKS,
                    x_msab_num_blocks = cfg.MODEL.ADRNN_XST.X_MSAB_NUM_BLOCKS,
                )
            ])

        self.DownsC = nn.ModuleList([
            DownSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 0),
            DownSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 1)
        ])
        if self.cfg.DATASETS.WITH_PAN:
            self.DownsP = nn.ModuleList([
                DownSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 0),
                DownSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 1)
            ])

        self.UpsC = nn.ModuleList([
            UpSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 2),
            UpSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 1)
        ])

        if self.cfg.DATASETS.WITH_PAN:
            self.UpsP = nn.ModuleList([
                UpSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 2),
                UpSample(cfg.MODEL.ADRNN_XST.DIM * 2 ** 1)
            ])

        self.fusionsC = nn.ModuleList([
            nn.Conv2d(
                in_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 2,
                out_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False
            ),
            nn.Conv2d(
                in_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1,
                out_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 0,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False
            )
        ])

        if self.cfg.DATASETS.WITH_PAN:
            self.fusionsP = nn.ModuleList([
                nn.Conv2d(
                    in_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 2,
                    out_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1,
                    kernel_size = 1,
                    stride = 1,
                    padding = 0,
                    bias = False
                ),
                nn.Conv2d(
                    in_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 1,
                    out_channels = cfg.MODEL.ADRNN_XST.DIM * 2 ** 0,
                    kernel_size = 1,
                    stride = 1,
                    padding = 0,
                    bias = False
                )
            ])

        self.mappingC = nn.Conv2d(cfg.MODEL.ADRNN_XST.DIM, cfg.MODEL.ADRNN_XST.OUT_DIM, kernel_size=3, stride=1, padding=1, bias=False)

        if self.cfg.DATASETS.WITH_PAN:
            self.mappingP = nn.Conv2d(cfg.MODEL.ADRNN_XST.DIM, cfg.MODEL.ADRNN_XST.OUT_DIM, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, inputs):
        xc = inputs['xc']
        b, c, h_inp, w_inp = xc.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        xc = F.pad(xc, [0, pad_w, 0, pad_h], mode='reflect')

        if self.cfg.DATASETS.WITH_PAN:
            xp = inputs['xp']
            b, c, h_inp, w_inp = xp.shape
            hb, wb = 16, 16
            pad_h = (hb - h_inp % hb) % hb
            pad_w = (wb - w_inp % wb) % wb
            xp = F.pad(xp, [0, pad_w, 0, pad_h], mode='reflect')


        xc1 = self.embeddingC(xc)
        if self.cfg.DATASETS.WITH_PAN:
            xp1 = self.embeddingP(xp)
        
        resC1 = self.EncoderC[0]({"q": xc1, "k": xp1}) if self.cfg.DATASETS.WITH_PAN else self.EncoderC[0]({"q": xc1})
        if self.cfg.DATASETS.WITH_PAN:
            resP1 = self.EncoderP[0]({"q": xp1, "k": xc1})

        xc2 = self.DownsC[0](resC1)
        if self.cfg.DATASETS.WITH_PAN:
            xp2 = self.DownsP[0](resP1)

        resC2 = self.EncoderC[1]({"q": xc2, "k": xp2}) if self.cfg.DATASETS.WITH_PAN else self.EncoderC[1]({"q": xc2})
        if self.cfg.DATASETS.WITH_PAN:
            resP2 = self.EncoderP[1]({"q": xp2, "k": xc2})

        xc4 = self.DownsC[1](resC2)
        if self.cfg.DATASETS.WITH_PAN:
            xp4 = self.DownsP[1](resP2)

        resC4 = self.BottleNeckC({"q": xc4, "k": xp4}) if self.cfg.DATASETS.WITH_PAN else self.BottleNeckC({"q": xc4})
        if self.cfg.DATASETS.WITH_PAN:
            resP4 = self.BottleNeckP({"q": xp4, "k": xc4})

        

        dec_resC2 = self.UpsC[0](resC4) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_resC2 = torch.cat([dec_resC2, resC2], dim=1) # dim * 2 ** 2
        dec_resC2 = self.fusionsC[0](dec_resC2) # dim * 2 ** 2 -> dim * 2 ** 1

        if self.cfg.DATASETS.WITH_PAN:
            dec_resP2 = self.UpsP[0](resP4) # dim * 2 ** 2 -> dim * 2 ** 1
            dec_resP2 = torch.cat([dec_resP2, resP2], dim=1) # dim * 2 ** 2
            dec_resP2 = self.fusionsP[0](dec_resP2) # dim * 2 ** 2 -> dim * 2 ** 1

        
        dec_resC2 = self.DecoderC[0]({"q": dec_resC2, "k": dec_resP2}) if self.cfg.DATASETS.WITH_PAN else self.DecoderC[0]({"q": dec_resC2})
        if self.cfg.DATASETS.WITH_PAN:
            dec_resP2 = self.DecoderP[0]({"q": dec_resP2 , "k": dec_resC2})

        dec_resC1 = self.UpsC[1](dec_resC2) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_resC1 = torch.cat([dec_resC1, resC1], dim=1) # dim * 2 ** 1 
        dec_resC1 = self.fusionsC[1](dec_resC1) # dim * 2 ** 1 -> dim * 2 ** 0    

        if self.cfg.DATASETS.WITH_PAN:
            dec_resP1 = self.UpsP[1](dec_resP2) # dim * 2 ** 1 -> dim * 2 ** 0
            dec_resP1 = torch.cat([dec_resP1, resP1], dim=1) # dim * 2 ** 1 
            dec_resP1 = self.fusionsP[1](dec_resP1) # dim * 2 ** 1 -> dim * 2 ** 0       

        
        dec_resC1 = self.DecoderC[1]({"q": dec_resC1, "k": dec_resP1}) if self.cfg.DATASETS.WITH_PAN else self.DecoderC[1]({"q": dec_resC1})
        if self.cfg.DATASETS.WITH_PAN:
            dec_resP1 = self.DecoderP[1]({"q": dec_resP1, "k": dec_resC1})

        out = self.mappingC(dec_resC1) + xc
        # out = self.mappingP(dec_resP1) + xp
        if self.cfg.DATASETS.WITH_PAN:
            out += self.mappingP(dec_resP1) + xp

        return out[:, :, :h_inp, :w_inp] / 2
    



def A(x, Phi):
    B, nC, H, W = x.shape
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y

def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [B, C, H, W] = inputs.shape
    temp = torch.zeros((B, C, H, W+(C-1)*step)).to(inputs.device)
    temp[:, :, :, :W] = inputs
    for i in range(C):
        temp[:,i,:,:] = torch.roll(temp[:,i,:,:], shifts=step*i, dims=2)
    return temp

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        # this op will result in inpalce operation
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs


class DP(nn.Module):
    def __init__(self, cfg):
        super(DP, self).__init__()
        self.cfg = cfg
        self.mu_c = torch.nn.Parameter(torch.Tensor([0.009]))
        self.mu = torch.nn.Parameter(torch.Tensor([0.001]))
        if self.cfg.DATASETS.WITH_PAN:
            self.mu_p = torch.nn.Parameter(torch.Tensor([0.009]))

    def forward(self, inputs):
        out = defaultdict()
        z = inputs['z']
        yc = inputs['yc']
        phi_c = inputs['phi_c']
        if self.cfg.DATASETS.WITH_PAN:
            xc_k_1 = inputs['xc']
            xp_k_1 = inputs['xp']


        B, C, HC, WC = z.shape
        phi_c_s = torch.sum(phi_c**2,1)
        phi_c_s[phi_c_s==0] = 1
        Phi_zc = A(z, phi_c)
        if self.cfg.DATASETS.WITH_PAN:
            Phi_xp_k_1 = A(xp_k_1, phi_c)
            xc = (self.mu_c / (self.mu + self.mu_c)) * (z + At(torch.div(yc-Phi_zc, (self.mu + self.mu_c)+phi_c_s), phi_c)) \
            + (self.mu / (self.mu + self.mu_c)) * (xp_k_1 + At(torch.div(yc-Phi_xp_k_1, (self.mu + self.mu_c)+phi_c_s), phi_c))
        else:
            xc = z + At(torch.div(yc-Phi_zc, self.mu_c+phi_c_s), phi_c)
        out['xc'] = xc

        if self.cfg.DATASETS.WITH_PAN:
            yp = inputs['yp']
            phi_p = inputs['phi_p']
            B, C, HP, WP = phi_p.shape
            phi_p_s = torch.sum(phi_p**2,1)
            phi_p_s[phi_p_s==0] = 1
            z = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :WP]            
            Phi_zp = A(z, phi_p)
            Phi_xc_k_1 = A(xc_k_1, phi_p)
            xp = (self.mu_p / (self.mu + self.mu_p)) * (z + At(torch.div(yp-Phi_zp, (self.mu + self.mu_p)+phi_p_s), phi_p)) \
            + (self.mu / (self.mu + self.mu_p)) * (xc_k_1 + At(torch.div(yp-Phi_xc_k_1, (self.mu + self.mu_p)+phi_p_s), phi_p))

            out['xp'] = xp

        return out
    

class ADRNN_XST(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.fusion = nn.Conv2d(cfg.DATASETS.WAVE_LENS*2, cfg.DATASETS.WAVE_LENS, 1, padding=0, bias=True)

        self.DP = nn.ModuleList([
           DP(cfg) for _ in range(cfg.MODEL.ADRNN_XST.STAGES)
        ]) if not cfg.MODEL.ADRNN_XST.SHARE_PARAMS else DP(cfg)

        self.PP = nn.ModuleList([
            XST(cfg) for _ in range(cfg.MODEL.ADRNN_XST.STAGES)
        ]) if not cfg.MODEL.ADRNN_XST.SHARE_PARAMS else XST(cfg)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_input(self, data):
        hsi = data['hsi']
        mask = data['mask']

        Meas = gen_meas_torch_batch(hsi, mask, step=self.cfg.DATASETS.STEP, wave_len=self.cfg.DATASETS.WAVE_LENS, mask_type=self.cfg.DATASETS.MASK_TYPE, with_noise=self.cfg.DATASETS.TRAIN.WITH_NOISE)

        data['MeasC'] = Meas['MeasC']
        data['MeasH'] = Meas['MeasH']
        data['PhiC'] = mask

        if self.cfg.DATASETS.WITH_PAN:
            data['MeasP'] = Meas['MeasP']
            data['PhiP'] = torch.ones_like(data['MeasH'])


        return data
    
    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC = self.cfg.DATASETS.WAVE_LENS
        step = self.cfg.DATASETS.STEP
        bs, nC, row, col = Phi.shape
        y_shift = torch.zeros(bs, nC, row, col).to(y.device).float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z
    


    def forward_train(self, data):
        yc = data['MeasC']
        x = data['MeasH']
        phi_c = data['PhiC']
        
        if self.cfg.DATASETS.WITH_PAN:
            phi_p = data['PhiP']
            yp = data['MeasP']
        
        B, nC, H_, W_ = x.shape 

        if self.cfg.EXPERIMENTAL_TYPE == "real":
            yc = yc * nC / 2 / 1.2
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * nC / 1.2

        z = self.initial(yc, phi_c)
        xc_k_1, xp_k_1 = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_], z
        for i in range(self.cfg.MODEL.ADRNN_XST.STAGES):

            inputs = {"yc": yc, "yp": yp, "z": z,  "xc": xc_k_1, "xp": xp_k_1,"phi_c": phi_c, "phi_p": phi_p} if self.cfg.DATASETS.WITH_PAN else {"yc": yc, "z": z, "phi_c": phi_c}
            out = self.DP[i](inputs) if not self.cfg.MODEL.ADRNN_XST.SHARE_PARAMS else self.DP(inputs)
            out['xc'] = shift_back_3d(out['xc'], step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
            xc_k_1, xp_k_1 = out['xc'], shift_3d(out['xp'], step=self.cfg.DATASETS.STEP)
            z = self.PP[i](out) if not self.cfg.MODEL.ADRNN_XST.SHARE_PARAMS else self.PP(out)
            z = shift_3d(z, step=self.cfg.DATASETS.STEP)


        z = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_]

        return z 

    
    def forward_test(self, data):
        yc = data['MeasC']
        x = data['MeasH']
        phi_c = data['PhiC']
        if self.cfg.DATASETS.WITH_PAN:
            phi_p = data['PhiP']
            yp = data['MeasP']
        
        B, nC, H_, W_ = x.shape  
        

        if self.cfg.EXPERIMENTAL_TYPE == "real":
            yc = yc * nC / 2 / 1.2
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * nC / 1.2

        z = self.initial(yc, phi_c)
        xc_k_1, xp_k_1 = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_], z

        for i in range(self.cfg.MODEL.ADRNN_XST.STAGES):
            inputs = {"yc": yc, "yp": yp, "z": z,  "xc": xc_k_1, "xp": xp_k_1,"phi_c": phi_c, "phi_p": phi_p} if self.cfg.DATASETS.WITH_PAN else {"yc": yc, "z": z, "phi_c": phi_c}
            out = self.DP[i](inputs) if not self.cfg.MODEL.ADRNN_XST.SHARE_PARAMS else self.DP(inputs)
            out['xc'] = shift_back_3d(out['xc'], step=self.cfg.DATASETS.STEP)[:, :, :, :W_]

            xc_k_1, xp_k_1 = out['xc'], shift_3d(out['xp'], step=self.cfg.DATASETS.STEP)
            z = self.PP[i](out) if not self.cfg.MODEL.ADRNN_XST.SHARE_PARAMS else self.PP(out)

            z = shift_3d(z, step=self.cfg.DATASETS.STEP)

        z = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
        

        return z 



class ADRNN_XST_Profiling(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.fusion = nn.Conv2d(cfg.DATASETS.WAVE_LENS*2, cfg.DATASETS.WAVE_LENS, 1, padding=0, bias=True)

        self.DP = nn.ModuleList([
           DP(cfg) for _ in range(cfg.MODEL.ADRNN_XST.STAGES)
        ]) if not cfg.MODEL.ADRNN_XST.SHARE_PARAMS else DP(cfg)

        self.PP = nn.ModuleList([
            XST(cfg) for _ in range(cfg.MODEL.ADRNN_XST.STAGES)
        ]) if not cfg.MODEL.ADRNN_XST.SHARE_PARAMS else XST(cfg)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

            
    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC = self.cfg.DATASETS.WAVE_LENS
        step = self.cfg.DATASETS.STEP
        bs, nC, row, col = Phi.shape
        y_shift = torch.zeros(bs, nC, row, col).to(y.device).float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z
    


            
    def forward(self, yc, yp, phi_c, phi_p, x):
        
        B, nC, H_, W_ = x.shape   

        if self.cfg.EXPERIMENTAL_TYPE == "real":
            y = y * nC / 2 / 1.2
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * nC / 1.2

        z = self.initial(yc, phi_c)
        xc_k_1, xp_k_1 = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_], z

        for i in range(self.cfg.MODEL.ADRNN_XST.STAGES):

            inputs = {"yc": yc, "yp": yp, "z": z,  "xc": xc_k_1, "xp": xp_k_1,"phi_c": phi_c, "phi_p": phi_p} if self.cfg.DATASETS.WITH_PAN else {"yc": yc, "z": z, "phi_c": phi_c}
            out = self.DP[i](inputs) if not self.cfg.MODEL.ADRNN_XST.SHARE_PARAMS else self.DP(inputs)
            out['xc'] = shift_back_3d(out['xc'], step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
            xc_k_1, xp_k_1 = out['xc'], shift_3d(out['xp'], step=self.cfg.DATASETS.STEP)
            z = self.PP[i](out) if not self.cfg.MODEL.ADRNN_XST.SHARE_PARAMS else self.PP(out)
            z = shift_3d(z, step=self.cfg.DATASETS.STEP)
            


        z = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_]

        return z  
 