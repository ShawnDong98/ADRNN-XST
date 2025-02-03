from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import BaseModel
from csi.data import gen_meas_torch_batch

class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out
    

class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()
        self.E1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

    def forward(self, x):
        ## encoding blocks
        E1 = self.E1(x)
        E2 = self.E2(F.avg_pool2d(E1, kernel_size=2, stride=2))
        E3 = self.E3(F.avg_pool2d(E2, kernel_size=2, stride=2))
        E4 = self.E4(F.avg_pool2d(E3, kernel_size=2, stride=2))
        E5 = self.E5(F.avg_pool2d(E4, kernel_size=2, stride=2))
        return E1, E2, E3, E4, E5
    


class Decoding(nn.Module):
    def __init__(self, Ch=28, kernel_size=[7,7,7]):
        super(Decoding, self).__init__()
        self.upMode = 'bilinear'
        self.Ch = Ch
        out_channel1 = Ch * kernel_size[0]
        out_channel2 = Ch * kernel_size[1]
        out_channel3 = Ch * kernel_size[2]
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=128+128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D4 = nn.Sequential(nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

        self.w_generator = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=self.Ch, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=self.Ch, out_channels=self.Ch, kernel_size=1, stride=1, padding=0)
                                         )
        

        self.filter_g_1      = nn.Sequential(nn.Conv2d(64 + 32, out_channel1, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel1, out_channel1, 1, 1, 0)
                                             )

        self.filter_g_2      = nn.Sequential(nn.Conv2d(64 + 32, out_channel2, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel2, out_channel2, 1, 1, 0)
                                             )

        self.filter_g_3      = nn.Sequential(nn.Conv2d(64 + 32, out_channel3, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel3, out_channel3, 1, 1, 0)
                                             )
        

    def forward(self, E1, E2, E3, E4, E5):
        ## decoding blocks
        D1 = self.D1(torch.cat([E4, F.interpolate(E5, scale_factor=2, mode=self.upMode)], dim=1))
        D2 = self.D2(torch.cat([E3, F.interpolate(D1, scale_factor=2, mode=self.upMode)], dim=1))
        D3 = self.D3(torch.cat([E2, F.interpolate(D2, scale_factor=2, mode=self.upMode)], dim=1))
        D4 = self.D4(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))

        ## estimating the regularization parameters w
        w = self.w_generator(D4)

        ## generate 3D filters
        f1 = self.filter_g_1(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))
        f2 = self.filter_g_2(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))
        f3 = self.filter_g_3(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))
        return w, f1, f2, f3
    

class DGSMP(BaseModel):
    def __init__(self, cfg):
        super(DGSMP, self).__init__(cfg)
        Ch = cfg.MODEL.DGSMP.CH
        stages = cfg.MODEL.DGSMP.STAGES
        self.Ch = Ch
        self.s  = stages
        self.filter_size = [7,7,7]  ## 3D filter size

        ## The modules for learning the measurement matrix A and A^T
        self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.A  = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        ## Encoding blocks
        self.Encoding = Encoding()

        ## Decoding blocks
        self.Decoding = Decoding(Ch=self.Ch, kernel_size=self.filter_size)

        ## Dense connection
        if self.cfg.DATASETS.WITH_PAN:
            self.conv  = nn.Conv2d(Ch+1, 32, kernel_size=3, stride=1, padding=1)
        else:
            self.conv  = nn.Conv2d(Ch, 32, kernel_size=3, stride=1, padding=1)
        self.Den_con1 = nn.Conv2d(32    , 32, kernel_size=1, stride=1, padding=0)
        self.Den_con2 = nn.Conv2d(32 * 2, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con3 = nn.Conv2d(32 * 3, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con4 = nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con5 = nn.Conv2d(32 * 5, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con6 = nn.Conv2d(32 * 6, 32, kernel_size=1, stride=1, padding=0)


        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

        self._initialize_weights()
        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)


    def Filtering_1(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [self.filter_size[0] // 2, self.filter_size[0] // 2, 0, 0], mode='replicate')
        img_stack = []
        for i in range(self.filter_size[0]):
            img_stack.append(cube_pad[:, :, :, i:i + width])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out
    
    def Filtering_2(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [0, 0, self.filter_size[1] // 2, self.filter_size[1] // 2], mode='replicate')
        img_stack = []
        for i in range(self.filter_size[1]):
            img_stack.append(cube_pad[:, :, i:i + height, :])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out

    def Filtering_3(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube.unsqueeze(0).unsqueeze(0), pad=(0, 0, 0, 0, self.filter_size[2] // 2, self.filter_size[2] // 2)).squeeze(0).squeeze(0)
        img_stack = []
        for i in range(self.filter_size[2]):
            img_stack.append(cube_pad[:, i:i + bandwidth, :, :])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out
    

    def recon(self, res1, res2, Xt, i):
        if i == 0 :
            delta = self.delta_0
        elif i == 1:
            delta = self.delta_1
        elif i == 2:
            delta = self.delta_2
        elif i == 3:
            delta = self.delta_3
        # elif i == 4:
        #     delta = self.delta_4
        # elif i == 5:
        #     delta = self.delta_5

        Xt     =   Xt - 2 * delta * (res1 + res2)
        return Xt

    
    def y2x(self, y):
        ##  Spilt operator
        sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, self.cfg.DATASETS.WAVE_LENS, sz[2], sz[3] - self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)]).to(y.device)
        start = sz[3] - self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)
        for t in range(self.cfg.DATASETS.WAVE_LENS):
            temp = y[:, :, :, 0 + self.cfg.DATASETS.STEP * t : start + self.cfg.DATASETS.STEP * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x

    def x2y(self, x):
        ##  Shift and Sum operator
        sz = x.size()
        if len(sz) == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = x.size()
        y = torch.zeros([bs, 1, sz[2], sz[3] + self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)]).to(x.device)
        for t in range(self.cfg.DATASETS.WAVE_LENS):
            y[:, :, :, 0 + self.cfg.DATASETS.STEP * t : sz[3] + self.cfg.DATASETS.STEP * t] = x[:, t, :, :].unsqueeze(1) + y[:, :, :, 0 + self.cfg.DATASETS.STEP * t : sz[3] + self.cfg.DATASETS.STEP * t]
        return y

    
    def forward_train(self, data):
        yc = data['MeasC']
        h = data['MeasH']
        b, c, h_inp, w_inp = h.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        yc = F.pad(yc, [0, pad_w, 0, pad_h], mode='reflect')

        if self.cfg.DATASETS.WITH_PAN:
            yp = data['MeasP']
            yp = F.pad(yp, [0, pad_w, 0, pad_h], mode='reflect')


        ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
        B, C, _, _ = h.shape
        if self.cfg.EXPERIMENTAL_TYPE == "simulation":
            yc = yc / C * 2
        elif self.cfg.EXPERIMENTAL_TYPE == "real":
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * C / 1.2 
                
        yc = yc.unsqueeze(1)
        Xt = self.y2x(yc)
        feature_list = []

        for i in range(0, self.s):
            AXt = self.x2y(self.A(Xt))  # y = Ax
            Res1 = self.AT(self.y2x(AXt - yc))   # A^T * (Ax − y)

            if self.cfg.DATASETS.WITH_PAN:
                fea = self.conv(torch.cat([Xt, yp.unsqueeze(1)], dim=1))
            else:
                fea = self.conv(Xt)

            if i == 0:
                feature_list.append(fea)
                fufea = self.Den_con1(fea)
            elif i == 1:
                feature_list.append(fea)
                fufea = self.Den_con2(torch.cat(feature_list, 1))
            elif i == 2:
                feature_list.append(fea)
                fufea = self.Den_con3(torch.cat(feature_list, 1))
            elif i == 3:
                feature_list.append(fea)
                fufea = self.Den_con4(torch.cat(feature_list, 1))
            # elif i == 4:
            #     feature_list.append(fea)
            #     fufea = self.Den_con5(torch.cat(feature_list, 1))
            # elif i == 5:
            #     feature_list.append(fea)
            #     fufea = self.Den_con6(torch.cat(feature_list, 1))

            E1, E2, E3, E4, E5 = self.Encoding(fufea)
            W, f1, f2, f3 = self.Decoding(E1, E2, E3, E4, E5)

            batch_size, p, height, width = f1.size()
            f1                           = F.normalize(f1.view(batch_size, self.filter_size[0], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f2.size()
            f2                           = F.normalize(f2.view(batch_size, self.filter_size[1], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f3.size()
            f3                           = F.normalize(f3.view(batch_size, self.filter_size[2], self.Ch, height, width),dim=1)

            ## Estimating the local means U
            u1 = self.Filtering_1(Xt, f1)
            u2 = self.Filtering_2(u1, f2)
            U = self.Filtering_3(u2, f3)

            ## w * (x − u)
            Res2 = (Xt - U).mul(W)

            ## Reconstructing HSIs
            Xt = self.recon(Res1, Res2, Xt, i)

        return Xt[:, :, :h_inp, :w_inp]
    
    def forward_test(self, data):
        yc = data['MeasC']
        h = data['MeasH']
        b, c, h_inp, w_inp = h.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        yc = F.pad(yc, [0, pad_w, 0, pad_h], mode='reflect')

        if self.cfg.DATASETS.WITH_PAN:
            yp = data['MeasP']
            yp = F.pad(yp, [0, pad_w, 0, pad_h], mode='reflect')

        ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
        B, C, _, _ = h.shape
        if self.cfg.EXPERIMENTAL_TYPE == "simulation":
            yc = yc / C * 2
        elif self.cfg.EXPERIMENTAL_TYPE == "real":
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * C / 1.2 
        yc = yc.unsqueeze(1)
        Xt = self.y2x(yc)
        feature_list = []

        for i in range(0, self.s):
            AXt = self.x2y(self.A(Xt))  # y = Ax
            Res1 = self.AT(self.y2x(AXt - yc))   # A^T * (Ax − y)

            if self.cfg.DATASETS.WITH_PAN:
                fea = self.conv(torch.cat([Xt, yp.unsqueeze(1)], dim=1))
            else:
                fea = self.conv(Xt)

            if i == 0:
                feature_list.append(fea)
                fufea = self.Den_con1(fea)
            elif i == 1:
                feature_list.append(fea)
                fufea = self.Den_con2(torch.cat(feature_list, 1))
            elif i == 2:
                feature_list.append(fea)
                fufea = self.Den_con3(torch.cat(feature_list, 1))
            elif i == 3:
                feature_list.append(fea)
                fufea = self.Den_con4(torch.cat(feature_list, 1))
            # elif i == 4:
            #     feature_list.append(fea)
            #     fufea = self.Den_con5(torch.cat(feature_list, 1))
            # elif i == 5:
            #     feature_list.append(fea)
            #     fufea = self.Den_con6(torch.cat(feature_list, 1))

            E1, E2, E3, E4, E5 = self.Encoding(fufea)
            W, f1, f2, f3 = self.Decoding(E1, E2, E3, E4, E5)

            batch_size, p, height, width = f1.size()
            f1                           = F.normalize(f1.view(batch_size, self.filter_size[0], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f2.size()
            f2                           = F.normalize(f2.view(batch_size, self.filter_size[1], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f3.size()
            f3                           = F.normalize(f3.view(batch_size, self.filter_size[2], self.Ch, height, width),dim=1)

            ## Estimating the local means U
            u1 = self.Filtering_1(Xt, f1)
            u2 = self.Filtering_2(u1, f2)
            U = self.Filtering_3(u2, f3)

            ## w * (x − u)
            Res2 = (Xt - U).mul(W)

            ## Reconstructing HSIs
            Xt = self.recon(Res1, Res2, Xt, i)

        return Xt[:, :, :h_inp, :w_inp]
    


class DGSMP_Profiling(BaseModel):
    def __init__(self, cfg):
        super(DGSMP_Profiling, self).__init__(cfg)
        Ch = cfg.MODEL.DGSMP.CH
        stages = cfg.MODEL.DGSMP.STAGES
        self.Ch = Ch
        self.s  = stages
        self.filter_size = [7,7,7]  ## 3D filter size

        ## The modules for learning the measurement matrix A and A^T
        self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.A  = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        ## Encoding blocks
        self.Encoding = Encoding()

        ## Decoding blocks
        self.Decoding = Decoding(Ch=self.Ch, kernel_size=self.filter_size)

        ## Dense connection
        if self.cfg.DATASETS.WITH_PAN:
            self.conv  = nn.Conv2d(Ch+1, 32, kernel_size=3, stride=1, padding=1)
        else:
            self.conv  = nn.Conv2d(Ch, 32, kernel_size=3, stride=1, padding=1)
        self.Den_con1 = nn.Conv2d(32    , 32, kernel_size=1, stride=1, padding=0)
        self.Den_con2 = nn.Conv2d(32 * 2, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con3 = nn.Conv2d(32 * 3, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con4 = nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con5 = nn.Conv2d(32 * 5, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con6 = nn.Conv2d(32 * 6, 32, kernel_size=1, stride=1, padding=0)


        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

        self._initialize_weights()
        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)


    def Filtering_1(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [self.filter_size[0] // 2, self.filter_size[0] // 2, 0, 0], mode='replicate')
        img_stack = []
        for i in range(self.filter_size[0]):
            img_stack.append(cube_pad[:, :, :, i:i + width])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out
    
    def Filtering_2(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [0, 0, self.filter_size[1] // 2, self.filter_size[1] // 2], mode='replicate')
        img_stack = []
        for i in range(self.filter_size[1]):
            img_stack.append(cube_pad[:, :, i:i + height, :])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out

    def Filtering_3(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube.unsqueeze(0).unsqueeze(0), pad=(0, 0, 0, 0, self.filter_size[2] // 2, self.filter_size[2] // 2)).squeeze(0).squeeze(0)
        img_stack = []
        for i in range(self.filter_size[2]):
            img_stack.append(cube_pad[:, i:i + bandwidth, :, :])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out
    

    def recon(self, res1, res2, Xt, i):
        if i == 0 :
            delta = self.delta_0
        elif i == 1:
            delta = self.delta_1
        elif i == 2:
            delta = self.delta_2
        elif i == 3:
            delta = self.delta_3
        # elif i == 4:
        #     delta = self.delta_4
        # elif i == 5:
        #     delta = self.delta_5

        Xt     =   Xt - 2 * delta * (res1 + res2)
        return Xt
    
    def y2x(self, y):
        ##  Spilt operator
        sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, self.cfg.DATASETS.WAVE_LENS, sz[2], sz[3] - self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)]).to(y.device)
        start = sz[3] - self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)
        for t in range(28):
            temp = y[:, :, :, 0 + self.cfg.DATASETS.STEP * t : start + self.cfg.DATASETS.STEP * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x

    def x2y(self, x):
        ##  Shift and Sum operator
        sz = x.size()
        if len(sz) == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = x.size()
        y = torch.zeros([bs, 1, sz[2], sz[3] + self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)]).to(x.device)
        for t in range(28):
            y[:, :, :, 0 + self.cfg.DATASETS.STEP * t : sz[3] + self.cfg.DATASETS.STEP * t] = x[:, t, :, :].unsqueeze(1) + y[:, :, :, 0 + self.cfg.DATASETS.STEP * t : sz[3] + self.cfg.DATASETS.STEP * t]
        return y

    
    def forward(self, yc, yp, phi_c, phi_p, x):

        B, C, H, W = x.shape

        b, c, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        yc = F.pad(yc, [0, pad_w, 0, pad_h], mode='reflect')

        if self.cfg.DATASETS.WITH_PAN:
            yp = F.pad(yp, [0, pad_w, 0, pad_h], mode='reflect')


        if self.cfg.EXPERIMENTAL_TYPE == "simulation":
            yc = yc / C * 2
        elif self.cfg.EXPERIMENTAL_TYPE == "real":
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * C / 1.2 
        ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
        yc = yc.unsqueeze(1)
        Xt = self.y2x(yc)

        b, c, h_inp, w_inp = Xt.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        Xt = F.pad(Xt, [0, pad_w, 0, pad_h], mode='reflect')

        feature_list = []

        for i in range(0, self.s):
            AXt = self.x2y(self.A(Xt))  # y = Ax
            Res1 = self.AT(self.y2x(AXt - yc))   # A^T * (Ax − y)

            if self.cfg.DATASETS.WITH_PAN:
                fea = self.conv(torch.cat([Xt, yp.unsqueeze(1)], dim=1))
            else:
                fea = self.conv(Xt)

            if i == 0:
                feature_list.append(fea)
                fufea = self.Den_con1(fea)
            elif i == 1:
                feature_list.append(fea)
                fufea = self.Den_con2(torch.cat(feature_list, 1))
            elif i == 2:
                feature_list.append(fea)
                fufea = self.Den_con3(torch.cat(feature_list, 1))
            elif i == 3:
                feature_list.append(fea)
                fufea = self.Den_con4(torch.cat(feature_list, 1))
            # elif i == 4:
            #     feature_list.append(fea)
            #     fufea = self.Den_con5(torch.cat(feature_list, 1))
            # elif i == 5:
            #     feature_list.append(fea)
            #     fufea = self.Den_con6(torch.cat(feature_list, 1))

            E1, E2, E3, E4, E5 = self.Encoding(fufea)
            W, f1, f2, f3 = self.Decoding(E1, E2, E3, E4, E5)

            batch_size, p, height, width = f1.size()
            f1                           = F.normalize(f1.view(batch_size, self.filter_size[0], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f2.size()
            f2                           = F.normalize(f2.view(batch_size, self.filter_size[1], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f3.size()
            f3                           = F.normalize(f3.view(batch_size, self.filter_size[2], self.Ch, height, width),dim=1)

            ## Estimating the local means U
            u1 = self.Filtering_1(Xt, f1)
            u2 = self.Filtering_2(u1, f2)
            U = self.Filtering_3(u2, f3)

            ## w * (x − u)
            Res2 = (Xt - U).mul(W)

            ## Reconstructing HSIs
            Xt = self.recon(Res1, Res2, Xt, i)

        return Xt[:, :, :h_inp, :w_inp]
    
   




# class DGSMP(BaseModel):
#     def __init__(self, cfg):
#         super(DGSMP, self).__init__(cfg)
#         Ch = cfg.MODEL.DGSMP.CH
#         stages = cfg.MODEL.DGSMP.STAGES
#         self.Ch = Ch
#         self.s  = stages
#         self.filter_size = [7,7,7]  ## 3D filter size

#         ## The modules for learning the measurement matrix A and A^T
#         self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
#                                 Resblock(64), Resblock(64),
#                                 nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
#         self.A  = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
#                                 Resblock(64), Resblock(64),
#                                 nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

#         ## Encoding blocks
#         self.Encoding = Encoding()

#         ## Decoding blocks
#         self.Decoding = Decoding(Ch=self.Ch, kernel_size=self.filter_size)

#         ## Dense connection
#         self.conv  = nn.Conv2d(Ch, 32, kernel_size=3, stride=1, padding=1)
#         self.Den_con1 = nn.Conv2d(32    , 32, kernel_size=1, stride=1, padding=0)
#         self.Den_con2 = nn.Conv2d(32 * 2, 32, kernel_size=1, stride=1, padding=0)
#         self.Den_con3 = nn.Conv2d(32 * 3, 32, kernel_size=1, stride=1, padding=0)
#         self.Den_con4 = nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0)
#         # self.Den_con5 = nn.Conv2d(32 * 5, 32, kernel_size=1, stride=1, padding=0)
#         # self.Den_con6 = nn.Conv2d(32 * 6, 32, kernel_size=1, stride=1, padding=0)


#         self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
#         self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
#         self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
#         self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
#         # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
#         # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

#         self._initialize_weights()
#         torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
#         torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
#         torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
#         torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
#         # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
#         # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)


#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0.0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0.0)


#     def Filtering_1(self, cube, core):
#         batch_size, bandwidth, height, width = cube.size()
#         cube_pad = F.pad(cube, [self.filter_size[0] // 2, self.filter_size[0] // 2, 0, 0], mode='replicate')
#         img_stack = []
#         for i in range(self.filter_size[0]):
#             img_stack.append(cube_pad[:, :, :, i:i + width])
#         img_stack = torch.stack(img_stack, dim=1)
#         out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
#         return out
    
#     def Filtering_2(self, cube, core):
#         batch_size, bandwidth, height, width = cube.size()
#         cube_pad = F.pad(cube, [0, 0, self.filter_size[1] // 2, self.filter_size[1] // 2], mode='replicate')
#         img_stack = []
#         for i in range(self.filter_size[1]):
#             img_stack.append(cube_pad[:, :, i:i + height, :])
#         img_stack = torch.stack(img_stack, dim=1)
#         out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
#         return out

#     def Filtering_3(self, cube, core):
#         batch_size, bandwidth, height, width = cube.size()
#         cube_pad = F.pad(cube.unsqueeze(0).unsqueeze(0), pad=(0, 0, 0, 0, self.filter_size[2] // 2, self.filter_size[2] // 2)).squeeze(0).squeeze(0)
#         img_stack = []
#         for i in range(self.filter_size[2]):
#             img_stack.append(cube_pad[:, i:i + bandwidth, :, :])
#         img_stack = torch.stack(img_stack, dim=1)
#         out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
#         return out
    

#     def recon(self, res1, res2, Xt, i):
#         if i == 0 :
#             delta = self.delta_0
#         elif i == 1:
#             delta = self.delta_1
#         elif i == 2:
#             delta = self.delta_2
#         elif i == 3:
#             delta = self.delta_3
#         # elif i == 4:
#         #     delta = self.delta_4
#         # elif i == 5:
#         #     delta = self.delta_5

#         Xt     =   Xt - 2 * delta * (res1 + res2)
#         return Xt

#     def y2x(self, y):
#         if self.cfg.DATASETS.WITH_PAN:
#             yc, yp = y['MeasC'], y['MeasP']
#         else:
#             yc = y['MeasC']
#         ##  Spilt operator
#         sz = yc.size()
#         if len(sz) == 3:
#             yc = yc.unsqueeze(1)
#             if self.cfg.DATASETS.WITH_PAN:
#                 yp = yp.unsqueeze(1)

#         bs = sz[0]
#         sz = yc.size()
#         xc = torch.zeros([bs, self.cfg.DATASETS.WAVE_LENS, sz[2], sz[2]]).to(yc.device)
#         for t in range(self.cfg.DATASETS.WAVE_LENS):
#             temp = yc[:, :, :, 0 + self.cfg.DATASETS.STEP * t : sz[2] + self.cfg.DATASETS.STEP * t]

#             xc[:, t, :, :] = temp.squeeze(1)
#         if self.cfg.DATASETS.WITH_PAN:
#             xp = yp.repeat(1, self.cfg.DATASETS.WAVE_LENS, 1, 1)
#             x = xc + xp
#         else:
#             x = xc
        
#         return x
    
#     def x2y(self, x):
#         y = defaultdict()
#         ##  Shift and Sum operator
#         sz = x.size()
#         if len(sz) == 3:
#             x = x.unsqueeze(0).unsqueeze(0)
#             bs = 1
#         else:
#             bs = sz[0]
#         sz = x.size()
#         yc = torch.zeros([bs, sz[2], sz[2] + self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)]).to(x.device)
#         yp = torch.zeros([bs, sz[2], sz[2]]).to(x.device) 
#         for t in range(self.cfg.DATASETS.WAVE_LENS):
#             yc[:, :, 0 + self.cfg.DATASETS.STEP * t : sz[2] + self.cfg.DATASETS.STEP * t] = x[:, t, :, :] + yc[:, :, 0 + self.cfg.DATASETS.STEP * t : sz[2] + self.cfg.DATASETS.STEP * t]
#         yp = torch.sum(x, dim=1)

#         y['MeasC'] = yc
#         if self.cfg.DATASETS.WITH_PAN:
#             y['MeasP'] = yp

#         return y
    
#     def prepare_input(self, data):
#         hsi = data['hsi']
#         mask = data['mask']

#         Meas = gen_meas_torch_batch(hsi, mask, step=self.cfg.DATASETS.STEP, wave_len=self.cfg.DATASETS.WAVE_LENS, mask_type=self.cfg.DATASETS.MASK_TYPE, with_noise=self.cfg.DATASETS.TRAIN.WITH_NOISE)

#         data['MeasC'] = Meas['MeasC']
#         data['MeasH'] = Meas['MeasH']

#         if self.cfg.DATASETS.WITH_PAN:
#             data['MeasP'] = Meas['MeasP']

#         return data
    

    
#     def forward_train(self, data):
#         if self.cfg.DATASETS.WITH_PAN:
#             yc = data['MeasC']
#             yp = data['MeasP']
#         else:
#             yc = data['MeasC']
#         ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
#         y = {"MeasC": yc, "MeasP": yp}
#         Xt = self.y2x(y)
#         feature_list = []

#         for i in range(0, self.s):
#             AXt = self.x2y(self.A(Xt))  # y = Ax
#             AXt['MeasC'] = AXt['MeasC'] - yc
#             if self.cfg.DATASETS.WITH_PAN:
#                 AXt['MeasP'] = AXt['MeasP'] - yp
#             Res1 = self.AT(self.y2x(AXt))   # A^T * (Ax − y)

#             fea = self.conv(Xt)

#             if i == 0:
#                 feature_list.append(fea)
#                 fufea = self.Den_con1(fea)
#             elif i == 1:
#                 feature_list.append(fea)
#                 fufea = self.Den_con2(torch.cat(feature_list, 1))
#             elif i == 2:
#                 feature_list.append(fea)
#                 fufea = self.Den_con3(torch.cat(feature_list, 1))
#             elif i == 3:
#                 feature_list.append(fea)
#                 fufea = self.Den_con4(torch.cat(feature_list, 1))
#             # elif i == 4:
#             #     feature_list.append(fea)
#             #     fufea = self.Den_con5(torch.cat(feature_list, 1))
#             # elif i == 5:
#             #     feature_list.append(fea)
#             #     fufea = self.Den_con6(torch.cat(feature_list, 1))

#             E1, E2, E3, E4, E5 = self.Encoding(fufea)
#             W, f1, f2, f3 = self.Decoding(E1, E2, E3, E4, E5)

#             batch_size, p, height, width = f1.size()
#             f1                           = F.normalize(f1.view(batch_size, self.filter_size[0], self.Ch, height, width),dim=1)
#             batch_size, p, height, width = f2.size()
#             f2                           = F.normalize(f2.view(batch_size, self.filter_size[1], self.Ch, height, width),dim=1)
#             batch_size, p, height, width = f3.size()
#             f3                           = F.normalize(f3.view(batch_size, self.filter_size[2], self.Ch, height, width),dim=1)

#             ## Estimating the local means U
#             u1 = self.Filtering_1(Xt, f1)
#             u2 = self.Filtering_2(u1, f2)
#             U = self.Filtering_3(u2, f3)

#             ## w * (x − u)
#             Res2 = (Xt - U).mul(W)

#             ## Reconstructing HSIs
#             Xt = self.recon(Res1, Res2, Xt, i)

#         return Xt
    
#     def forward_test(self, data):
#         if self.cfg.DATASETS.WITH_PAN:
#             yc = data['MeasC']
#             yp = data['MeasP']
#         else:
#             yc = data['MeasC']
#         ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
#         y = {"MeasC": yc, "MeasP": yp}
#         Xt = self.y2x(y)
#         feature_list = []

#         for i in range(0, self.s):
#             AXt = self.x2y(self.A(Xt))  # y = Ax
#             AXt['MeasC'] = AXt['MeasC'] - yc
#             if self.cfg.DATASETS.WITH_PAN:
#                 AXt['MeasP'] = AXt['MeasP'] - yp
#             Res1 = self.AT(self.y2x(AXt))   # A^T * (Ax − y)

#             fea = self.conv(Xt)

#             if i == 0:
#                 feature_list.append(fea)
#                 fufea = self.Den_con1(fea)
#             elif i == 1:
#                 feature_list.append(fea)
#                 fufea = self.Den_con2(torch.cat(feature_list, 1))
#             elif i == 2:
#                 feature_list.append(fea)
#                 fufea = self.Den_con3(torch.cat(feature_list, 1))
#             elif i == 3:
#                 feature_list.append(fea)
#                 fufea = self.Den_con4(torch.cat(feature_list, 1))
#             # elif i == 4:
#             #     feature_list.append(fea)
#             #     fufea = self.Den_con5(torch.cat(feature_list, 1))
#             # elif i == 5:
#             #     feature_list.append(fea)
#             #     fufea = self.Den_con6(torch.cat(feature_list, 1))

#             E1, E2, E3, E4, E5 = self.Encoding(fufea)
#             W, f1, f2, f3 = self.Decoding(E1, E2, E3, E4, E5)

#             batch_size, p, height, width = f1.size()
#             f1                           = F.normalize(f1.view(batch_size, self.filter_size[0], self.Ch, height, width),dim=1)
#             batch_size, p, height, width = f2.size()
#             f2                           = F.normalize(f2.view(batch_size, self.filter_size[1], self.Ch, height, width),dim=1)
#             batch_size, p, height, width = f3.size()
#             f3                           = F.normalize(f3.view(batch_size, self.filter_size[2], self.Ch, height, width),dim=1)

#             ## Estimating the local means U
#             u1 = self.Filtering_1(Xt, f1)
#             u2 = self.Filtering_2(u1, f2)
#             U = self.Filtering_3(u2, f3)

#             ## w * (x − u)
#             Res2 = (Xt - U).mul(W)

#             ## Reconstructing HSIs
#             Xt = self.recon(Res1, Res2, Xt, i)

#         return Xt
    


# class DGSMP_Profiling(BaseModel):
#     def __init__(self, cfg):
#         super(DGSMP_Profiling, self).__init__(cfg)
#         Ch = cfg.MODEL.DGSMP.CH
#         stages = cfg.MODEL.DGSMP.STAGES
#         self.Ch = Ch
#         self.s  = stages
#         self.filter_size = [7,7,7]  ## 3D filter size

#         ## The modules for learning the measurement matrix A and A^T
#         self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
#                                 Resblock(64), Resblock(64),
#                                 nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
#         self.A  = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
#                                 Resblock(64), Resblock(64),
#                                 nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

#         ## Encoding blocks
#         self.Encoding = Encoding()

#         ## Decoding blocks
#         self.Decoding = Decoding(Ch=self.Ch, kernel_size=self.filter_size)

#         ## Dense connection
#         self.conv  = nn.Conv2d(Ch, 32, kernel_size=3, stride=1, padding=1)
#         self.Den_con1 = nn.Conv2d(32    , 32, kernel_size=1, stride=1, padding=0)
#         self.Den_con2 = nn.Conv2d(32 * 2, 32, kernel_size=1, stride=1, padding=0)
#         self.Den_con3 = nn.Conv2d(32 * 3, 32, kernel_size=1, stride=1, padding=0)
#         self.Den_con4 = nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0)
#         # self.Den_con5 = nn.Conv2d(32 * 5, 32, kernel_size=1, stride=1, padding=0)
#         # self.Den_con6 = nn.Conv2d(32 * 6, 32, kernel_size=1, stride=1, padding=0)


#         self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
#         self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
#         self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
#         self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
#         # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
#         # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

#         self._initialize_weights()
#         torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
#         torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
#         torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
#         torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
#         # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
#         # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)


#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0.0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0.0)


#     def Filtering_1(self, cube, core):
#         batch_size, bandwidth, height, width = cube.size()
#         cube_pad = F.pad(cube, [self.filter_size[0] // 2, self.filter_size[0] // 2, 0, 0], mode='replicate')
#         img_stack = []
#         for i in range(self.filter_size[0]):
#             img_stack.append(cube_pad[:, :, :, i:i + width])
#         img_stack = torch.stack(img_stack, dim=1)
#         out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
#         return out
    
#     def Filtering_2(self, cube, core):
#         batch_size, bandwidth, height, width = cube.size()
#         cube_pad = F.pad(cube, [0, 0, self.filter_size[1] // 2, self.filter_size[1] // 2], mode='replicate')
#         img_stack = []
#         for i in range(self.filter_size[1]):
#             img_stack.append(cube_pad[:, :, i:i + height, :])
#         img_stack = torch.stack(img_stack, dim=1)
#         out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
#         return out

#     def Filtering_3(self, cube, core):
#         batch_size, bandwidth, height, width = cube.size()
#         cube_pad = F.pad(cube.unsqueeze(0).unsqueeze(0), pad=(0, 0, 0, 0, self.filter_size[2] // 2, self.filter_size[2] // 2)).squeeze(0).squeeze(0)
#         img_stack = []
#         for i in range(self.filter_size[2]):
#             img_stack.append(cube_pad[:, i:i + bandwidth, :, :])
#         img_stack = torch.stack(img_stack, dim=1)
#         out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
#         return out
    

#     def recon(self, res1, res2, Xt, i):
#         if i == 0 :
#             delta = self.delta_0
#         elif i == 1:
#             delta = self.delta_1
#         elif i == 2:
#             delta = self.delta_2
#         elif i == 3:
#             delta = self.delta_3
#         # elif i == 4:
#         #     delta = self.delta_4
#         # elif i == 5:
#         #     delta = self.delta_5

#         Xt     =   Xt - 2 * delta * (res1 + res2)
#         return Xt

#     def y2x(self, y):
#         if self.cfg.DATASETS.WITH_PAN:
#             yc, yp = y['MeasC'], y['MeasP']
#         else:
#             yc = y['MeasC']
#         ##  Spilt operator
#         sz = yc.size()
#         if len(sz) == 3:
#             yc = yc.unsqueeze(1)
#             if self.cfg.DATASETS.WITH_PAN:
#                 yp = yp.unsqueeze(1)

#         bs = sz[0]
#         sz = yc.size()
#         xc = torch.zeros([bs, self.cfg.DATASETS.WAVE_LENS, sz[2], sz[2]]).to(yc.device)
#         for t in range(self.cfg.DATASETS.WAVE_LENS):
#             temp = yc[:, :, :, 0 + self.cfg.DATASETS.STEP * t : sz[2] + self.cfg.DATASETS.STEP * t]

#             xc[:, t, :, :] = temp.squeeze(1)
#         if self.cfg.DATASETS.WITH_PAN:
#             xp = yp.repeat(1, self.cfg.DATASETS.WAVE_LENS, 1, 1)
#             x = xc + xp
#         else:
#             x = xc
        
#         return x
    
#     def x2y(self, x):
#         y = defaultdict()
#         ##  Shift and Sum operator
#         sz = x.size()
#         if len(sz) == 3:
#             x = x.unsqueeze(0).unsqueeze(0)
#             bs = 1
#         else:
#             bs = sz[0]
#         sz = x.size()
#         yc = torch.zeros([bs, sz[2], sz[2] + self.cfg.DATASETS.STEP * (self.cfg.DATASETS.WAVE_LENS-1)]).to(x.device)
#         yp = torch.zeros([bs, sz[2], sz[2]]).to(x.device) 
#         for t in range(self.cfg.DATASETS.WAVE_LENS):
#             yc[:, :, 0 + self.cfg.DATASETS.STEP * t : sz[2] + self.cfg.DATASETS.STEP * t] = x[:, t, :, :] + yc[:, :, 0 + self.cfg.DATASETS.STEP * t : sz[2] + self.cfg.DATASETS.STEP * t]
#         yp = torch.sum(x, dim=1)

#         y['MeasC'] = yc
#         if self.cfg.DATASETS.WITH_PAN:
#             y['MeasP'] = yp

#         return y
    

    
#     def forward(self, yc, yp, phi_c, phi_p, x):
#         y = {"MeasC": yc, "MeasP": yp}
#         ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
#         Xt = self.y2x(y)
#         feature_list = []

#         for i in range(0, self.s):
#             AXt = self.x2y(self.A(Xt))  # y = Ax
#             AXt['MeasC'] = AXt['MeasC'] - yc
#             if self.cfg.DATASETS.WITH_PAN:
#                 AXt['MeasP'] = AXt['MeasP'] - yp
#             Res1 = self.AT(self.y2x(AXt))   # A^T * (Ax − y)

#             fea = self.conv(Xt)

#             if i == 0:
#                 feature_list.append(fea)
#                 fufea = self.Den_con1(fea)
#             elif i == 1:
#                 feature_list.append(fea)
#                 fufea = self.Den_con2(torch.cat(feature_list, 1))
#             elif i == 2:
#                 feature_list.append(fea)
#                 fufea = self.Den_con3(torch.cat(feature_list, 1))
#             elif i == 3:
#                 feature_list.append(fea)
#                 fufea = self.Den_con4(torch.cat(feature_list, 1))
#             # elif i == 4:
#             #     feature_list.append(fea)
#             #     fufea = self.Den_con5(torch.cat(feature_list, 1))
#             # elif i == 5:
#             #     feature_list.append(fea)
#             #     fufea = self.Den_con6(torch.cat(feature_list, 1))

#             E1, E2, E3, E4, E5 = self.Encoding(fufea)
#             W, f1, f2, f3 = self.Decoding(E1, E2, E3, E4, E5)

#             batch_size, p, height, width = f1.size()
#             f1                           = F.normalize(f1.view(batch_size, self.filter_size[0], self.Ch, height, width),dim=1)
#             batch_size, p, height, width = f2.size()
#             f2                           = F.normalize(f2.view(batch_size, self.filter_size[1], self.Ch, height, width),dim=1)
#             batch_size, p, height, width = f3.size()
#             f3                           = F.normalize(f3.view(batch_size, self.filter_size[2], self.Ch, height, width),dim=1)

#             ## Estimating the local means U
#             u1 = self.Filtering_1(Xt, f1)
#             u2 = self.Filtering_2(u1, f2)
#             U = self.Filtering_3(u2, f3)

#             ## w * (x − u)
#             Res2 = (Xt - U).mul(W)

#             ## Reconstructing HSIs
#             Xt = self.recon(Res1, Res2, Xt, i)

#         return Xt