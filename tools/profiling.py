import os
import sys
import time

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torch_ema import ExponentialMovingAverage
from fvcore.nn import FlopCountAnalysis

import cv2
import numpy as np
from scipy import io as sio
from tqdm import tqdm

from csi.config import get_cfg
from csi.engine import default_argument_parser, default_setup

from csi.architectures import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

model = eval(cfg.MODEL.TYPE)(cfg).to(device)
print(model)
yc = torch.randn((1, 256, 310)).to(device)
yp = torch.randn((1, 256, 256)).to(device)
phi_c = torch.randn((1, 28, 256, 310)).to(device)
phi_p = torch.randn((1, 28, 256, 256)).to(device)
x = torch.randn((1, 28, 256, 256)).to(device)


flops = FlopCountAnalysis(model, (yc, yp, phi_c, phi_p, x))

all_flops = flops.total()
n_param = sum([p.nelement() for p in model.parameters()])
print(f'GMac:{flops.total()/(1024*1024*1024)}')
print(f'Params:{n_param / (1024*1024)}M')


start_time = time.time()
model(yc, yp, phi_c, phi_p, x)
end_time = time.time()

print(f'Time:{end_time - start_time}s')