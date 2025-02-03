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

import cv2
import numpy as np
from scipy import io as sio
from tqdm import tqdm

from csi.config import get_cfg
from csi.engine import default_argument_parser, default_setup
from csi.data import CSITrainDataset, LoadVal, shift_back_batch, generate_mask_3d, generate_mask_3d_shift, gen_meas_torch_batch
from csi.architectures import *
from csi.utils.schedulers import get_cosine_schedule_with_warmup
from csi.losses import CharbonnierLoss, TVLoss
from csi.metrics import torch_psnr, torch_ssim, torch_sam
from csi.utils.utils import checkpoint

args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if cfg.DATASETS.MASK_TYPE == "mask_3d":
    if cfg.EXPERIMENTAL_TYPE == "simulation":
        mask = generate_mask_3d(mask_path=cfg.DATASETS.VAL.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
    elif cfg.EXPERIMENTAL_TYPE == "real":
        mask_indoor = generate_mask_3d(mask_path=cfg.DATASETS.VAL_INDOOR.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
        mask_outdoor = generate_mask_3d(mask_path=cfg.DATASETS.VAL_OUTDOOR.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
elif cfg.DATASETS.MASK_TYPE == "mask_3d_shift":
    if cfg.EXPERIMENTAL_TYPE == "simulation":
        mask = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL.MASK_PATH).to(device)
    elif cfg.EXPERIMENTAL_TYPE == "real":
        mask_indoor = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL_INDOOR.MASK_PATH).to(device)
        mask_outdoor = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL_OUTDOOR.MASK_PATH).to(device)



if cfg.EXPERIMENTAL_TYPE == "simulation":
    val_datas = LoadVal(cfg.DATASETS.VAL.PATH)
elif cfg.EXPERIMENTAL_TYPE == "real":
    val_datas_indoor = LoadVal(cfg.DATASETS.VAL_INDOOR.PATH)
    val_datas_outdoor = LoadVal(cfg.DATASETS.VAL_OUTDOOR.PATH)



model = eval(cfg.MODEL.TYPE)(cfg).to(device)

ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

if cfg.PRETRAINED_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
    save_state = torch.load(cfg.PRETRAINED_CKPT_PATH, map_location=device)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])


def eval(val_data, mask, name="KAIST"):
    begin = time.time()
    model.eval()
    psnr_list, ssim_list, sam_list = [], [], []
    val_gt = torch.stack([torch.from_numpy(val_label).permute(2, 0, 1).to(device).float() for val_label in val_data['hsi']])
    model_out = []
    for val_label in val_data['hsi']:
        gt = torch.from_numpy(val_label).permute(2, 0, 1).to(device).float().unsqueeze(0)
        val_mask = mask.unsqueeze(0)
        Meas = gen_meas_torch_batch(
        gt, 
        val_mask, 
        step=cfg.DATASETS.STEP, 
        wave_len=cfg.DATASETS.WAVE_LENS, 
        mask_type=cfg.DATASETS.MASK_TYPE, 
        with_pan=cfg.DATASETS.WITH_PAN,
        with_noise=cfg.DATASETS.TRAIN.WITH_NOISE)

        data = {}
        data['hsi'] = gt
        data['MeasH'] = Meas['MeasH']
        B, C, H, W = data['MeasH'].shape
        data['mask'] = val_mask
        data['MeasC'] = Meas['MeasC']
        if cfg.DATASETS.MASK_TYPE == "mask_3d_shift":
            data['PhiC'] = val_mask
        if cfg.DATASETS.WITH_PAN:
            data['MeasP'] = Meas['MeasP']
            data['PhiP'] = torch.ones((B, C, H, W)).to(device)
 
        with torch.no_grad():
            with ema.average_parameters():
                out = model(data)
                model_out.append(out)

    model_out = torch.cat(model_out, axis=0)

    for i in range(len(model_out)):
        psnr_val = torch_psnr(model_out[i, :, :, :], val_gt[i, :, :, :])
        ssim_val = torch_ssim(model_out[i, :, :, :], val_gt[i, :, :, :])
        sam_val = torch_sam(model_out[i, :, :, :], val_gt[i, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val.detach().cpu().numpy())

    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(val_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    sam_mean = np.mean(np.asarray(sam_list))

    end = time.time()

    print('===> {}: testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, time: {:.2f}'
                .format(name, psnr_mean, ssim_mean, sam_mean, (end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, sam_list, psnr_mean, ssim_mean, sam_mean


def main():
    if cfg.EXPERIMENTAL_TYPE == "simulation":
        (pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval(val_datas, mask, name="KAIST")
    elif cfg.EXPERIMENTAL_TYPE == "real":
        (pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval(val_datas_indoor, mask_indoor, name="CAVE")
        (pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval(val_datas_outdoor, mask_outdoor, name="ICVL")

    sio.savemat("./results/ADRNN_XST_simu_KAIST.mat", {"pred": pred, "truth" : truth})



if __name__ == "__main__":
    main()