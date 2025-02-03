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
from csi.data import CSITrainDataset, LoadVal, LoadTestMeas, generate_mask_3d, generate_mask_3d_shift, gen_meas_torch_batch, shift_back_batch
from csi.architectures import *
from csi.utils.schedulers import get_cosine_schedule_with_warmup
from csi.losses import CharbonnierLoss, TVLoss
from csi.metrics import torch_psnr, torch_ssim, torch_sam
from csi.utils.utils import checkpoint



args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()
logger, writer, output_dir = default_setup(cfg, args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if cfg.DATASETS.MASK_TYPE == "mask_3d":
    if cfg.EXPERIMENTAL_TYPE == "simulation":
        mask = generate_mask_3d(mask_path=cfg.DATASETS.VAL.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
    elif cfg.EXPERIMENTAL_TYPE == "real":
        mask_indoor = generate_mask_3d(mask_path=cfg.DATASETS.VAL_INDOOR.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
        mask_outdoor = generate_mask_3d(mask_path=cfg.DATASETS.VAL_OUTDOOR.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
        mask_test_indoor = generate_mask_3d(mask_path=cfg.DATASETS.TEST_INDOOR.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
        mask_test_outdoor = generate_mask_3d(mask_path=cfg.DATASETS.TEST_OUTDOOR.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS).to(device)
elif cfg.DATASETS.MASK_TYPE == "mask_3d_shift":
    if cfg.EXPERIMENTAL_TYPE == "simulation":
        mask = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL.MASK_PATH).to(device)
    elif cfg.EXPERIMENTAL_TYPE == "real":
        mask_indoor = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL_INDOOR.MASK_PATH).to(device)
        mask_outdoor = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL_OUTDOOR.MASK_PATH).to(device)
        mask_test_indoor = generate_mask_3d_shift(mask_path=cfg.DATASETS.TEST_INDOOR.MASK_PATH).to(device)
        mask_test_outdoor = generate_mask_3d_shift(mask_path=cfg.DATASETS.TEST_OUTDOOR.MASK_PATH).to(device)


if cfg.EXPERIMENTAL_TYPE == "simulation":
    val_datas = LoadVal(cfg.DATASETS.VAL.PATH)
elif cfg.EXPERIMENTAL_TYPE == "real":
    val_datas_indoor = LoadVal(cfg.DATASETS.VAL_INDOOR.PATH)
    val_datas_outdoor = LoadVal(cfg.DATASETS.VAL_OUTDOOR.PATH)

if cfg.EXPERIMENTAL_TYPE == "real":
    test_indoor_meas = LoadTestMeas(cfg.DATASETS.TEST_INDOOR.PATH, cfg.DATASETS.WITH_PAN)
    test_outdoor_meas = LoadTestMeas(cfg.DATASETS.TEST_OUTDOOR.PATH, cfg.DATASETS.WITH_PAN)


model = eval(cfg.MODEL.TYPE)(cfg).to(device)

ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

# optimizing
optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR, betas=(0.9, 0.999))

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(np.floor(cfg.DATASETS.TRAIN.ITERATION / cfg.DATALOADER.BATCH_SIZE)), 
    num_training_steps=int(np.floor(cfg.DATASETS.TRAIN.ITERATION / cfg.DATALOADER.BATCH_SIZE)) * cfg.OPTIMIZER.MAX_EPOCH, 
    eta_min=1e-6)

if cfg.LOSSES.L1_LOSS: l1_loss = CharbonnierLoss().to(device)
if cfg.LOSSES.TV_LOSS: tv_loss = TVLoss().to(device)

start_epoch = 0

if cfg.RESUME_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.RESUME_CKPT_PATH}")
    save_state = torch.load(cfg.RESUME_CKPT_PATH)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])
    optimizer.load_state_dict(save_state['optimizer'])
    scheduler.load_state_dict(save_state['scheduler'])
    start_epoch = save_state['epoch']

if cfg.LOSSES.L1_LOSS: l1_loss = CharbonnierLoss().to(device)
if cfg.LOSSES.L2_LOSS: l2_loss = torch.nn.MSELoss().to(device)
if cfg.LOSSES.TV_LOSS: tv_loss = TVLoss().to(device)

def train(epoch, train_loader):
    model.train()
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(cfg.DATASETS.TRAIN.ITERATION / train_loader.batch_size))
    train_tqdm = tqdm(range(batch_num)[:5])  if cfg.DEBUG else tqdm(range(batch_num))

    loss_dict = {}
    for i in train_tqdm:
        data_time = time.time()
        try:
            data = next(data_iter)
        except:
            data_iter = iter(train_loader)
            data = next(data_iter)
        
        data = {k:v.to(device) for k, v in data.items()}
       
        data_time = time.time() - data_time

        model_time = time.time()
        # model_out = model(meas_batch)
        model_out = model(data)
        model_time = time.time() - model_time

        loss = 0
        if cfg.LOSSES.L1_LOSS: 
            loss_l1 = l1_loss(model_out, data['hsi'])
            loss_dict['loss_l1'] = f"{loss_l1.item():.4f}"
            loss += loss_l1
        if cfg.LOSSES.L2_LOSS: 
            loss_l2 = l2_loss(model_out, data['hsi'])
            loss_dict['loss_l2'] = f"{loss_l2.item():.4f}"
            loss += loss_l2
        if cfg.LOSSES.TV_LOSS:
            loss_tv = tv_loss(model_out)
            loss_dict['loss_tv'] = f"{loss_tv.item():.4f}"
            loss += loss_tv

        loss.backward()
        if cfg.OPTIMIZER.GRAD_CLIP:
            clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()
        optimizer.zero_grad()
        ema.update()
        loss_dict['data_time'] = data_time
        loss_dict['model_time'] = model_time
        train_tqdm.set_postfix(loss_dict)
        epoch_loss += loss.data
        writer.add_scalar('LR/train',optimizer.state_dict()['param_groups'][0]['lr'], epoch * batch_num + i)
        scheduler.step()
    end = time.time()
    train_loss = epoch_loss / batch_num
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, train_loss, (end - begin)))
    return train_loss


def eval(epoch, val_data, mask, name="KAIST"):
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

    model_out = F.interpolate(model_out, size=(128, 128))
    for i, out in enumerate(model_out):
        out_grid = make_grid(out.unsqueeze(1).clip(0, 1), nrow=7)
        writer.add_image(f'images/{name}_scene{i}', out_grid, epoch)

    end = time.time()

    logger.info('===> {} Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, time: {:.2f}'
                .format(name, epoch, psnr_mean, ssim_mean, sam_mean, (end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, sam_list, psnr_mean, ssim_mean, sam_mean


def test(epoch, meas, mask, name="DCCHI_Indoor"):
    logger.info(f"Testing {name}...")
    model.eval()

    MeasC = meas['MeasC'].to(device)
    MeasH = shift_back_batch(MeasC, step=cfg.DATASETS.STEP, nC=cfg.DATASETS.WAVE_LENS)
    if cfg.DATASETS.WITH_PAN:
        MeasP = meas['MeasP'].to(device)
    model_out = []

    # For large-scale images, memory is limited
    for i in range(len(MeasC)):
        data = {}
        data['MeasC'] = MeasC[i:i+1, :, :]
        data['MeasH'] = MeasH[i:i+1, :, :, :]
        B, C, H, W = data['MeasH'].shape
        data['mask'] = mask.unsqueeze(0).tile((B, 1, 1, 1))
        if cfg.DATASETS.MASK_TYPE == "mask_3d_shift":
            data['PhiC'] = data['mask']
        if cfg.DATASETS.WITH_PAN:
            data['MeasP'] = MeasP[i:i+1, :, :]
            data['PhiP'] = torch.ones((B, C, H, W)).to(device)

        with torch.no_grad():
            with ema.average_parameters():
                out = model(data)
                model_out.append(out)

        out_plot = F.interpolate(out, size=(128, 128))
        if name == "TSA": out_plot = torch.flip(out_plot, dims=(2, 3))
        grid = make_grid(out_plot.permute(1, 0, 2, 3).clip(0, 1), nrow=7)
        writer.add_image('images/' + name + f'_scene{i}', grid, epoch)

    model_out = torch.cat(model_out, axis=0)
    model_out = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()

    return model_out



def main():
    psnr_max = 0
    sam_min = 9999
    dataset = CSITrainDataset(cfg, crop_size=cfg.DATASETS.TRAIN.CROP_SIZE)
    for epoch in range(start_epoch+1, cfg.OPTIMIZER.MAX_EPOCH):
        train_loader = DataLoader(
            dataset = dataset,
            batch_size = cfg.DATALOADER.BATCH_SIZE,
            shuffle = True,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
            pin_memory = False,
            drop_last = True
        )
        train_loss = train(epoch, train_loader)
        torch.cuda.empty_cache()
        if cfg.EXPERIMENTAL_TYPE == "simulation":
            (pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval(val_datas, mask, name="KAIST")
            if sam_mean < sam_min:
                sam_min = sam_mean
                checkpoint(model, ema, optimizer, scheduler, epoch, output_dir, logger)
                sio.savemat(os.path.join(output_dir, "val", f"epoch{epoch}_SAM{sam_mean}.mat"), {"pred": pred, "truth": truth})
        elif cfg.EXPERIMENTAL_TYPE == "real":
            checkpoint(model, ema, optimizer, scheduler, epoch, output_dir, logger)
            (pred_indoor, truth_indoor, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval(epoch, val_datas_indoor, mask_indoor, name="CAVE")
            sio.savemat(os.path.join(output_dir, "val", f"val_indoor_epoch{epoch}_SAM{sam_mean}.mat"), {"pred": pred_indoor, "truth": truth_indoor})
            if cfg.MODEL.TYPE in ("Lambda_Net"): continue
            (pred_outdoor, truth_outdoor, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval(epoch, val_datas_outdoor, mask_outdoor, name="ICVL")
            sio.savemat(os.path.join(output_dir, "val", f"val_outdoor_epoch{epoch}_SAM{sam_mean}.mat"), {"pred": pred_outdoor, "truth": truth_outdoor})
            test_indoor_out = test(epoch, test_indoor_meas, mask_test_indoor, "DCCHI_Indoor")
            sio.savemat(os.path.join(output_dir, "test", f"test_indoor_epoch{epoch}_SAM{sam_mean}.mat"), {"pred": test_indoor_out})
            test_outdoor_out = test(epoch, test_outdoor_meas, mask_test_outdoor, "DCCHI_Outdoor")
            sio.savemat(os.path.join(output_dir, "test", f"test_outdoor_epoch{epoch}_SAM{sam_mean}.mat"), {"pred": test_outdoor_out})
            


if __name__ == '__main__':
    main()