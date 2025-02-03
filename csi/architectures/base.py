import torch
from torch import nn
from csi.data import gen_meas_torch_batch

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_input(self, data):
        hsi = data['hsi']
        mask = data['mask']

        Meas = gen_meas_torch_batch(hsi, mask, step=self.cfg.DATASETS.STEP, wave_len=self.cfg.DATASETS.WAVE_LENS, mask_type=self.cfg.DATASETS.MASK_TYPE, with_noise=self.cfg.DATASETS.TRAIN.WITH_NOISE)

        data['MeasC'] = Meas['MeasC']
        data['MeasH'] = Meas['MeasH']

        if self.cfg.DATASETS.WITH_PAN:
            data['MeasP'] = Meas['MeasP']

        return data

    def forward(self, data):
        if self.training:
            data = self.prepare_input(data)
            out = self.forward_train(data)
        else:
            out = self.forward_test(data)

        return out

    def forward_train(self, data):
        pass


    def forward_test(self, data):
        pass