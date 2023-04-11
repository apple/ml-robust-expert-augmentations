#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Block2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            # print(x.shape)
            x = block(x)
            # print(x.shape)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Encoder2d(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block2d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            # print(x.shape)
            x = block(x)
            # print(x.shape)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), cond_size=0):
        super().__init__()
        self.chs = chs
        cond_sizes = [0] * len(chs)
        cond_sizes[0] = cond_size
        self.upconvs = nn.ModuleList([nn.ConvTranspose1d(chs[i] + cond_sizes[i],
                                                         chs[i + 1], 2, stride=2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, 1])(enc_ftrs.unsqueeze(3)).squeeze(3)
        return enc_ftrs


class Decoder2d(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), cond_size=0):
        super().__init__()
        self.chs = chs
        cond_sizes = [0] * len(chs)
        cond_sizes[0] = cond_size
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i] + cond_sizes[i],
                                                         chs[i + 1], 2, stride=2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block2d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)#.squeeze(3)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1,
                 retain_dim=True):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv1d(dec_chs[-1], num_class, 1)
        self.final_act = nn.Sigmoid()
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class UNet2d(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1,
                 retain_dim=True):
        super().__init__()
        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.final_act = nn.Sigmoid()
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class ConditionalUNet(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64),
                 num_class=1, retain_dim=True, cond_dim=0, final_act=None):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs, cond_size=cond_dim)
        self.head = nn.Conv1d(dec_chs[-1], num_class, 1)
        self.final_act = nn.Sigmoid() if final_act is not None else nn.Identity()
        self.retain_dim = retain_dim

    def forward(self, x, cond=None):
        enc_ftrs = self.encoder(x)
        if cond is not None:
            cond = cond.unsqueeze(2).expand(-1, -1, enc_ftrs[-1].shape[2])
            enc_ftrs[-1] = torch.cat((cond, enc_ftrs[-1]), 1)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class ConditionalUNetReactionDiffusion(nn.Module):
    def __init__(self, z_a_dim=1, enc_chs=(2, 16, 32, 64, 128), dec_chs=(128, 64, 32, 16),
                 num_class=2, retain_dim=True, final_act=None):
        super().__init__()
        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs, cond_size=z_a_dim)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.final_act = nn.Sigmoid() if final_act is not None else nn.Identity()
        self.retain_dim = retain_dim

    def forward(self, x, cond=None):
        enc_ftrs = self.encoder(x)
        if cond is not None:
            cond = cond.unsqueeze(2).unsqueeze(2).expand(-1, -1, enc_ftrs[-1].shape[2], enc_ftrs[-1].shape[3])
            enc_ftrs[-1] = torch.cat((cond, enc_ftrs[-1]), 1)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out