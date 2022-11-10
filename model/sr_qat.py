import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import os
from model.layer import *


def apply_q(m):
    if isinstance(m, ConvBiasRelu) or isinstance(m, QuantInOut) or isinstance(m, ResBlk):
        print(f"q_convert: {m}")
        m.convert = True
        # m.quant_out.update_minmax = False
        if isinstance(m, ConvBiasRelu):
            m.conv_layer.convert = True
            m.bias_layer.convert = True
            m.quant_layer.convert = True
            # m.conv_layer.update_minmax = False
            # m.quant_layer.update_minmax = False


class SrNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=3, ch=8, skip=False, per_c=False, scale=3, lyr=6, kernel_size=3,
                 update_minmax=True, convert=False, param_root=(os.getcwd() + '/param/')):
        super().__init__()
        """ network architecture """
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch = ch
        self.kernel_size = kernel_size
        self.sr_scale = scale
        self.skip = skip

        """ quantization options """
        self.per_c = per_c
        self.param_root = param_root
        self.convert = convert
        self.update_minmax = update_minmax

        self.build_model()

    def build_model(self):

        # print('skip: ', self.skip)
        features = []
        features.append(
            ConvBiasRelu(ch_in=self.ch_in, ch_out=self.ch, kernel_size=self.kernel_size))
        features.append(
            ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))

        if self.skip:
            features.append(
                ResBlk(ch=self.ch, kernel_size=self.kernel_size))

        else:
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))

        features.append(
            ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        features.append(ConvBiasRelu(isRelu=False, symmetric=True, ch_in=self.ch,
                                     ch_out=self.ch_out * (self.sr_scale ** 2), kernel_size=self.kernel_size))

        self.shuffle = nn.PixelShuffle(upscale_factor=self.sr_scale)
        self.UpSample = nn.Upsample(scale_factor=3, mode='bicubic')
        self.features = nn.Sequential(*features)

    def forward(self, lr):
        out = self.features[0](lr)
        for i in range(1, len(self.features)):
            out = self.features[i](out)
        sr = self.shuffle(out)
        return sr


class NrNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=3, ch=8, skip=False, scale=1, lyr=5, kernel_size=3,
                 param_root=(os.getcwd() + '/param/')):
        super().__init__()
        """ network architecture """
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch = ch
        self.kernel_size = kernel_size
        self.sr_scale = scale
        self.skip = skip

        features = []
        features.append(
            ConvBiasRelu(ch_in=self.ch_in, ch_out=self.ch, kernel_size=self.kernel_size))

        if self.skip:
            features.append(
                ResBlk(ch=self.ch, kernel_size=self.kernel_size))

        else:
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
            features.append(
                ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))

        features.append(
            ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        features.append(ConvBiasRelu(isRelu=False, symmetric=True, ch_in=self.ch,
                                     ch_out=self.ch_out, kernel_size=self.kernel_size))
        self.features = nn.Sequential(*features)

    def forward(self, lr):
        out = self.features[0](lr)
        for i in range(1, len(self.features)):
            out = self.features[i](out)
        return out

class NrNet_D(nn.Module):
    def __init__(self, ch_in=3, ch_out=3, ch=8, skip=False, scale=2, lyr=5, kernel_size=3,
                 update_minmax=True, convert=False, bits_act=8, bits_weight=6, bits_bias=32,
                 param_root=(os.getcwd() + '/param/')):
        super().__init__()
        """ network architecture """
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch = ch
        self.kernel_size = kernel_size
        self.sr_scale = scale
        self.skip = skip

        features = []
        features.append(
            ConvBiasRelu_D(ch_in=self.ch_in, ch_out=self.ch, kernel_size=self.kernel_size))

        if self.skip:
            features.append(
                ResBlk_D(ch=self.ch, kernel_size=self.kernel_size))

        else:
            features.append(
                ConvBiasRelu_D(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))

        features.append(
            ConvBiasRelu_D(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        features.append(ConvBiasRelu_D(isRelu=False, symmetric=True, ch_in=self.ch,
                                       ch_out=self.ch_out, kernel_size=self.kernel_size))
        self.features = nn.Sequential(*features)

    def forward(self, lr):
        re, scale_in = self.quant_in(lr, None)
        for i in range(len(self.features)):
            re, scale_in = self.features[i](re, scale_in)
        sr = lr + re
        return sr

