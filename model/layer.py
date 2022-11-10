import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import tensor_to_numpy, print_round_err, print_minmax_err

import torch.nn.utils.spectral_norm as SN

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)
CONST = nn.init.constant_
CONST0 = nn.init.zeros_
NORMAL = nn.init.normal_
CONV_INIT = nn.init.xavier_normal_

class ConvBiasRelu(nn.Module):
    def __init__(self, ch_in, ch_out, debug=False, isRelu=True, kernel_size=3, symmetric=False):
        super().__init__()
        self.name = 'ConvBiasRelu'
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.isRelu = isRelu
        self.symmetric = symmetric
        self.debug = debug

        self.conv_layer = conv_layer(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size)
        self.bias_layer = bias_layer(ch_out=ch_out)

    def __repr__(self):
        return f'{self.__class__.__name__}(ch_in={self.ch_in}, ch_out={self.ch_out}, relu={self.isRelu})'

    def forward(self, input):
        out = self.conv_layer(input)
        out = self.bias_layer(out)
        return out


class ConvBiasRelu_D(nn.Module):
    def __init__(self, ch_in, ch_out, debug=False, isRelu=True, kernel_size=3, symmetric=False):
        super().__init__()
        self.name = 'ConvBiasRelu'
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.isRelu = isRelu
        self.symmetric = symmetric
        self.debug = debug

        self.conv_layer = SN(conv_layer(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size,))
        self.bias_layer = bias_layer(ch_out=ch_out)

    def __repr__(self):
        return f'{self.__class__.__name__}(ch_in={self.ch_in}, ch_out={self.ch_out}, relu={self.isRelu})'

    def forward(self, input):
        out = self.conv_layer(input)
        out = self.bias_layer(out)
        return out

class conv_layer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3):
        super().__init__()
        self.name = 'Conv_Layer'
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size

        self.momentum = 1.0

        self.weight = nn.Parameter(torch.Tensor(self.ch_out, self.ch_in, self.kernel_size, self.kernel_size).to(device))
        CONV_INIT(self.weight)

    def __repr__(self):
        return f'{self.__class__.__name__}(ch_in={self.ch_in}, ch_out={self.ch_out})'

    def forward(self, input):
        out = F.conv2d(input, weight=self.weight, bias=None, padding=(self.kernel_size // 2))

        return out

class bias_layer(nn.Module):
    def __init__(self, name='Bias_Layer', ch_out=16):
        super().__init__()
        self.ch_out = ch_out
        self.name = name
        self.bias = nn.Parameter(torch.Tensor(np.zeros([1, self.ch_out, 1, 1])))

        CONST0(self.bias)

    def __repr__(self):
        return f'{self.__class__.__name__}(ch={self.ch_out})'

    def forward(self, input):
        out = input + self.bias.repeat(input.shape[0], 1, input.shape[2], input.shape[3])

        return out


class ResBlk(nn.Module):
    def __init__(self, ch=16, kernel_size=3):
        super().__init__()
        self.ch = ch
        self.kernel_size = kernel_size

        features = []
        features.append(ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        features.append(ConvBiasRelu(isRelu=False, ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        self.features = nn.Sequential(*features)

    def forward(self, input):
        res_input = input.clone()
        re = input

        for i in range(len(self.features)):
            re = self.features[i](re)

        out = res_input + re
        return out

class ResBlk_D(nn.Module):
    def __init__(self, ch=16, kernel_size=3):
        super().__init__()
        self.ch = ch
        self.kernel_size = kernel_size

        features = []
        features.append(
            ConvBiasRelu_D(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        features.append(ConvBiasRelu_D(isRelu=False, ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        self.features = nn.Sequential(*features)

    def forward(self, input):
        res_input = input.clone()
        re = input

        for i in range(len(self.features)):
            re = self.features[i](re)

        out = res_input + re

        return out



class DenseBlk(nn.Module):
    def __init__(self, ch=16, kernel_size=3):
        super().__init__()
        self.ch = ch
        self.kernel_size = kernel_size

        features = []
        features.append(
            ConvBiasRelu(ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        features.append(ConvBiasRelu(isRelu=False, ch_in=self.ch, ch_out=self.ch, kernel_size=self.kernel_size))
        self.features = nn.Sequential(*features)

    def forward(self, input):
        res_input = input.clone()
        re = input

        for i in range(len(self.features)):
            re = self.features[i](re)

        out = res_input + re

        return torch.cat([res_input, out], 1)
