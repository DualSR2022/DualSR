from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Resize Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--size', type=int, required=True, help='outputsize')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image)
r, g, b = img.split()

out_img_g = g.resize((opt.size, opt.size), Image.BICUBIC)
out_img_b = b.resize((opt.size, opt.size), Image.BICUBIC)
out_img_r = r.resize((opt.size, opt.size), Image.BICUBIC)
out_img = Image.merge('RGB', [out_img_r, out_img_g, out_img_b])

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
