from __future__ import print_function
import argparse
import random
import torch
from torch.autograd import Variable
from PIL import Image
import tensorflow as tf
import cv2
import os

import numpy as np
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')

parser.add_argument('--lr_dir', type=str, required=True, help='LR directory')
parser.add_argument('--hlr_dir', type=str, required=True, help='HighQuality LR directory')
parser.add_argument('--hr_dir', type=str, required=True, help='HR directory')

parser.add_argument('--patchsize', type=int, default=128, required=True, help='patch size')
parser.add_argument('--boundary', type=int, default=150, help='boundary size')
parser.add_argument('--scale', type=int, default=3, required=True, help='scale')

parser.add_argument('--output_dir', type=str, help='where to save the output image')
opt = parser.parse_args()

print(opt)

lr_img = Image.open(opt.lr_dir+'/'+opt.input_image).convert('RGB')
hr_img = Image.open(opt.hr_dir+'/'+opt.input_image).convert('RGB')
hlr_img = Image.open(opt.hlr_dir+'/'+opt.input_image).convert('RGB')

random_rotate = [0, 90, 180, 270]

def sel_patch(patch_in, level_th, patch_size):
    patch_in = patch_in.crop((patch_size*2/5, patch_size*2/5, patch_size*3/5, patch_size*3/5))
    patch = patch_in.convert("YCbCr")
    psize = patch.size[0]
    patch = np.asarray(patch)
    sobel = cv2.Sobel(patch[:, :, 0], -1, 1, 1, ksize=15)
    # print(patch.shape)
    # print(f"{sobel.shape}, {np.min(sobel)}, {np.max(sobel)}, {np.count_nonzero(sobel>20)}")

    if np.count_nonzero(sobel > level_th/36) > 0.2 * psize * psize:
        return True
    else:
        return False


def func_crop_patch(img_lr, img_hlr, img_hr, patch_size, scale, boundary):
    sel_edge = True

    if sel_edge:
        flag_stop = False
        force_stop_cnt = 0
        force_stop = False
        while not (flag_stop or force_stop):
            crop_x = random.randint(boundary, 1280  -boundary -patch_size)
            crop_y = random.randint(boundary, 720  -boundary -patch_size)

            patch_hr = img_hr.crop(
                (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
            res_sel = sel_patch(patch_hr, 1000-force_stop_cnt, patch_size=patch_size)
            force_stop_cnt += 1
            if force_stop_cnt > 1000:
                force_stop = True
            if res_sel:
                flag_stop = True

    else:
        crop_x = random.randint(boundary, 1280 - boundary - patch_size)
        crop_y = random.randint(boundary, 720 - boundary - patch_size)

    #rot = random_rotate[random.randrange(0, 4)]
    rot = 0

    print(crop_x,  crop_y, '\n')

    patch_hr = img_hr.crop(
        (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    patch_hr = patch_hr.rotate(rot)

    patch_hlr = img_hlr.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    patch_hlr = patch_hlr.rotate(rot)

    patch_lr = img_lr.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    patch_lr = patch_lr.rotate(rot)

    return patch_lr, patch_hlr, patch_hr


for multinum in range(1):
    patch_lr, patch_hlr, patch_hr = func_crop_patch(img_lr=lr_img, img_hlr=hlr_img, img_hr=hr_img,
                                                    patch_size=opt.patchsize, scale=opt.scale, boundary=opt.boundary)
    # rx = random.randint(0, w-tw)
    # ry = random.randint(0, h-th)
    #
    # crop_img = img.crop(( rx,ry, rx+tw , ry+th))


    base = opt.input_image.split('.')[0]
    ext  = opt.input_image.split('.')[1]
    out_name = base.format(multinum) + '.' + ext
    #out_name = base + '.' + ext

    out_hr_name = opt.output_dir + '/hr/' + out_name
    out_lr_name = opt.output_dir + '/lr/' + out_name
    out_hlr_name = opt.output_dir + '/hlr/' + out_name

    print('output image saved to ', out_hr_name)
    print('output image saved to ', out_lr_name)
    print('output image saved to ', out_hlr_name)

    patch_hr.save(out_hr_name)
#    patch_hlr.save(out_hlr_name)
    patch_lr.save(out_lr_name)
