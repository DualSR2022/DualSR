import argparse
from asyncore import write
from csv import writer
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import torch.onnx
import math

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from utils import *
from tqdm import tqdm

from dataset import SR_Dataset, SR_Dataset_x3, SR_Dataset_x2
#from model.sr_qat import SrNet, NRNet, NRNet_D
from model.rcan import RCAN

# # Setup warnings
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torch.quantization')

def get_features(input, md, is_target):
    if is_target:
        model = RCAN(args=opt).to(device).eval()
        model.state_dict(md.state_dict())
    else:
        model = md

    features = []
    for _, layer in enumerate(model.head.children()+model.body.children()+model.tail.children()):
        input = layer(input)
        features.append(input)

    return features

if __name__ == '__main__':
    cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='psnr_log_221026_with_distBN.txt',
                        help='pre-trained model directory')
    parser.add_argument('--input', type=str)
    parser.add_argument('--dir_out', type=str, default='./dualSR_output/')

    parser.add_argument('--weight_folder', type=str, default='./runs/')
    parser.add_argument('--weight_S1', type=str, default='epoch_1103_RCAN_rB6_rG5_S1d_distNorm_015.pth')
    parser.add_argument('--weight_S2', type=str, default='epoch_1103_RCAN_rB6_rG5_S1d_distNorm_015.pth')

    parser.add_argument('--in_format', type=str, default='RGB')
    parser.add_argument('--out_format', type=str, default='RGB')
    parser.add_argument('--scale', type=int, default=3)

    # RCAN Model specifications
    parser.add_argument('--model', default='RCAN', help='model name')
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    parser.add_argument('--pre_train', type=str, default='.', help='pre-trained model directory')
    parser.add_argument('--extend', type=str, default='.', help='pre-trained model directory')
    parser.add_argument('--n_resblocks', type=int, default=6, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=3, help='residual scaling')
    parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'),
                        help='FP precision for test (single | half)')
    parser.add_argument('--n_resgroups', type=int, default=5, help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')

    parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--noise', type=str, default='.',
                        help='Gaussian noise std.')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')

    opt = parser.parse_args()

    parserS1 = argparse.ArgumentParser()
    parserS1.add_argument('--model', default='RCAN',
                        help='model name')

    parserS1.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parserS1.add_argument('--pre_train', type=str, default='.',
                        help='pre-trained model directory')
    parserS1.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parserS1.add_argument('--n_resblocks', type=int, default=6,
                        help='number of residual blocks')
    parserS1.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parserS1.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parserS1.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parserS1.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')
    parserS1.add_argument('--n_resgroups', type=int, default=5, help='number of residual groups')
    parserS1.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')

    parserS1.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    parserS1.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    parserS1.add_argument('--noise', type=str, default='.',  help='Gaussian noise std.')
    parserS1.add_argument('--chop', action='store_true',  help='enable memory-efficient forward')

    optS1 = parserS1.parse_args()

    root_eval_dir = './DualSR_db/validation_set'
    eval_dir_hr = [root_eval_dir + '/hr/']
    eval_dir_lr = [root_eval_dir + '/lr/']

    print(opt.dir_out)
    if not os.path.exists(opt.dir_out):
        os.makedirs(opt.dir_out)

    net_inp_ch = 1 if opt.in_format == 'L' else 3
    net_out_ch = 1 if opt.out_format == 'L' else 3

    print("RF model created")
    #model_S1 = S1Net(ch_in=net_inp_ch, ch_out=net_out_ch, ch=opt.ch, skip=opt.skip, scale=1).to(device)
    #model_SR = SrNet(ch_in=net_inp_ch, ch_out=net_out_ch, ch=opt.ch, skip=opt.skip, scale=opt.scale).to(device)

    # model_S1 = RCAN(args=optS1).to(device)
    # model_SR = RCAN(args=opt).to(device)
    #
    # model_S1.load_state_dict(opt.weight_folder + opt.weight_S1)
    # model_SR.load_state_dict(opt.weight_folder + opt.weight_S2)

    model_S1 = torch.load(opt.weight_folder + opt.weight_S1).eval()
    model_SR = torch.load(opt.weight_folder + opt.weight_S2).eval()

    eval_dataset = SR_Dataset_x2(dir_hr=eval_dir_hr, dir_lr=eval_dir_lr, in_img_format=opt.in_format,
                            out_img_format=opt.out_format, transforms=transforms_train, patch_size=opt.patch_size)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=11, shuffle=False, num_workers=opt.workers,
                            pin_memory=True, drop_last=True)

    print("Data loading completed")

    # print("Opening sheet")
    # sheet = open_sheet()

    epoch_loss_S1 = AverageMeter()
    epoch_loss_S2 = AverageMeter()
    epoch_loss_Distill = AverageMeter()

    n_mix = 0

    if (os.path.isfile(opt.log_file) == True):
        os.remove(opt.log_file)
    file = open(opt.log_file, 'a')

    psnr_avg = 0
    ssim_avg = 0
    num_f = 0
    init = 0

    model_S1.eval()
    model_SR.eval()

    avg_psnr = test_model_SR(model_S1, model_SR, device, eval_dataloader, 0, 0)


    def test_model_SR(model_NR, model_SR, device, dataloader, epoch, idx):
        # psnr = PSNR()
        # ssim = SSIM()

        avg_psnr = 0
        # avg_ssim = 0
        test_cnt = 0
        with torch.no_grad():
            model_SR.eval()
            model_NR.eval()
            for i, (lr, hr) in enumerate(dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                s1, feature_out_s1 = model_NR(lr)
                sr, feature_out_s2 = model_SR(s1)

                batch = lr.shape[0]
                for j in range(batch):
                    if ((epoch == 0) & (idx == 0)):
                        save_image(hr.data[j], f'./result/images/img{i * batch + j:02d}_hr.png')
                        save_image(lr.data[j], f'./result/images/img{i * batch + j:02d}_lr.png')
                    save_image(sr.data[j], f'./result/images/img{i * batch + j:02d}_ep{epoch:03d}_sr.png')
                    save_image(s1.data[j], f'./result/images/img{i * batch + j:02d}_ep{epoch:03d}_s1.png')
                    avg_psnr += psnr(hr.data[j], sr.data[j])
                    # avg_ssim += ssim(hr.data[j], sr.data[j])
                    test_cnt += 1
        return avg_psnr / test_cnt


    # sendSpreadsheet(sheet, avg_psnr, epoch, idx)
    model_SR.train()
    model_S1.train()
    line = str('psnr: ref={:2.3f}'.format(avg_psnr))
    print(line)
    file.write(line)
    file.write('\n')

file.close()
