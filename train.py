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

# LPIPS
# import LPIPS.models.dist_model as dm
#
# model_LPIPS = dm.DistModel()
# model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=True)

# # Setup warnings
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torch.quantization')


def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear

    if reduce:
        return torch.mean(losses)
    else:
        return losses


def L1_Loss(input, target, reduce=True):
    abs_error = torch.abs(input - target)
    if reduce:
        return torch.mean(abs_error)
    else:
        return abs_error


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

# def self_dist_loss(HQ_features, LQ_features):
#     # loss = 0
#     loss
#     # for idx in range(len(HQ_features)):
#     #     loss += F.l1_loss(HQ_features[idx].detach(), LQ_features[idx])
#     return loss

if __name__ == '__main__':
    cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='psnr_log_221026_with_distBN.txt',
                        help='pre-trained model directory')
    parser.add_argument('--dir_hr', type=str, nargs='+')
    parser.add_argument('--dir_hrd', type=str)
    parser.add_argument('--dir_lr', type=str, nargs='+')
    parser.add_argument('--dir_hlr_', type=str, nargs='+')
    parser.add_argument('--S1sr', default='sr', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--in_format', type=str, default='RGB')
    parser.add_argument('--out_format', type=str, default='RGB')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--ch', type=int, default=128)
    parser.add_argument('--skip', type=str, default='on', choices=['on', 'off'])
    parser.add_argument('--perchannel', type=str, choices=['on', 'off'])
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma_tv', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--isquant', type=int, default=0)
    parser.add_argument('--isPerceptual', type=int, default=0)
    parser.add_argument('--q_epoch', type=int, default=25)
    parser.add_argument('--p_weight', type=float)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

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

    torch.manual_seed(opt.seed)
    gamma_lr = 1.

    root_dir = './youtubedb'
    opt.dir_hr = [root_dir + '/hr/']
    opt.dir_lr = [root_dir + '/lr/']
    opt.dir_hlr = [root_dir + '/hlr/']

    root_eval_dir = './test'
    eval_dir_hr = [root_eval_dir + '/hr/']
    eval_dir_lr = [root_eval_dir + '/lr/']

    if opt.perchannel == 'on':
        opt.perchannel = True
    else:
        opt.perchannel = False

    if opt.skip == 'on':
        opt.skip = True
    else:
        opt.skip = False

    if opt.isquant == 1:
        opt.isquant = True
    else:
        opt.isquant = False

    if opt.isPerceptual == 1:
        opt.isPerceptual = True
    else:
        opt.isPerceptual = False

    opt.dir_out = 'result/train/weight_RCAN_distillationBN_img'
    writer = SummaryWriter('runs/weight_RCAN_distillationBN_img')

    print(opt.dir_out)
    if not os.path.exists(opt.dir_out):
        os.makedirs(opt.dir_out)

    #### SR
    opt.isGAN = False
    opt.isPerceptual = True

    gain_hr = 1.0
    gain_dhr = 1.0
    gamma_adv = 0.01
    gamma_tv = 0. / ((opt.patch_size - 1) * opt.patch_size) / 2.

    transforms_train = transforms.Compose([transforms.ToTensor()])
    criterion = nn.L1Loss()

    net_inp_ch = 1 if opt.in_format == 'L' else 3
    net_out_ch = 1 if opt.out_format == 'L' else 3

    print("RF model created")
    #model_S1 = S1Net(ch_in=net_inp_ch, ch_out=net_out_ch, ch=opt.ch, skip=opt.skip, scale=1).to(device)
    #model_SR = SrNet(ch_in=net_inp_ch, ch_out=net_out_ch, ch=opt.ch, skip=opt.skip, scale=opt.scale).to(device)

    model_S1 = RCAN(args=optS1).to(device)
    model_SR = RCAN(args=opt).to(device)

    # opt_model_S1 = optim.Adam(model_S1.parameters(), lr=opt.lr)
    # opt_model_SR = optim.Adam(model_SR.parameters(), lr=opt.lr)

    opt_model_All = optim.Adam(list(model_SR.parameters())+list(model_S1.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(opt_model_All, step_size=3, gamma=0.8)

    dataset = SR_Dataset_x3(dir_hr=opt.dir_hr, dir_lr=opt.dir_lr, dir_su = opt.dir_hlr, in_img_format=opt.in_format,
                            out_img_format=opt.out_format, transforms=transforms_train, patch_size=opt.patch_size)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                            pin_memory=True, drop_last=True)

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

    for epoch in range(opt.num_epochs + 1):
        model_S1.train()
        model_SR.train()
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {:3d}/{}'.format(epoch + 1, opt.num_epochs))
            for idx, (lr, hr, hlr) in enumerate(dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                hlr = hlr.to(device)

                # print(lr[0].size())
                # print(hr[0].size())
                #
                opt_model_All.zero_grad()
                S1, S1_feature = model_S1(lr)

                # Calculate Pixel Loss - Generator

                # Calculate Dist Loss
                # HQ_features = get_features(hlr, model_SR, True)
                # LQ_features = get_features(lr, model_SR, False)
                # loss_Dist = self_dist_loss(HQ_features, LQ_features)
                # loss_Diff = criterion(S1, hlr)
                # loss_S1 = loss_Diff + loss_Dist

                loss_S1 = criterion(S1, hlr)

                sr, LQ_features = model_SR(S1)

                model_SR_copy = RCAN(args=opt).to(device).eval()
                model_SR_copy.state_dict(model_SR.state_dict())

                tsr, HQ_features = model_SR_copy(hlr)

                loss_diff = criterion(sr, hr)
                loss_dist = F.l1_loss(LQ_features.view(LQ_features.size(0), -1), F.normalize(HQ_features.view(HQ_features.size(0), -1)).detach())
                # loss_dist = self_dist_loss(HQ_features, LQ_features)
                # loss_dist = criterion(HQ_features, LQ_features)

                loss_SR = loss_diff
                loss_All = loss_SR + loss_dist + loss_S1

                loss_All.backward()

                torch.nn.utils.clip_grad_norm_(model_SR.parameters(), max_norm=0.01)
                torch.nn.utils.clip_grad_norm_(model_S1.parameters(), max_norm=0.01)
                opt_model_All.step()

                epoch_loss_S1.update(loss_S1.item(), len(lr))
                epoch_loss_S2.update(loss_SR.item(), len(lr))
                epoch_loss_Distill.update(loss_dist.item(), len(lr))
                # epoch_loss_Dist.update(loss_Dist.item(), len(lr))

                if idx%200 == 0:
                    avg_psnr = test_model_SR(model_S1, model_SR, device, eval_dataloader, epoch, idx)
                    # sendSpreadsheet(sheet, avg_psnr, epoch, idx)
                    model_SR.train()
                    model_S1.train()
                    line = str('{:03d} Epoch , {:06d} Idx => psnr: ref={:2.3f}'.format(epoch, idx, avg_psnr))
                    print(line)
                    file.write(line)
                    file.write('\n')

                    torch.save(model_S1.state_dict(),
                               os.path.join(opt.dir_out, f'epoch_RCAN_rB6_rG5_S1_addPerceptual_{(epoch):03d}.pth'))
                    torch.save(model_SR.state_dict(),
                               os.path.join(opt.dir_out, f'epoch_RCAN_rB6_rG5_S2_addPerceptual_{(epoch):03d}.pth'))

                _tqdm.set_postfix_str(s=f'S1: {epoch_loss_S1.avg:.5f}, S2: {epoch_loss_S2.avg:.5f}, Dist: {epoch_loss_Distill.avg:.5f}')
                _tqdm.update(len(lr))

        writer.add_scalar('Stage1.', epoch_loss_S1.avg, epoch + 1)
        writer.add_scalar('Stage2.', epoch_loss_S2.avg, epoch + 1)

        scheduler.step()


        if epoch % 1 == 0:
            torch.save(model_S1.state_dict(), os.path.join(opt.dir_out, f'epoch_RCAN_rB6_rG5_S1_addPerceptual_{(epoch):03d}.pth'))
            torch.save(model_SR.state_dict(), os.path.join(opt.dir_out, f'epoch_RCAN_rB6_rG5_S2_addPerceptual_{(epoch):03d}.pth'))
        writer.close()

file.close()
