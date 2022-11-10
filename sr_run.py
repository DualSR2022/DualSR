import argparse
import os
import io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as pil_image
import glob
import numpy as np
import cv2
from utils import imread, imsave

from model.layer import ConvBiasRelu, QuantInOut, ResBlk
from model.sr_qat import SrNet, NrNet

cudnn.benchmark = False
bits_out = 16  ## 8 or 16
val_max = (2. ** bits_out) - 1.
val_half = 2. ** (bits_out - 1.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_format', type=str, default='L')
    parser.add_argument('--in_format', type=str, default='L')
    parser.add_argument('--out_format', type=str, default='L')
    parser.add_argument('--skip', type=str, choices=['on', 'off'])
    parser.add_argument('--perchannel', type=str, choices=['on', 'off'])
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, default='./result/image', required=True)
    parser.add_argument('--dir_lr', type=str, required=True)
    parser.add_argument('--dir_param', type=str)
    parser.add_argument('--extern_su', type=int, default=0)
    parser.add_argument('--epoch', type=str)
    parser.add_argument('--dir_su', type=str)
    parser.add_argument('--isquant', type=int, default=0)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--ch', type=int, default=16)
    parser.add_argument('--nrsr', default='sr', type=str)
    parser.add_argument('--dump_bin', type=int, default=0)




    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_hr', type=str, nargs='+')
    parser.add_argument('--dir_hrd', type=str)
    parser.add_argument('--dir_lr', type=str, nargs='+')
    parser.add_argument('--dir_hlr_', type=str, nargs='+')
    parser.add_argument('--nrsr', default='sr', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--in_format', type=str, default='L')
    parser.add_argument('--out_format', type=str, default='L')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--ch', type=int, default=512)
    parser.add_argument('--skip', type=str, default='on', choices=['on', 'off'])
    parser.add_argument('--perchannel', type=str, choices=['on', 'off'])
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
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
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)


    opt, unknown = parser.parse_known_args()

    if opt.extern_su == 1:
        opt.extern_su = True
    else:
        opt.extern_su = False

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

    if opt.dump_bin == 1:
        opt.dump_bin = True
    else:
        opt.dump_bin = False

    print(opt.isquant)

    net_inp_ch = 1 if opt.img_format == 'L' else 3

    lrnames = [os.path.join(opt.dir_lr, file) for file in os.listdir(opt.dir_lr) if
               file.endswith(("ppm", "jpeg", "png", "jpg", "bmp"))]
    lrnames = sorted(lrnames)
    # if opt.extern_su:
    #     # opt.dir_su = "/home1/sunguk.lim/input/sd/dns5/sux4_rgb"
    #     # opt.dir_su = "/home1/sunguk.lim/input/fhd/mnr/sux2rgb"
    #     sunames = [os.path.join(opt.dir_su, file) for file in os.listdir(opt.dir_su) if file.endswith(("ppm", "jpeg", "png", "jpg", "bmp"))]
    #     sunames = sorted(sunames)

    if not os.path.exists(opt.outputs_dir):
         os.makedirs(opt.outputs_dir)

    net_inp_ch = 1 if opt.in_format == 'L' else 3
    net_out_ch = 1 if opt.out_format == 'L' else 3


    model_NR = torch.load('./runs/epoch_NR_006.pth')
    model_SR = torch.load('./runs/epoch_SR_006.pth')

    model_NR.eval()
    model_SR.eval()

    model_name = os.path.split(os.path.split(opt.weights_path)[0])[1]
    gain = 1.0
    print(f"gain: {gain}")
    if opt.nrsr == 'nr':  ## NR inference
        for idx, lrname in enumerate(lrnames):
            print("file is", lrname)
            img_name = os.path.splitext(os.path.basename(lrname))[0]

            if opt.dump_bin:
                def apply_exportparams(m):
                    if (isinstance(m, QuantInOut) or isinstance(m, ResBlk)):
                        m.bin_root = opt.dir_param + img_name + '/'
                        m.dump_bin = True
                    if (isinstance(m, ConvBiasRelu)):
                        m.conv_layer.bin_root = opt.dir_param + img_name + '/'
                        m.conv_layer.dump_bin = True
                        m.bias_layer.bin_root = opt.dir_param + img_name + '/'
                        m.bias_layer.dump_bin = True
                        m.quant_layer.bin_root = opt.dir_param + img_name + '/'
                        m.quant_layer.dump_bin = True


                model.apply(apply_exportparams)
                if not os.path.exists(opt.dir_param + img_name):
                    os.makedirs(opt.dir_param + img_name)

            lr_ycbcr = imread(lrname, mode=opt.in_format)

            # print(lr_ycbcr.shape)
            if opt.extern_su:
                suname = sunames[idx]
                su_ycbcr = imread(suname, mode='YCbCr')
            else:
                su_ycbcr = imread(lrname, mode='YCbCr', scale=opt.scale)
            sr_ycbcr = imread(lrname, mode='YCbCr', scale=opt.scale)
            sr_ycbcr2 = imread(lrname, mode='YCbCr', scale=opt.scale)

            lr_input = lr_ycbcr
            su_input = su_ycbcr[:, :, 0]

            lr_input = transforms.ToTensor()(lr_input).unsqueeze(0).to(device).float()
            su_input = transforms.ToTensor()(su_input).unsqueeze(0).to(device).float()

            sr_out_path0 = os.path.join(opt.outputs_dir,
                                        '{}_{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name,
                                                                 opt.epoch, "g06"))

            re_out_path = os.path.join(opt.outputs_dir,
                                       '{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name, "re"))
            su_out_path = os.path.join(opt.outputs_dir, '{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], "su"))
            with torch.no_grad():
                pred_sr, pred_re = model(lr_input)

            pred_re = (pred_re[0, :, :, :]).squeeze(0).cpu().numpy()
            pred_re = np.clip(pred_re, -0.5, 0.5)
            # re_output = pred_re + 0.5
            # imsave(re_out_path, re_output, mode=opt.out_format, bits=bits_out)

            pred_sr = (pred_sr[0, :, :, :]).squeeze(0).cpu().numpy()

            sr_ycbcr[:, :, 0] = sr_ycbcr[:, :, 0] + pred_re * 0.6
            sr_ycbcr = np.clip(sr_ycbcr, 0., 1.)
            imsave(sr_out_path0, sr_ycbcr, mode='YCbCr', bits=bits_out)
            #
            # # lr_img = imread(lrname, mode=opt.in_format)
            # # sr = imread(lrname, mode=opt.in_format)
            # # if opt.in_format == 'RGB':
            # #     lr_input = lr_img[:,:,(1,2,0)] # RGB to GBR
            # # elif opt.in_format == 'YCbCr':
            # #     lr_input = lr_img
            # # else:
            # #     print("MODE", opt.in_format, "is not valid!!")
            # #     exit(-1)
            #
            # lr_input = transforms.ToTensor()(lr_input).unsqueeze(0).to(device).float()
            #
            # sr_out_path = os.path.join(opt.outputs_dir, '{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name, "sr"))
            # re_out_path = os.path.join(opt.outputs_dir, '{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name, "re"))
            # lr_out_path = os.path.join(opt.outputs_dir, '{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name, "lr"))
            # with torch.no_grad():
            #     pred_sr, pred_re = model(lr_input)
            # pred_re = (pred_re[0,:,:,:]).squeeze(0).cpu().numpy()
            # pred_sr = (pred_sr[0,:,:,:]).squeeze(0).cpu().numpy()
            # pred_re = np.clip(pred_re, -0.5, 0.5)
            # pred_re = np.transpose(pred_re, (1,2,0)) ## (c,h,w) to (h,w,c)
            # pred_sr = np.transpose(pred_sr, (1,2,0))
            #
            # if opt.in_format == 'RGB':
            #     pred_re = pred_re[:,:,(2,0,1)] # GBR to RGB
            #
            # elif opt.in_format == 'YCbCr':
            #     sr = sr + gain * pred_re
            #     imsave(sr_out_path, sr, mode=opt.out_format, bits=bits_out)
            #     re_output = pred_re + 0.5
            #     imsave(re_out_path, re_output, mode=opt.out_format, bits=bits_out)
            # else:
            #     print("MODE", opt.in_format, "is not valid!!")
            #     exit(-1)
            # # print(sr.shape)
            # # print(pred_re.shape)
            # sr = sr + gain * pred_re
            # imsave(sr_out_path, sr, mode=opt.out_format, bits=bits_out)
            # re_output = pred_re + 0.5
            # imsave(re_out_path, re_output, mode=opt.out_format, bits=bits_out)
            # imsave(lr_out_path, lr_img, mode=opt.out_format, bits=bits_out)
            exit(0)

    else:  ## SR inference
        for idx, lrname in enumerate(lrnames):
            if opt.dump_bin and idx:
                exit(0)
            print("file is", lrname)

            img_name = os.path.splitext(os.path.basename(lrname))[0]

            if opt.dump_bin:
                def apply_exportparams(m):
                    if (isinstance(m, QuantInOut) or isinstance(m, ResBlk)):
                        m.bin_root = opt.dir_param + img_name + '/'
                        m.dump_bin = True
                    if (isinstance(m, ConvBiasRelu)):
                        m.conv_layer.bin_root = opt.dir_param + img_name + '/'
                        m.conv_layer.dump_bin = True
                        m.bias_layer.bin_root = opt.dir_param + img_name + '/'
                        m.bias_layer.dump_bin = True
                        m.quant_layer.bin_root = opt.dir_param + img_name + '/'
                        m.quant_layer.dump_bin = True


                model.apply(apply_exportparams)
                if not os.path.exists(opt.dir_param + img_name):
                    os.makedirs(opt.dir_param + img_name)

            lr_ycbcr = imread(lrname, mode=opt.in_format)

            # print(lr_ycbcr.shape)
            if opt.extern_su:
                suname = sunames[idx]
                # su_ycbcr = imread(suname, mode='YCbCr')
                su_ycbcr = imread(suname, mode='SUL')
            else:
                su_ycbcr = imread(lrname, mode='YCbCr', scale=opt.scale)
            sr_ycbcr = imread(lrname, mode='YCbCr', scale=opt.scale)

            lr_input = lr_ycbcr
            su_input = su_ycbcr[:, :, 0]

            lr_input = transforms.ToTensor()(lr_input).unsqueeze(0).to(device).float()
            su_input = transforms.ToTensor()(su_input).unsqueeze(0).to(device).float()

            sr_out_path = os.path.join(opt.outputs_dir,
                                       '{}_{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name,
                                                                opt.epoch, "sr"))
            # sr_out_path = os.path.join(opt.outputs_dir, '{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name, "sr_6bT"))
            re_out_path = os.path.join(opt.outputs_dir,
                                       '{}_{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], model_name, "re"))
            su_out_path = os.path.join(opt.outputs_dir, '{}_{}.png'.format(lrname.split("/")[-1].split(".")[0], "su"))
            with torch.no_grad():
                pred_sr, pred_re = model(lr_input, su_input)

            pred_re = (pred_re[0, :, :, :]).squeeze(0).cpu().numpy()
            # pred_re = np.clip(pred_re, -0.5, 0.5)

            # pred_re.tofile('/home/sunguk.lim/temp/residual.bin')
            # print(pred_re.shape)
            # exit(0)
            pred_sr = (pred_sr[0, :, :, :]).squeeze(0).cpu().numpy()
            sr_ycbcr[:, :, 0] = sr_ycbcr[:, :, 0] + gain * pred_re
            # print(np.min(sr_ycbcr[:,:,0]),np.max(sr_ycbcr[:,:,0]))
            # print(np.min(sr_ycbcr[:,:,1]),np.max(sr_ycbcr[:,:,1]))
            # print(np.min(sr_ycbcr[:,:,2]),np.max(sr_ycbcr[:,:,2]))
            sr_ycbcr = np.clip(sr_ycbcr, 0., 1.)
            imsave(sr_out_path, sr_ycbcr, mode='YCbCr', bits=bits_out)
            # re_output = pred_re + 0.5
            # print(np.min(pred_re), np.max(pred_re))
            # imsave(re_out_path, re_output, mode='L', bits=bits_out)

            # if not opt.extern_su :
            #     imsave(su_out_path, su_ycbcr, mode='YCbCr', bits=bits_out)
            exit(0)


#################################################

    dataset = SR_Dataset_x3(dir_hr=opt.dir_hr, dir_lr=opt.dir_lr, dir_su = opt.dir_hlr, in_img_format=opt.in_format,
                            out_img_format=opt.out_format, transforms=transforms_train, patch_size=opt.patch_size)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                            pin_memory=True, drop_last=True)
    print("Data loading completed")

    epoch_loss_NR = AverageMeter()
    epoch_loss_SR = AverageMeter()

    n_mix = 0

    for epoch in range(opt.num_epochs + 1):
        model_NR.train()
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
                opt_model_NR.zero_grad()
                nr = model_NR(lr)

                # Calculate Pixel Loss - Generator

                # Calculate Dist Loss
                # HQ_features = get_features(hr, model, True)
                # LQ_features = get_features(lr, model, False)
                # loss_Dist = self_dist_loss(HQ_features, LQ_features)
                loss_NR = criterion(nr, hlr)

                # loss = loss_NR
                loss_NR.backward()
                opt_model_NR.step()

                opt_model_SR.zero_grad()
                sr = model_SR(hlr)
                loss_SR = criterion(sr, hr)
                loss_SR.backward()

                opt_model_SR.step()

                epoch_loss_NR.update(loss_NR.item(), len(lr))
                epoch_loss_SR.update(loss_SR.item(), len(lr))
                # epoch_loss_Dist.update(loss_Dist.item(), len(lr))

                _tqdm.set_postfix_str(s=f'NR: {epoch_loss_NR.avg:.6f}, SR: {epoch_loss_SR.avg:.6f}')
                _tqdm.update(len(lr))

        writer.add_scalar('NR.', epoch_loss_NR.avg, epoch + 1)
        writer.add_scalar('SR.', epoch_loss_SR.avg, epoch + 1)

        if epoch % 1 == 0:
            torch.save(model_NR.state_dict(), os.path.join(opt.dir_out, f'epoch_NR_{(epoch):03d}.pth'))
            torch.save(model_SR.state_dict(), os.path.join(opt.dir_out, f'epoch_SR_{(epoch):03d}.pth'))
        writer.close()
