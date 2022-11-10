import torch
import torchvision
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import os
import numpy as np

from collections.abc import Sequence
from torchvision import transforms
from PIL import Image
import cv2


# from psnr import PSNR
# from pytorch_ssim import SSIM


class SelfDistLoss():
    def __init__(self):
        super(SelfDistLoss, self).__init__()

    def forward(self, HQ, LQ, net):
        self_dist_loss = 0
        quant = []
        quant.append(net.quant_in.to('cuda').eval())
        quant = torch.nn.ModuleList(quant)

        blocks = []
        blocks.append(net.features[0].to('cuda').eval())
        blocks.append(net.features[1].to('cuda').eval())
        blocks.append(net.features[2].to('cuda').eval())
        blocks.append(net.features[3].to('cuda').eval())
        blocks = torch.nn.ModuleList(blocks)

        for i, block in enumerate(quant):
            LQ_re, LQ_si = block(LQ, None)
            HQ_re, HQ_si = block(HQ, None)

        for i, block in enumerate(blocks):
            LQ_re, LQ_si = block(LQ_re, LQ_si)
            HQ_re, HQ_si = block(HQ_re, HQ_si)
            self_dist_loss += torch.nn.functional.l1_loss(LQ_re, HQ_re)
        return self_dist_loss


def print_minmax_err(input_ts, scale, z_point, msg):
    print(msg)
    print(tensor_to_numpy(torch.min((input_ts / scale) + z_point)),
          tensor_to_numpy(torch.max((input_ts / scale) + z_point)))
    return


def print_round_err(input_ts, scale, msg):
    print(msg)
    print(
        tensor_to_numpy(
            torch.min((input_ts / scale) -
                      (torch.round(input_ts / scale)).float())),
        tensor_to_numpy(
            torch.max((input_ts / scale) -
                      (torch.round(input_ts / scale)).float())))
    return


def print_params(params):
    for param in params:
        print(param.shape)
        try:
            print(param.grad.shape, '\n')
        except:
            print('no grad', '\n')

    return


## imread, resize : PIL.Image
## imsave (16bit) : cv2


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def criterion_tv(sr):
    loss_tv = (torch.sum(torch.abs(sr[:, :, :, :-1] - sr[:, :, :, 1:])) +
               torch.sum(torch.abs(sr[:, :, :-1, :] - sr[:, :, 1:, :])))
    return loss_tv


def criterion_lr(lr, sr, F_down, F_criterion=nn.L1Loss()):
    # print(lr.shape)
    # print(sr.shape)
    sr_down = F_down(sr)
    # print(sr_down.shape)
    # exit(0)

    return F_criterion(lr, sr_down)


class ContentLoss(torch.nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True).to("cuda").eval()
        for parameters in self.feature_extractor.parameters():
            parameters.required_grad = False
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406].view(1, 3, 1, 1)))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225].view(1, 3, 1, 1)))

    def forward(self, sr, hr):
        if sr.shape[1] != 3:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))
        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        # blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda:0").features[:4].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda:0").features[4:9].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda:0").features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda:0").features[:16].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).to("cuda:0").features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class CriterionPerceptual:

    def __init__(self, net, gamma, F_criterion=nn.L1Loss()):
        self.pnet = net
        self.gamma = gamma
        self.F_criterion = F_criterion

    def __call__(self, hr, sr):
        feat_hr = self.pnet(hr)
        feat_sr = self.pnet(sr)
        if not len(self.gamma) == len(feat_hr):
            print(f"net_vgg({len(feat_hr)}) and gamma({len(self.gamma)})")
            exit(-1)

        loss = 0.
        for i in range(len(feat_hr)):
            if (self.gamma[i] > 0.):
                loss += self.F_criterion(feat_hr[i], feat_sr[i]) * self.gamma[i]
        return loss


# def criterion_p(hr, sr, net_vgg, gamma=[1., 1., 1., 1., 1.], F_criterion = nn.L1Loss()):
#     feat_hr = net_vgg(hr)
#     feat_sr = net_vgg(sr)
#     if not len(gamma) == len(feat_hr):
#         print(f"net_vgg({len(feat_hr)}) and gamma({len(gamma)})")
#         exit(-1)
#     loss = 0.
#     for i in range(len(feat_hr)):
#         if (gamma[i]>0.):
#             loss += F_criterion(feat_hr[i], feat_sr[i])*gamma[i]
#     return loss


def criterion_style(feat_hr, feat_sr, criterion=nn.L1Loss()):
    gram_hr = gram_matrix(feat_hr)
    gram_sr = gram_matrix(feat_sr)

    return criterion(gram_hr, gram_sr)


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            'Input tensor should be a torch tensor. Got {}.'.format(
                type(tensor)))

    if tensor.ndim < 3:
        raise ValueError(
            'Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
            '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'
                .format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def tensor_to_numpy(input):
    return input.cpu().detach().numpy()


def imread(dir, mode='RGB', scale=1.):
    # return type / range : float [0,1]
    img = Image.open(dir)
    img = img.resize((img.size[0] * scale, img.size[1] * scale),
                     resample=Image.BICUBIC)
    img = np.array(img) / 255.
    if mode == 'RGB':
        pass
    elif mode == 'YCbCr':
        img = csc_rgb2ycbcr(img)
    elif mode == 'L':
        img = csc_rgb2ycbcr(img)[:, :, 0]
    else:
        print("MODE", mode, "is not valid!!")
        return -1
    return img


def imsave(name, arr, mode='RGB', bits=16):
    arr = np.clip(arr, 0., 1.)
    val_max = (2. ** bits) - 1.
    val_half = 2. ** (bits - 1.)

    if mode == 'RGB':
        arr = arr[:, :, (2, 1, 0)]
        if bits == 16:
            return cv2.imwrite(name, (arr * val_max).astype(np.uint16))
        elif bits == 8:
            return cv2.imwrite(name, (arr * val_max).astype(np.uint8))
        else:
            print("check bits to save")
            return -1
    elif mode == 'YCbCr':
        arr = csc_ycbcr2rgb(arr)
        arr = arr[:, :, (2, 1, 0)]
        if bits == 16:
            return cv2.imwrite(name, (arr * val_max).astype(np.uint16))
        elif bits == 8:
            return cv2.imwrite(name, (arr * val_max).astype(np.uint8))
        else:
            print("check bits to save")
            return -1
    elif mode == 'L':
        if bits == 16:
            return cv2.imwrite(name, (arr * val_max).astype(np.uint16))
        elif bits == 8:
            return cv2.imwrite(name, (arr * val_max).astype(np.uint8))
        else:
            print("check bits to save")
            return -1
    else:
        print("imsave: check mode")
        return -1
    return -1


# def csc_bgr2ycbcr(bgr, std='bt709'):
#     ## PQE CSC, NOT USED
#     wgt = np.zeros([3, 3])
#     ofs = np.zeros([3, 1])
#
#     h, w, c = bgr.shape
#
#     if std == 'bt709':
#         wgt[0, :] = [ 0.1821,  0.6124,  0.0618]
#         wgt[1, :] = [-0.1003, -0.3376,  0.4379]
#         wgt[2, :] = [ 0.4379, -0.3978, -0.0402]
#     elif std == 'bt2020':
#         wgt[0, :] = [ 0.2250,  0.5806,  0.0508]
#         wgt[1, :] = [-0.1223, -0.3156,  0.4379]
#         wgt[2, :] = [ 0.4379, -0.4027, -0.0352]
#
#     ofs[0] = 64.
#     ofs[1] = 512.
#     ofs[2] = 512.
#
#     rgb = bgr[:, :, (2, 1, 0)]
#     rgb10 = rgb.astype(float)*4
#
#     rsh_rgb10 = np.reshape(np.transpose(rgb10, (2, 0, 1)), (c, h*w))
#     rsh_ycc10 = np.around(np.dot(wgt, rsh_rgb10) + ofs)
#     rsh_ycc10 = np.clip(rsh_ycc10, 64., 940.)
#     ycc10 = np.transpose(np.reshape(rsh_ycc10.astype(int), (c, h, w)), (1, 2, 0))
#
#     return ycc10
#
#
# def csc_ycbcr2bgr(ycc10, std='bt709'):
#     ## PQE CSC, NOT USED
#     wgt = np.zeros([3, 3])
#     ofs = np.zeros([3, 1])
#     if std == 'bt709':
#         wgt[0, :] = [1.1678,  0.0000,  1.7980]
#         wgt[1, :] = [1.1678, -0.2139, -0.5345]
#         wgt[2, :] = [1.1678,  2.1186,  0.0000]
#     elif std == 'bt2020':
#         wgt[0, :] = [1.1678,  0.0000,  1.6836]
#         wgt[1, :] = [1.1678, -0.1879, -0.6523]
#         wgt[2, :] = [1.1678,  2.1481,  0.0000]
#
#     ofs[0] = 64.
#     ofs[1] = 512.
#     ofs[2] = 512.
#
#     h, w, c = ycc10.shape
#
#     rsh_ycc10 = np.reshape(np.transpose(ycc10, (2, 0, 1)), (c, h*w))
#     rsh_rgb10 = np.around(np.dot(wgt, (rsh_ycc10 - ofs)))
#     rsh_rgb10 = np.clip(rsh_rgb10, 0, 1023.)
#     rgb10 = np.transpose(np.reshape(rsh_rgb10.astype(int), (c, h, w)), (1, 2, 0))
#     bgr10 = rgb10[:, :, (2, 1, 0)]
#     bgr = (np.around(bgr10//4)).astype(np.uint8)
#
#     return bgr


def csc_rgb2ycbcr(rgb, std='bt709'):
    ## float[0,1] to float[0,1]
    wgt = np.zeros([3, 3])
    ofs = np.zeros([3, 1])

    h, w, c = rgb.shape

    if std == 'bt709':
        wgt[0, :] = [0.1821, 0.6124, 0.0618]
        wgt[1, :] = [-0.1003, -0.3376, 0.4379]
        wgt[2, :] = [0.4379, -0.3978, -0.0402]
    elif std == 'bt2020':
        wgt[0, :] = [0.2250, 0.5806, 0.0508]
        wgt[1, :] = [-0.1223, -0.3156, 0.4379]
        wgt[2, :] = [0.4379, -0.4027, -0.0352]

    ofs[0] = 64.
    ofs[1] = 512.
    ofs[2] = 512.

    rgb10 = rgb.astype(float) * 1023.

    rsh_rgb10 = np.reshape(np.transpose(rgb10, (2, 0, 1)), (c, h * w))
    rsh_ycc10 = np.around(np.dot(wgt, rsh_rgb10) + ofs)
    rsh_ycc10 = np.clip(rsh_ycc10, 64., 940.)
    ycc = np.transpose(np.reshape(rsh_ycc10 / 1023., (c, h, w)), (1, 2, 0))

    return ycc


def csc_ycbcr2rgb(ycc, std='bt709'):
    ## float[0,1] to float[0,1]

    ycc10 = ycc * 1023.
    wgt = np.zeros([3, 3])
    ofs = np.zeros([3, 1])
    if std == 'bt709':
        wgt[0, :] = [1.1678, 0.0000, 1.7980]
        wgt[1, :] = [1.1678, -0.2139, -0.5345]
        wgt[2, :] = [1.1678, 2.1186, 0.0000]
    elif std == 'bt2020':
        wgt[0, :] = [1.1678, 0.0000, 1.6836]
        wgt[1, :] = [1.1678, -0.1879, -0.6523]
        wgt[2, :] = [1.1678, 2.1481, 0.0000]

    ofs[0] = 64.
    ofs[1] = 512.
    ofs[2] = 512.

    h, w, c = ycc10.shape

    rsh_ycc10 = np.reshape(np.transpose(ycc10, (2, 0, 1)), (c, h * w))
    rsh_rgb10 = np.around(np.dot(wgt, (rsh_ycc10 - ofs)))
    rsh_rgb10 = np.clip(rsh_rgb10, 0, 1023.)
    rgb10 = np.transpose(np.reshape(rsh_rgb10, (c, h, w)), (1, 2, 0))
    rgb = rgb10 / 1023.

    return rgb


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_mean=(0.5, 0.5, 0.5), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        print(self.weight.data.size())
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.requires_grad = False


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeter_tensor(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, momentum=0.1, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return

    def update(self, val):
        self.val = val.cpu().detach().numpy()
        # self.val = (1. - self.momentum) * self.val + self.momentum * val
        self.count += 1
        self.avg = (1. - self.momentum) * self.avg + self.momentum * self.val
        # print(self.val)
        # print(self.avg)
        return


def summary_model(model, prekey=''):
    tmpstr = ''  # model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        if len(module._modules) > 0:
            modstr = summary_model(module, prekey + '.' + key)
        else:
            modstr = '{:30s} : {}\n'.format(prekey + '.' + key,
                                            module.__repr__())
        tmpstr += modstr
    return tmpstr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)

# def load_model(model_file):
#     model = MobileNetV2()
#     state_dict = torch.load(model_file)
#     model.load_state_dict(state_dict)
#     model.to('cpu')
#     return model


from torchvision.utils import save_image

eval_images = [
    '024',
    '027',
    '028',
    '1071',
    '1078',
    '145',
    '2182',
    '2187',
    '281',
    '286',
    '355',
    '4102',
    '4109',
    '4335',
    '4339',
    '5033',
    '5061',
    '582',
    '9005',
    '9008',
    '930',
    '932',
]

import math

def psnr(imgA, imgB):
    rmse = torch.mean((imgA-imgB)**2)
    if rmse == 0:
        return 100
    return -10 * math.log10(rmse)

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


# Send spreadsheet
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# from gspread_formatting import set_row_height
# from gspread_formatting import set_column_width

#
# def getToday():
#     now = time.localtime()
#     # ret = f"{(now.tm_year%100):02d}_"
#     ret = f"{now.tm_mon:02d}_{now.tm_mday:02d}_{now.tm_hour:02d}{now.tm_min:02d}{now.tm_sec:02d}"
#     return ret
#
#
# def open_sheet():
#     scope = [
#         'https://spreadsheets.google.com/feeds',
#         'https://www.googleapis.com/auth/drive',
#     ]
#     json_file_name = './web/key_parksaem82.json'
#     credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
#     gc = gspread.authorize(credentials)
#     spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1Q0UWSaFntGywHBUYwi8qG_vwlNBBOE1cmO7VqSL_U-8/edit#gid=0'
#     doc = gc.open_by_url(spreadsheet_url)
#     today = getToday()
#     sheet = doc.add_worksheet(title=today, rows="100", cols="20")
#     sheet.append_row(['EPOCH', 'IDX', 'PSNR', 'IMG', "HD_IMG"])
#     # set_row_height(sheet, '2:1000', 512)
#     set_column_width(sheet, 'D:E', 512)
#     return sheet
#
#
# def sendSpreadsheet(sheet, avg_psnr, epoch, idx):
#     url = 'http://147.47.125.234:8888/PycharmProjects/DualSR_InputDistillation/image/'
#     img_str = f'{url}img00_ep{epoch:03d}_iter{idx:05d}_sr.png'
#     # img = f'=image("{img_str}")'
#     # img_ori = f'=image("{url}img00_hr.png")'
#     img = img_str
#     img_ori = f'{url}img00_hr.png'
#     sheet.append_row([epoch, idx, avg_psnr, img, img_ori])
