import os
import io
import random
import glob
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True
import multiprocessing as mp

from utils import imread


class SR_Dataset_x3_hrd(object):
    def __init__(self,
                 dir_lr=None,
                 dir_hr=None,
                 dir_hrd=None,
                 dir_su=None,
                 in_img_format='L',
                 out_img_format='L',
                 transforms=None,
                 patch_size=48):
        self.fn_lrimgs = []
        self.fn_hrimgs = []
        self.fn_hrdimgs = []
        self.fn_suimgs = []

        if dir_hr is not None:
            for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hr, ext)))
        if dir_hrd is not None:
            for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                self.fn_hrdimgs += sorted(glob.glob('{}/*.{}'.format(dir_hrd, ext)))
        if dir_lr is not None:
            for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lr, ext)))

        self.out_img_format = out_img_format
        self.in_img_format = in_img_format
        self.patch_size = patch_size
        self.transforms = transforms
        self.random_rotate = [0, 90, 180, 270]

    def __getitem__(self, idx):

        img_lr = imread(self.fn_lrimgs[idx], mode=self.in_img_format)
        img_hr = imread(self.fn_hrimgs[idx], mode=self.out_img_format)
        img_hrd = imread(self.fn_hrdimgs[idx], mode=self.out_img_format)

        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, -1)
        if len(img_hr.shape) == 2:
            img_hr = np.expand_dims(img_hr, -1)
        if len(img_hrd.shape) == 2:
            img_hrd = np.expand_dims(img_hrd, -1)
        if len(img_su.shape) == 2:
            img_su = np.expand_dims(img_su, -1)

        img_lr = torch.from_numpy(np.transpose(img_lr, (2, 0, 1))).float()
        img_hr = torch.from_numpy(np.transpose(img_hr, (2, 0, 1))).float()
        img_hrd = torch.from_numpy(np.transpose(img_hrd, (2, 0, 1))).float()

        return img_lr, img_hr, img_hrd

    def __len__(self):
        return len(self.fn_lrimgs)


class TSSR_Dataset_x3(object):
    def __init__(self, dir_lr=None, dir_hr=None, dir_su=None, in_img_format='L', out_img_format='L', transforms=None,
                 patch_size=48):

        self.fn_lrimgs = []
        self.fn_hrimgs = []
        self.fn_suimgs = []

        if dir_hr is not None:
            for dir_hrs in dir_hr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hrs, ext)))
        if dir_lr is not None:
            for dir_lrs in dir_lr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lrs, ext)))
        if dir_su is not None:
            for dir_sus in dir_su:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_suimgs += sorted(glob.glob('{}/*.{}'.format(dir_sus, ext)))

        self.out_img_format = out_img_format
        self.in_img_format = in_img_format
        self.patch_size = patch_size
        self.transforms = transforms
        # self.random_rotate = [0, 90, 180, 270]

        print(f"SR_LR_Data: {dir_lr}, {len(self.fn_lrimgs)}")
        print(f"SR_HR_Data: {dir_hr}, {len(self.fn_hrimgs)}")
        print(f"SR_SU_Data: {dir_su}, {len(self.fn_suimgs)}")

    def __getitem__(self, idx):

        img_lr = Image.open(self.fn_lrimgs[idx]).convert(self.in_img_format)
        img_hr = Image.open(self.fn_hrimgs[idx]).convert(self.out_img_format)
        img_su = Image.open(self.fn_suimgs[idx]).convert(self.out_img_format)

        if self.transforms is not None:
            img_hr = self.transforms(img_hr)
            img_lr = self.transforms(img_lr)
            img_su = self.transforms(img_su)

        return img_lr, img_hr, img_su

    def __len__(self):
        return len(self.fn_lrimgs)


class TSSR_Dataset_x3(object):
    def __init__(self, dir_lr=None, dir_hr=None, dir_su=None, in_img_format='L', out_img_format='L', transforms=None,
                 patch_size=48):

        self.fn_lrimgs = []
        self.fn_hrimgs = []
        self.fn_suimgs = []

        if dir_hr is not None:
            for dir_hrs in dir_hr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hrs, ext)))
        if dir_lr is not None:
            for dir_lrs in dir_lr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lrs, ext)))
        if dir_su is not None:
            for dir_sus in dir_su:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_suimgs += sorted(glob.glob('{}/*.{}'.format(dir_sus, ext)))

        self.out_img_format = out_img_format
        self.in_img_format = in_img_format
        self.patch_size = patch_size
        self.transforms = transforms
        # self.random_rotate = [0, 90, 180, 270]

        print(f"SR_LR_Data: {dir_lr}, {len(self.fn_lrimgs)}")
        print(f"SR_HR_Data: {dir_hr}, {len(self.fn_hrimgs)}")
        print(f"SR_SU_Data: {dir_su}, {len(self.fn_suimgs)}")

    def __getitem__(self, idx):

        img_lr = Image.open(self.fn_lrimgs[idx]).convert(self.in_img_format)
        img_hr = Image.open(self.fn_hrimgs[idx]).convert(self.out_img_format)
        img_su = Image.open(self.fn_suimgs[idx]).convert(self.in_img_format)

        if self.transforms is not None:
            img_hr = self.transforms(img_hr)
            img_lr = self.transforms(img_lr)
            img_su = self.transforms(img_su)

        return img_lr, img_hr, img_su

    def __len__(self):
        return len(self.fn_lrimgs)


class SR_Dataset_x3(object):
    def __init__(self, dir_lr=None, dir_hr=None, dir_su=None, in_img_format='L', out_img_format='L', transforms=None,
                 patch_size=48):

        self.fn_lrimgs = []
        self.fn_hrimgs = []
        self.fn_suimgs = []

        if dir_hr is not None:
            for dir_hrs in dir_hr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hrs, ext)))
        if dir_lr is not None:
            for dir_lrs in dir_lr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lrs, ext)))
        if dir_su is not None:
            for dir_sus in dir_su:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_suimgs += sorted(glob.glob('{}/*.{}'.format(dir_sus, ext)))

        self.out_img_format = out_img_format
        self.in_img_format = in_img_format
        self.patch_size = patch_size
        self.transforms = transforms
        # self.random_rotate = [0, 90, 180, 270]

        print(f"SR_LR_Data: {dir_lr}, {len(self.fn_lrimgs)}")
        print(f"SR_HR_Data: {dir_hr}, {len(self.fn_hrimgs)}")
        print(f"SR_SU_Data: {dir_su}, {len(self.fn_suimgs)}")

    def __getitem__(self, idx):

        img_lr = Image.open(self.fn_lrimgs[idx]).convert(self.in_img_format)
        img_hr = Image.open(self.fn_hrimgs[idx]).convert(self.out_img_format)
        img_su = Image.open(self.fn_suimgs[idx]).convert(self.in_img_format)

        if self.transforms is not None:
            img_hr = self.transforms(img_hr)
            img_lr = self.transforms(img_lr)
            img_su = self.transforms(img_su)

        return img_lr, img_hr, img_su

    def __len__(self):
        return len(self.fn_lrimgs)


class SR_Dataset(object):
    def __init__(self,
                 dir_lr=None,
                 dir_hr=None,
                 dir_su=None,
                 classes=['46', '42', '38', '34', '30', '26'],
                 in_img_format='L',
                 out_img_format='L',
                 transforms=None,
                 patch_size=48):
        self.fn_lrimgs = []
        self.fn_hrimgs = []
        self.fn_suimgs = []

        if dir_hr is not None:
            for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hr, ext)))

        if dir_lr is not None:
            for cls in classes:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lr + cls, ext)))
        if dir_su is not None:
            for cls in classes:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_suimgs += sorted(glob.glob('{}/*.{}'.format(dir_su + cls, ext)))
        self.fn_lrimgs = self.fn_lrimgs
        self.fn_hrimgs = self.fn_hrimgs * len(classes)
        self.fn_suimgs = self.fn_suimgs

        self.out_img_format = out_img_format
        self.in_img_format = in_img_format
        self.patch_size = patch_size
        self.transforms = transforms
        self.random_rotate = [0, 90, 180, 270]
        print(f"SR_LR_Data: {dir_lr}, {len(self.fn_lrimgs)}")
        print(f"SR_HR_Data: {dir_hr}, {len(self.fn_hrimgs)}")
        print(f"SR_SU_Data: {dir_su}, {len(self.fn_suimgs)}")

    def __getitem__(self, idx):
        rot = self.random_rotate[random.randrange(0, 4)]

        img_lr = Image.open(self.fn_lrimgs[idx]).convert(self.in_img_format)
        img_hr = Image.open(self.fn_hrimgs[idx]).convert(self.out_img_format)
        img_su = Image.open(self.fn_suimgs[idx]).convert(self.out_img_format)

        img_lr = img_lr.rotate(rot)
        img_hr = img_hr.rotate(rot)
        img_su = img_su.rotate(rot)

        if self.transforms is not None:
            img_hr = self.transforms(img_hr)
            img_lr = self.transforms(img_lr)
            img_su = self.transforms(img_su)

        return img_lr, img_hr, img_su

    def __len__(self):
        return len(self.fn_lrimgs)


class SR_Dataset_x2(object):
    def __init__(self, dir_lr=None, dir_hr=None, in_img_format='L', out_img_format='L', transforms=None,
                 patch_size=48):

        self.fn_lrimgs = []
        self.fn_hrimgs = []

        if dir_hr is not None:
            for dir_hrs in dir_hr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hrs, ext)))
        if dir_lr is not None:
            for dir_lrs in dir_lr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lrs, ext)))

        self.out_img_format = out_img_format
        self.in_img_format = in_img_format
        self.patch_size = patch_size
        self.transforms = transforms
        # self.random_rotate = [0, 90, 180, 270]

        print(f"SR_LR_Data: {dir_lr}, {len(self.fn_lrimgs)}")
        print(f"SR_HR_Data: {dir_hr}, {len(self.fn_hrimgs)}")

    def __getitem__(self, idx):

        img_lr = None
        img_hr = None
        try:
            img_lr = Image.open(self.fn_lrimgs[idx]).convert(self.in_img_format)
            img_hr = Image.open(self.fn_hrimgs[idx]).convert(self.out_img_format)
        except:
            print(self.fn_lrimgs[idx])
            print(self.fn_hrimgs[idx])
            print("Cannot Get Image")
            exit(1)

        if self.transforms is not None:
            img_hr = self.transforms(img_hr)
            img_lr = self.transforms(img_lr)

        return img_lr, img_hr

    def __len__(self):
        return len(self.fn_lrimgs)


class NR_Dataset(object):
    def __init__(self,
                 dir_lr=None,
                 dir_hr=None,
                 in_img_format='RGB',
                 out_img_format='RGB',
                 transforms=None,
                 patch_size=48):
        self.fn_lrimgs = []
        self.fn_hrimgs = []

        # if dir_hr is not None:
        #     for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
        #         self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hr, ext)))
        # if dir_lr is not None:
        #     for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
        #         self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lr, ext)))

        if dir_hr is not None:
            for dir_hrs in dir_hr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_hrimgs += sorted(glob.glob('{}/*.{}'.format(dir_hrs, ext)))
        if dir_lr is not None:
            for dir_lrs in dir_lr:
                for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
                    self.fn_lrimgs += sorted(glob.glob('{}/*.{}'.format(dir_lrs, ext)))

        self.out_img_format = out_img_format
        self.in_img_format = in_img_format
        self.patch_size = patch_size
        self.transforms = transforms
        self.random_rotate = [0, 0, 0, 0]
        print(f"NR_LR_Data: {dir_lr}, {len(self.fn_lrimgs)}")
        print(f"NR_HR_Data: {dir_hr}, {len(self.fn_hrimgs)}")

    def __getitem__(self, idx):
        rot = self.random_rotate[random.randrange(0, 4)]

        img_lr = Image.open(self.fn_lrimgs[idx]).convert(self.in_img_format)
        img_hr = Image.open(self.fn_hrimgs[idx]).convert(self.out_img_format)

        img_lr = img_lr.rotate(rot)
        img_hr = img_hr.rotate(rot)

        if self.transforms is not None:
            img_hr = self.transforms(img_hr)
            img_lr = self.transforms(img_lr)

        if self.out_img_format == 'RGB':
            img_hr = img_hr[(1, 2, 0), :, :]
            img_lr = img_lr[(1, 2, 0), :, :]

        if random.randint(0, 1):
            img_lr = 1. - img_lr
            img_hr = 1. - img_hr

        # print("LR")
        # print(img_lr.shape)
        # print("HR")
        # print(img_hr.shape)
        # print(torch.min(img_lr))
        # print(torch.max(img_lr))
        # print(torch.min(img_hr))
        # print(torch.max(img_hr))
        # exit(0)
        return img_lr, img_hr

    def __len__(self):
        return len(self.fn_lrimgs)


def crop_patches_(dirs):
    hr_imgs = dirs[0]
    lr_imgs = dirs[1]
    su_imgs = dirs[2]
    print('imgs: ', hr_imgs)
    hr_img = Image.open(hr_imgs)
    lr_img = Image.open(lr_imgs)
    su_img = Image.open(su_imgs)

    out_dir_lr = '/ssd_data/sunguk.lim/div2k_perceptual/patch/lr'
    out_dir_su = '/ssd_data/sunguk.lim/div2k_perceptual/patch/su'
    out_dir_hr = '/ssd_data/sunguk.lim/div2k_perceptual/patch/hr'

    bname = os.path.splitext(os.path.basename(lr_imgs))[0]
    # print(hr_imgs[i])
    # print(lr_imgs[i])
    # print(su_imgs[i])
    # exit(0)

    patch_size = 32
    random_rotate = [0, 90, 180, 270]
    scale = 3

    for j in range(200):
        crop_x = random.randint(0, lr_img.width - patch_size - 10)
        crop_y = random.randint(0, lr_img.height - patch_size - 10)
        rot = random_rotate[random.randrange(0, 4)]
        patch_hr = hr_img.crop((crop_x - patch_size, crop_y - patch_size, crop_x - patch_size + scale * patch_size,
                                crop_y - patch_size + scale * patch_size))
        patch_hr = patch_hr.rotate(rot)

        patch_lr = lr_img.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
        patch_lr = patch_lr.rotate(rot)

        patch_su = su_img.crop(
            (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
        patch_su = patch_su.rotate(rot)

        patch_hr.save(out_dir_hr + f'/{bname}_{j:04d}.png')
        patch_lr.save(out_dir_lr + f'/{bname}_{j:04d}.png')
        patch_su.save(out_dir_su + f'/{bname}_{j:04d}.png')
    # exit(0)


if __name__ == '__main__':
    dir_root = "/ssd_data/sunguk.lim/x2_IV_LRup_temp/lr/"
    lr_imgs = np.sort(glob.glob(dir_root + "lr*jpg"))
    hr = Image.open(dir_root + "hr.png").convert("YCbCr")
    hr = np.array(hr)
    ratios = [4., 3., 2., 1., -0.5, -1., -2., -3.]

    for i in range(len(lr_imgs)):
        lr = Image.open(lr_imgs[i])
        fname = os.path.basename(lr_imgs[i]).split(".")[0]
        su = lr.resize((3840, 2160), Image.BICUBIC)
        # lr = lr.convert("YCbCr")
        su = su.convert("YCbCr")
        # hr = Image.open("./hr.png").convert("YCbCr")

        # lr = np.array(lr)
        su = np.array(su)
        for j in range(len(ratios)):
            sr = hr.copy()
            # print(sr.shape)
            # print(su.shape)
            # print(hr.shape)
            sr[:, :, 0] = np.clip(((su[:, :, 0].astype(np.float32) + ratios[j] * (
                        hr[:, :, 0].astype(np.float32) - su[:, :, 0].astype(np.float32)))), 0., 255.).astype(np.uint8)
            print(f"ch_0: {np.min(sr[:, :, 0])}, {np.max(sr[:, :, 0])}")
            print(f"ch_1: {np.min(sr[:, :, 1])}, {np.max(sr[:, :, 1])}")
            print(f"ch_2: {np.min(sr[:, :, 2])}, {np.max(sr[:, :, 2])}")
            # sr=np.clip(sr, 0,1023)
            sr_img = Image.fromarray(sr, mode="YCbCr")
            sr_img = sr_img.convert("RGB")
            sr_img.save(dir_root + "results/" + fname + "_" + str(ratios[j]) + ".png")

    exit(0)
    ##### make NRLRGAN start
    hr_dir = '/ssd_data/sunguk.lim/data/DIV2K_train_HR'
    lr_dir = '/ssd_data/sunguk.lim/div2k_NRLRGAN/lr'
    su_dir = '/ssd_data/sunguk.lim/div2k_NRLRGAN/su'
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    if not os.path.exists(su_dir):
        os.makedirs(su_dir)
    patch_size = 64
    hr_imgs = np.sort(glob.glob(hr_dir + '/*png'))
    random_rotate = [0, 90, 180, 270]
    for i in range(len(hr_imgs)):
        fname = os.path.basename(hr_imgs[i])
        lr_img = Image.open(hr_imgs[i])
        w, h = lr_img.size
        scale = random.randint(50, 101)
        scale = scale / 100.
        lr_img = lr_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        lr_img = lr_img.resize((w, h), Image.BICUBIC)
        su_img = lr_img.copy()
        su_img = su_img.resize((w // 2, h // 2), Image.BICUBIC)
        print('scale : ', scale, 'fname : ', hr_imgs[i])

        for j in range(400):
            crop_lr_x = random.randint(0, lr_img.width - patch_size)
            crop_lr_y = random.randint(0, lr_img.height - patch_size)
            crop_su_x = random.randint(0, su_img.width - patch_size)
            crop_su_y = random.randint(0, su_img.height - patch_size)
            rot = random_rotate[random.randrange(0, 4)]
            patch_lr = lr_img.crop((crop_lr_x, crop_lr_y, (crop_lr_x + patch_size), (crop_lr_y + patch_size)))
            patch_lr = patch_lr.rotate(rot)

            patch_su = su_img.crop((crop_su_x, crop_su_y, crop_su_x + patch_size, crop_su_y + patch_size))
            patch_su = patch_su.rotate(rot)

            patch_lr.save(lr_dir + '/%04d_%04d.png' % (i, j))
            patch_su.save(su_dir + '/%04d_%04d.png' % (i, j))
            # exit(0)
    exit(0)
    ##### make NRLRGAN end

    # ################ make NR patch start
    # lr_root    = '/ssd_data/sunguk.lim/div2k_nr/DIV2K_train_HR/jpg'
    # hr_img_dir = '/ssd_data/sunguk.lim/div2k_nr/DIV2K_train_HR/*png'
    #
    # out_dir_lr = '/ssd_data/sunguk.lim/div2k_nr/patches/lr'
    # out_dir_hr = '/ssd_data/sunguk.lim/div2k_nr/patches/hr'
    #
    # if not os.path.exists(out_dir_lr):
    #     os.makedirs(out_dir_lr)
    # if not os.path.exists(out_dir_hr):
    #     os.makedirs(out_dir_hr)
    # noise_levels = [30,70]
    # scale = 1
    #
    # hr_imgs = np.sort(glob.glob(hr_img_dir))
    #
    # patch_size = 64
    # random_rotate = [0, 90, 180, 270]
    # # print(len(lr_imgs))
    # # print(len(su_imgs))
    # # print(len(hr_imgs))
    # # exit(0)
    # for i in range(len(hr_imgs)):
    #     print('imgs: ', i)
    #     hr_img = Image.open(hr_imgs[i])
    #     noise_level = random.randint(noise_levels[0], noise_levels[1])
    #     lr_img = Image.open(lr_root + str(noise_level).zfill(3)+'/'+str(i+1).zfill(4)+'.jpg')
    #
    #     # print(hr_imgs[i])
    #     # print(lr_root + str(noise_level).zfill(3)+'/'+str(i+1).zfill(4)+'.jpg')
    #     # print(hr_img)
    #     # print(lr_img)
    #     # exit(0)
    # #
    #     for j in range(500):
    #         crop_x = random.randint(0, lr_img.width - patch_size)
    #         crop_y = random.randint(0, lr_img.height - patch_size)
    #         rot = random_rotate[random.randrange(0, 4)]
    #         patch_hr = hr_img.crop((crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    #         patch_hr = patch_hr.rotate(rot)
    #
    #         patch_lr = lr_img.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    #         patch_lr = patch_lr.rotate(rot)
    #
    #         patch_hr.save(out_dir_hr + '/%04d_%04d.png' % (i,j))
    #         patch_lr.save(out_dir_lr + '/%04d_%04d.png' % (i,j))
    #         # exit(0)
    # exit(0)
    # ################ make NR patch end

    # ################ make SU_x3 start
    # scale = 3
    # # lr_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/720/*png'
    # # su_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/720_x3/'
    # lr_dir = '/ssd_data/sunguk.lim/div2k_nr/DIV2K_train_HR/*png'
    # su_dir = '/ssd_data/sunguk.lim/div2k_perceptual/suimgs/'
    # if not os.path.exists(su_dir):
    #     os.makedirs(su_dir)
    # lr_imgs = np.sort(glob.glob(lr_dir))
    # for i in range(len(lr_imgs)):
    #     fname = os.path.basename(lr_imgs[i])
    #     fname, _ = os.path.splitext(fname)
    #     lr_img = Image.open(lr_imgs[i])
    #     w, h = lr_img.size
    #     su_img = lr_img.resize((w * scale, h * scale), Image.BICUBIC)
    #     su_img.save(su_dir + fname + '.png')
    #     # print('scale : ', scales[j], 'fname : ', lr_imgs[i])
    #     print(su_dir + fname + '.png')
    # exit(0)
    # ################ make SU_x3 end

    # ################ make SU_x4 start
    # scale = 4
    # lr_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/540/*png'
    # su_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/540_x3/'
    # lr_imgs = np.sort(glob.glob(lr_dir))
    # for i in range(len(lr_imgs)):
    #     fname = os.path.basename(lr_imgs[i])
    #     fname, _ = os.path.splitext(fname)
    #     lr_img = Image.open(lr_imgs[i])
    #     w, h = lr_img.size
    #     su_img = lr_img.resize((w * scale, h * scale), Image.BICUBIC)
    #     su_img.save(su_dir + fname + '.png')
    #     # print('scale : ', scales[j], 'fname : ', lr_imgs[i])
    #     print(su_dir + fname + '.png')
    #     # exit(0)
    # ################ make SU_x4 end

    # ################ make perceptual patch x3 start
    lr_img_dir = '/ssd_data/sunguk.lim/div2k_nr/DIV2K_train_HR/*png'
    su_img_dir = '/ssd_data/sunguk.lim/div2k_perceptual/suimgs/*png'
    hr_img_dir = '/ssd_data/sunguk.lim/div2k_nr/DIV2K_train_HR/*png'

    out_dir_lr = '/ssd_data/sunguk.lim/div2k_perceptual/patch/lr'
    out_dir_su = '/ssd_data/sunguk.lim/div2k_perceptual/patch/su'
    out_dir_hr = '/ssd_data/sunguk.lim/div2k_perceptual/patch/hr'

    if not os.path.exists(out_dir_lr):
        os.makedirs(out_dir_lr)
    if not os.path.exists(out_dir_hr):
        os.makedirs(out_dir_hr)
    if not os.path.exists(out_dir_su):
        os.makedirs(out_dir_su)
    # scale = 3

    lr_imgs = np.sort(glob.glob(lr_img_dir))
    su_imgs = np.sort(glob.glob(su_img_dir))
    hr_imgs = np.sort(glob.glob(hr_img_dir))

    dirs = map(list, zip(hr_imgs, lr_imgs, su_imgs))
    dirs = tuple(dirs)
    # print(len(dirs))
    # exit(0)
    with mp.Pool(6) as p:
        p.map(crop_patches_, dirs)
    exit(0)

    # patch_size = 32
    # random_rotate = [0, 90, 180, 270]
    # print(len(lr_imgs))
    # print(len(su_imgs))
    # print(len(hr_imgs))
    # exit(0)
    # for i in range(len(lr_imgs)):
    #     print('imgs: ', i)
    #     hr_img = Image.open(hr_imgs[i])
    #     lr_img = Image.open(lr_imgs[i])
    #     su_img = Image.open(su_imgs[i])
    #     bname = os.path.splitext(os.path.basename(lr_imgs[i]))[0]
    #     # print(hr_imgs[i])
    #     # print(lr_imgs[i])
    #     # print(su_imgs[i])
    #     # exit(0)
    #
    #     for j in range(200):
    #         crop_x = random.randint(0, lr_img.width - patch_size- 10)
    #         crop_y = random.randint(0, lr_img.height - patch_size - 10)
    #         rot = random_rotate[random.randrange(0, 4)]
    #         patch_hr = hr_img.crop((crop_x - patch_size, crop_y - patch_size, crop_x - patch_size + scale*patch_size, crop_y - patch_size + scale*patch_size))
    #         patch_hr = patch_hr.rotate(rot)
    #
    #         patch_lr = lr_img.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    #         patch_lr = patch_lr.rotate(rot)
    #
    #         patch_su = su_img.crop((crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    #         patch_su = patch_su.rotate(rot)
    #
    #         patch_hr.save(out_dir_hr + '/{}_{}.png' % (bname,j))
    #         patch_lr.save(out_dir_lr + '/{}_{}.png' % (bname,j))
    #         patch_su.save(out_dir_su + '/{}_{}.png' % (bname,j))
    #         # exit(0)
    # ################ make perceptual patch x3 end

    # # ################ make patch x3 start
    # lr_img_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/720/*png'
    # su_img_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/720_x3/*png'
    # hr_img_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/IV_result/*png'
    #
    # out_dir_lr = '/ssd_data/sunguk.lim/o22_sr_c_model/patches/x3/lr'
    # out_dir_su = '/ssd_data/sunguk.lim/o22_sr_c_model/patches/x3/su'
    # out_dir_hr = '/ssd_data/sunguk.lim/o22_sr_c_model/patches/x3/hr'
    #
    # if not os.path.exists(out_dir_lr):
    #     os.makedirs(out_dir_lr)
    # if not os.path.exists(out_dir_hr):
    #     os.makedirs(out_dir_hr)
    # if not os.path.exists(out_dir_su):
    #     os.makedirs(out_dir_su)
    # scale = 3
    #
    # lr_imgs = np.sort(glob.glob(lr_img_dir))
    # su_imgs = np.sort(glob.glob(su_img_dir))
    # hr_imgs = np.sort(glob.glob(hr_img_dir))
    #
    # patch_size = 64
    # random_rotate = [0, 90, 180, 270]
    # # print(len(lr_imgs))
    # # print(len(su_imgs))
    # # print(len(hr_imgs))
    # # exit(0)
    # for i in range(len(lr_imgs)):
    #     print('imgs: ', i)
    #     hr_img = Image.open(hr_imgs[i])
    #     lr_img = Image.open(lr_imgs[i])
    #     su_img = Image.open(su_imgs[i])
    #     bname = os.path.splitext(os.path.basename(lr_imgs[i]))[0]
    #     # print(hr_imgs[i])
    #     # print(lr_imgs[i])
    #     # print(su_imgs[i])
    #     # exit(0)
    #
    #     for j in range(500):
    #         crop_x = random.randint(0, lr_img.width - patch_size)
    #         crop_y = random.randint(0, lr_img.height - patch_size)
    #         rot = random_rotate[random.randrange(0, 4)]
    #         patch_hr = hr_img.crop((crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    #         patch_hr = patch_hr.rotate(rot)
    #
    #         patch_lr = lr_img.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    #         patch_lr = patch_lr.rotate(rot)
    #
    #         patch_su = su_img.crop((crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    #         patch_su = patch_su.rotate(rot)
    #
    #         patch_hr.save(out_dir_hr + '/%04d_%04d.png' % (i,j))
    #         patch_lr.save(out_dir_lr + '/%04d_%04d.png' % (i,j))
    #         patch_su.save(out_dir_su + '/%04d_%04d.png' % (i,j))
    #         # exit(0)
    # # ################ make patch x3 end
    # # ################ make patch x4 start
    # lr_img_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/540/*png'
    # su_img_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/540_x4/*png'
    # hr_img_dir = '/ssd_data/sunguk.lim/o22_sr_c_model/IV_result/*png'
    #
    # out_dir_lr = '/ssd_data/sunguk.lim/o22_sr_c_model/patches/x4/lr'
    # out_dir_su = '/ssd_data/sunguk.lim/o22_sr_c_model/patches/x4/su'
    # out_dir_hr = '/ssd_data/sunguk.lim/o22_sr_c_model/patches/x4/hr'
    #
    # scale = 4
    #
    # lr_imgs = np.sort(glob.glob(lr_img_dir))
    # su_imgs = np.sort(glob.glob(su_img_dir))
    # hr_imgs = np.sort(glob.glob(hr_img_dir))
    #
    # patch_size = 64
    # random_rotate = [0, 90, 180, 270]
    # # print(len(lr_imgs))
    # # print(len(su_imgs))
    # # print(len(hr_imgs))
    # # exit(0)
    # for i in range(len(lr_imgs)):
    #     print('imgs: ', i)
    #     hr_img = Image.open(hr_imgs[i])
    #     lr_img = Image.open(lr_imgs[i])
    #     su_img = Image.open(su_imgs[i])
    #     # print(hr_imgs[i])
    #     # print(lr_imgs[i])
    #     # print(su_imgs[i])
    #     # exit(0)
    #
    #     for j in range(500):
    #         crop_x = random.randint(0, lr_img.width - patch_size)
    #         crop_y = random.randint(0, lr_img.height - patch_size)
    #         rot = random_rotate[random.randrange(0, 4)]
    #         patch_hr = hr_img.crop((crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    #         patch_hr = patch_hr.rotate(rot)
    #
    #         patch_lr = lr_img.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    #         patch_lr = patch_lr.rotate(rot)
    #
    #         patch_su = su_img.crop((crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    #         patch_su = patch_su.rotate(rot)
    #
    #         patch_hr.save(out_dir_hr + '/%04d_%04d.png' % (i,j))
    #         patch_lr.save(out_dir_lr + '/%04d_%04d.png' % (i,j))
    #         patch_su.save(out_dir_su + '/%04d_%04d.png' % (i,j))
    #         # exit(0)
    # # ################ make patch x4 end
    exit(0)

    # ################ make SU start
    # scales = [sys.argv[1]]
    # # print(scales)
    # # exit(0)
    # # print(len(scales))
    # # exit(0)
    # # scales = ['0.46', '0.42', '0.38', '0.34', '0.30', '0.26']
    # for j in range(len(scales)):
    #     lr_dir = '/ssd_data/sunguk.lim/data/DIV2K_train_LR_' + scales[j] + '_jpg95'
    #     su_dir = '/ssd_data/sunguk.lim/data/DIV2K_train_SU_' + scales[j] + '_jpg95'
    #     if not os.path.exists(su_dir):
    #         os.makedirs(su_dir)
    #
    #     lr_imgs = np.sort(glob.glob(lr_dir + '/*jpg'))
    #     for i in range(len(lr_imgs)):
    #         fname = os.path.basename(lr_imgs[i])
    #         fname, _ = os.path.splitext(fname)
    #         lr_img = Image.open(lr_imgs[i])
    #         w, h = lr_img.size
    #         # lr_img = lr_img.resize((int(w*float(scales[j])), int(h*float(scales[j]))), Image.BICUBIC)
    #         # lr_img = lr_img.resize((w//2, h//2), Image.BICUBIC)
    #         # lr_img.save(lr_dir+'/'+fname)
    #         su_img = lr_img.resize((w * 2, h * 2), Image.BICUBIC)
    #         su_img.save(su_dir + '/' + fname + '.png')
    #         print('scale : ', scales[j], 'fname : ', lr_imgs[i])
    #         if i == 10:
    #             exit(0)
    #         # exit(0)
    # ################ make SU end

    ################ make patches start
    # scales = ['0.46', '0.42', '0.38', '0.34', '0.30', '0.26']
    # in_dir_hr = '/ssd_data/sunguk.lim/data/DIV2K_train_HR'
    # in_dir_lr = []
    # in_dir_su = []
    # for s in range(len(scales)):
    #     in_dir_lr.append('/ssd_data/sunguk.lim/data/DIV2K_train_LR_' + scales[s] + '_jpg95')
    #     in_dir_su.append('/ssd_data/sunguk.lim/data/DIV2K_train_SU_' + scales[s] + '_jpg95')
    #
    # print(in_dir_hr)
    # for i in range(len(scales)):
    #     print(in_dir_lr[i])
    #     print(in_dir_su[i])
    # # exit(0)
    #
    # in_imgs_hr = np.sort(glob.glob(in_dir_hr + '/*png'))
    # in_imgs_lr = []
    # in_imgs_su = []
    # for s in range(len(scales)):
    #     in_imgs_lr.append(np.sort(glob.glob(in_dir_lr[s] + '/*jpg')))
    #     in_imgs_su.append(np.sort(glob.glob(in_dir_su[s] + '/*png')))
    #
    # print(len(in_imgs_hr))
    # for i in range(len(scales)):
    #     print(len(in_imgs_lr[i]))
    #     print(len(in_imgs_su[i]))
    #
    # out_root = '/ssd_data/sunguk.lim/data/patches/'
    # out_dir_hr = out_root + 'hr'
    # if not os.path.exists(out_dir_hr):
    #     os.makedirs(out_dir_hr)
    # out_dir_lr = []
    # out_dir_su = []
    # for s in range(len(scales)):
    #     out_dir_lr.append(out_root + 'lr_' + scales[s])
    #     out_dir_su.append(out_root + 'su_' + scales[s])
    #     if not os.path.exists(out_dir_lr[s]):
    #         os.makedirs(out_dir_lr[s])
    #     if not os.path.exists(out_dir_su[s]):
    #         os.makedirs(out_dir_su[s])

    # print(out_dir_hr)
    # for i in range(len(scales)):
    #     print(out_dir_lr[i])
    #     print(out_dir_su[i])
    # exit(0)
    # patch_size = 64
    # random_rotate = [0, 90, 180, 270]
    # for i in range(len(in_imgs_hr)):
    #     print('imgs: ', i)
    #     hr_img = Image.open(in_imgs_hr[i])
    #     lr_img = []
    #     su_img = []
    #     for s in range(len(scales)):
    #         lr_img.append(Image.open(in_imgs_lr[s][i]))
    #         su_img.append(Image.open(in_imgs_su[s][i]))
    #     for j in range(100):
    #         crop_x = random.randint(0, lr_img[0].width - patch_size)
    #         crop_y = random.randint(0, lr_img[0].height - patch_size)
    #         rot = random_rotate[random.randrange(0, 4)]
    #
    #         patch_hr = hr_img.crop((crop_x * 2, crop_y * 2, (crop_x + patch_size) * 2, (crop_y + patch_size) * 2))
    #         patch_hr = patch_hr.rotate(rot)
    #         patch_hr.save(out_dir_hr + '/%04d_%04d.png' % (i, j))
    #         patch_lr = []
    #         patch_su = []
    #         for s in range(len(scales)):
    #             patch_lr.append(lr_img[s].crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size)))
    #             patch_su.append(
    #                 su_img[s].crop((crop_x * 2, crop_y * 2, (crop_x + patch_size) * 2, (crop_y + patch_size) * 2)))
    #             patch_lr[s] = patch_lr[s].rotate(rot)
    #             patch_su[s] = patch_su[s].rotate(rot)
    #             patch_lr[s].save(out_dir_lr[s] + '/%04d_%04d.png' % (i, j))
    #             patch_su[s].save(out_dir_su[s] + '/%04d_%04d.png' % (i, j))
    ################ make patches end

    # scales = ['0.46', '0.42', '0.38', '0.34', '0.30', '0.26']
    # for j in range(len(scales)):
    #     hr_dir = '/ssd_data/sunguk.lim/data/DIV2K_train_HR'
    #     lr_dir = '/ssd_data/sunguk.lim/data/DIV2K_train_LR_'+scales[j]
    #     su_dir = '/ssd_data/sunguk.lim/data/DIV2K_train_SU_'+scales[j]
    #     if not os.path.exists(lr_dir):
    #         os.makedirs(lr_dir)
    #     if not os.path.exists(su_dir):
    #         os.makedirs(su_dir)
    #
    #     hr_imgs = np.sort(glob.glob(hr_dir+'/*png'))
    #     for i in range(len(hr_imgs)):
    #     # for i in range(1):
    #         fname = os.path.basename(hr_imgs[i])
    #         lr_img = Image.open(hr_imgs[i])
    #         w,h = lr_img.size
    #         lr_img = lr_img.resize((int(w*float(scales[j])), int(h*float(scales[j]))), Image.BICUBIC)
    #         lr_img = lr_img.resize((w//2, h//2), Image.BICUBIC)
    #         lr_img.save(lr_dir+'/'+fname)
    #         su_img = lr_img.resize((w, h), Image.BICUBIC)
    #         su_img.save(su_dir+'/'+fname)
    #         print('scale : ', scales[j], 'fname : ', hr_imgs[i])
    #         # exit(0)
