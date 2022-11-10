import os
import io
import random
import glob
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
import cv2
import sys
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import multiprocessing as mp
from multiprocessing import Process
from functools import partial

from utils import imread, imsave

random_rotate = [0, 90, 180, 270]


def img2patchesGridNR(dir_lr, dir_hr, dir_out, dbName, num_patches=1100, patch_size=64):
    # directory structure :
    # dir -- lr
    #     |
    #     -- hr
    #     |
    #     -- patch_{dbName} -- lr
    #                       |
    #                       -- hr

    dir_patches_lr = dir_out + f"/patch_{dbName}/lr"
    dir_patches_hr = dir_out + f"/patch_{dbName}/hr"

    if not os.path.exists(dir_patches_lr):
        os.makedirs(dir_patches_lr)
    if not os.path.exists(dir_patches_hr):
        os.makedirs(dir_patches_hr)

    # lr_imgs = np.sort(glob.glob(dir_lr+"/*.png"))
    # hr_imgs = np.sort(glob.glob(dir_hr+"/*.png"))
    lr_imgs = []
    for ext in ['png', 'jpg']:
        lr_imgs += sorted(glob.glob(f'{dir_lr}/*.{ext}'))
    hr_imgs = []
    for ext in ['png', 'jpg']:
        hr_imgs += sorted(glob.glob(f'{dir_hr}/*.{ext}'))

    idxList = range(len(hr_imgs))
    with mp.Pool(6) as p:
        p.map(partial(func_img2patchNRMP, dir=dir_out, dbName=dbName, img_lr_lst=lr_imgs, img_hr_lst=hr_imgs,
                      patch_size=patch_size), idxList)

    # for i in range(len(lr_imgs)):
    #     func_img2patchNR(dir_out, lr_imgs[i], hr_imgs[i], patch_size)

    return


def img2patchesGrid(dir, scale, num_patches, patch_size, ratio=1., ratio_hr=1.):
    # directory structure :
    # dir -- lr
    #     |
    #     -- hr
    #     |
    #     -- su
    #     |
    #     -- patch -- lr
    #              |
    #              -- hr
    #              |
    #              -- su

    dir_lr = dir + "/lr"
    dir_su = dir + "/su"
    dir_hr = dir + "/hr"
    lr_imgs = []
    for ext in ['png', 'jpg']:
        lr_imgs += sorted(glob.glob(f'{dir_lr}/*.{ext}'))
    hr_imgs = []
    for ext in ['png', 'jpg']:
        hr_imgs += sorted(glob.glob(f'{dir_hr}/*.{ext}'))

    dir_lr_r = dir + f"/lr_r{ratio}"
    dir_su_r = dir + f"/su_r{ratio}"
    dir_hr_r = dir + f"/hr_r{ratio_hr}"

    dir_patches_lr = dir + f"/patch_r{ratio}/lr"
    dir_patches_su = dir + f"/patch_r{ratio}/su"
    dir_patches_hr = dir + f"/patch_r{ratio_hr}/hr"

    if not os.path.exists(dir_su):
        os.makedirs(dir_su)

    if not os.path.exists(dir_lr_r):
        os.makedirs(dir_lr_r)
        with mp.Pool(6) as p:
            p.map(partial(func_resize_ratio, su_dir=dir_lr_r, ratio=ratio), lr_imgs)
    if not os.path.exists(dir_hr_r):
        os.makedirs(dir_hr_r)
        with mp.Pool(6) as p:
            p.map(partial(func_resize_ratio, su_dir=dir_hr_r, ratio=ratio_hr), hr_imgs)

    if not os.path.exists(dir_patches_lr):
        os.makedirs(dir_patches_lr)
    if not os.path.exists(dir_patches_su):
        os.makedirs(dir_patches_su)
    if not os.path.exists(dir_patches_hr):
        os.makedirs(dir_patches_hr)

    # lr_imgs = np.sort(glob.glob(dir_lr+"/*.png"))
    # hr_imgs = np.sort(glob.glob(dir_hr+"/*.png"))

    lr_imgs = []
    for ext in ['png', 'jpg']:
        lr_imgs += sorted(glob.glob(f'{dir_lr_r}/*.{ext}'))

    if not os.path.exists(dir_su_r):
        os.makedirs(dir_su_r)
        with mp.Pool(6) as p:
            p.map(partial(func_resize, su_dir=dir_su_r, scale=scale), lr_imgs)

    su_imgs = np.sort(glob.glob(dir_su_r + "/*.png"))
    # lr_imgs = []
    # lr_imgs = np.sort(glob.glob(dir_lr_r+"/*.png"))
    # hr_imgs = []
    # hr_imgs = np.sort(glob.glob(dir_hr_r+"/*.png"))

    hr_imgs = []
    for ext in ['png', 'jpg']:
        hr_imgs += sorted(glob.glob(f'{dir_hr_r}/*.{ext}'))

    # for i in range(len(lr_imgs)):
    #     func_img2patch(dir, lr_imgs[i], su_imgs[i], hr_imgs[i], num_patches, scale, patch_size, ratio=ratio,
    #                    ratio_hr=ratio_hr)

    arglist = range(len(lr_imgs))
    with mp.Pool(6) as p:
        p.map(partial(func_img2patchMP,
                      dir=dir,
                      img_lr_lst=lr_imgs,
                      img_hlr_lst=su_imgs,
                      img_hr_lst=hr_imgs,
                      num_patches=num_patches,
                      scale=scale,
                      patch_size=patch_size,
                      ratio=ratio,
                      ratio_hr=ratio_hr), arglist)
    # exit(0)
    return


def func_resize(dir_lr, su_dir, scale):
    # su_dir = '/ssd_data/sunguk.lim/div2k_perceptual/suimgs_test/'
    # scale = 3
    fname = os.path.basename(dir_lr)
    fname, _ = os.path.splitext(fname)
    lr_img = Image.open(dir_lr)
    w, h = lr_img.size

    # su_img = lr_img
    # su_img = lr_img.convert("YCbCr").resize((int(w * scale), int(h * scale)), Image.BICUBIC).convert("RGB")
    su_img = lr_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    # su_img = np.asarray(lr_img)
    # su_img = cv2.resize(su_img, dsize=None, fx=int(scale), fy=int(scale), interpolation = cv2.INTER_CUBIC)
    # su_img = Image.fromarray(su_img)

    su_img.save(su_dir + '/' + fname + '.png')
    print(f"img saved :{su_dir + '/' + fname + '.png'}")
    # exit(0)
    return


def func_resize_ratio(dir_lr, su_dir, ratio):
    # su_dir = '/ssd_data/test/'
    # scale = 3
    fname = os.path.basename(dir_lr)
    fname, _ = os.path.splitext(fname)
    lr_img = Image.open(dir_lr)
    w, h = lr_img.size

    # su_img = lr_img
    # su_img = lr_img.convert("YCbCr").resize((int(w * scale), int(h * scale)), Image.BICUBIC).convert("RGB")
    # su_img = lr_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    if ratio > 1.0:
        su_img = lr_img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)
    elif ratio == 1:
        su_img = lr_img
    else:
        su_img = lr_img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC).resize((int(w), int(h)), Image.BICUBIC)
    # su_img = np.asarray(lr_img)
    # su_img = cv2.resize(su_img, dsize=None, fx=int(scale), fy=int(scale), interpolation = cv2.INTER_CUBIC)
    # su_img = Image.fromarray(su_img)

    su_img.save(su_dir + '/' + fname + '.png')
    print(f"img saved :{su_dir + '/' + fname + '.png'}")
    # exit(0)
    return


def func_img2patchNR(dir, img_lr, img_hr, patch_size):
    dir_patches_lr = dir + f"/patch/lr"
    dir_patches_hr = dir + f"/patch/hr"

    lr_img = Image.open(img_lr)
    hr_img = Image.open(img_hr)

    fname_lr = os.path.splitext(os.path.basename(img_lr))[0]
    fname_hr = os.path.splitext(os.path.basename(img_hr))[0]
    print(img_lr)
    print(img_hr)

    # ### random patch start
    # for i in range(num_patches):
    #     if not i % 100:
    #         print(f'{i}_th patch done')
    #     patch_lr, patch_hlr, patch_hr = func_crop_patch(dir, lr_img, su_img, hr_img, patch_size=patch_size, scale=scale)
    #     patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:06d}.png')
    #     patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:06d}.png')
    #     patch_hlr.save(dir_patches_su + f'/{fname_su}_{i:06d}.png')
    # ### random patch end

    ### grid patch start
    w, h = lr_img.size
    dx = w // patch_size
    dy = h // patch_size
    i = 0
    for x in range(dx):
        for y in range(dy):
            patch_lr = lr_img.crop(
                (int(x * patch_size), int(y * patch_size), int((x + 1) * patch_size), int((y + 1) * patch_size)))
            patch_hr = hr_img.crop(
                (int(x * patch_size), int(y * patch_size), int((x + 1) * patch_size), int((y + 1) * patch_size)))
            patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:05d}.png')
            patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:05d}.png')
            i += 1
    ### grid patch end
    return


def func_img2patch(dir, img_lr, img_hlr, img_hr, num_patches, scale, patch_size, ratio, ratio_hr):
    # dir_patches_lr = dir+"/patch/lr"
    # dir_patches_su = dir+"/patch/su"
    # dir_patches_hr = dir+"/patch/hr"
    dir_patches_lr = dir + f"/patch_r{ratio}/lr"
    dir_patches_su = dir + f"/patch_r{ratio}/su"
    dir_patches_hr = dir + f"/patch_r{ratio_hr}/hr"

    lr_img = Image.open(img_lr)
    su_img = Image.open(img_hlr)
    hr_img = Image.open(img_hr)

    fname_lr = os.path.splitext(os.path.basename(img_lr))[0]
    fname_su = os.path.splitext(os.path.basename(img_hlr))[0]
    fname_hr = os.path.splitext(os.path.basename(img_hr))[0]
    print(img_lr)
    print(img_hlr)
    print(img_hr)

    # ### random patch start
    # for i in range(num_patches):
    #     if not i % 100:
    #         print(f'{i}_th patch done')
    #     patch_lr, patch_hlr, patch_hr = func_crop_patch(dir, lr_img, su_img, hr_img, patch_size=patch_size, scale=scale)
    #     patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:06d}.png')
    #     patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:06d}.png')
    #     patch_hlr.save(dir_patches_su + f'/{fname_su}_{i:06d}.png')
    # ### random patch end

    ### grid patch start
    w, h = lr_img.size
    dx = w // patch_size
    dy = h // patch_size
    i = 0
    for x in range(dx):
        for y in range(dy):
            patch_lr = lr_img.crop(
                (int(x * patch_size), int(y * patch_size), int((x + 1) * patch_size), int((y + 1) * patch_size)))
            patch_hr = hr_img.crop((int(x * patch_size * scale), int(y * patch_size * scale),
                                    int((x + 1) * patch_size * scale), int((y + 1) * patch_size * scale)))
            patch_hlr = su_img.crop((int(x * patch_size * scale), int(y * patch_size * scale),
                                    int((x + 1) * patch_size * scale), int((y + 1) * patch_size * scale)))
            patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:05d}.png')
            patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:05d}.png')
            patch_hlr.save(dir_patches_su + f'/{fname_su}_{i:05d}.png')
            i += 1

    # with mp.Pool(8) as p:
    #     p.map(partial(func_crop_patch, patch_size = patch_size, scale = scale), lr_imgs)

    ### grid patch end
    return


def func_img2patchMP(idx, dir, img_lr_lst, img_hlr_lst, img_hr_lst, num_patches, scale, patch_size, ratio, ratio_hr):
    # dir_patches_lr = dir+"/patch/lr"
    # dir_patches_su = dir+"/patch/su"
    # dir_patches_hr = dir+"/patch/hr"

    img_lr = img_lr_lst[idx]
    img_hlr = img_hlr_lst[idx]
    img_hr = img_hr_lst[idx]

    dir_patches_lr = dir + f"/patch_r{ratio}/lr"
    dir_patches_su = dir + f"/patch_r{ratio}/su"
    dir_patches_hr = dir + f"/patch_r{ratio_hr}/hr"

    lr_img = Image.open(img_lr)
    su_img = Image.open(img_hlr)
    hr_img = Image.open(img_hr)

    fname_lr = os.path.splitext(os.path.basename(img_lr))[0]
    fname_su = os.path.splitext(os.path.basename(img_hlr))[0]
    fname_hr = os.path.splitext(os.path.basename(img_hr))[0]
    print(img_lr)
    print(img_hlr)
    print(img_hr)

    # ### random patch start
    # for i in range(num_patches):
    #     if not i % 100:
    #         print(f'{i}_th patch done')
    #     patch_lr, patch_hlr, patch_hr = func_crop_patch(dir, lr_img, su_img, hr_img, patch_size=patch_size, scale=scale)
    #     patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:06d}.png')
    #     patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:06d}.png')
    #     patch_hlr.save(dir_patches_su + f'/{fname_su}_{i:06d}.png')
    # ### random patch end

    ### grid patch start
    w, h = lr_img.size
    dx = w // patch_size
    dy = h // patch_size
    i = 0
    for x in range(dx):
        for y in range(dy):
            patch_lr = lr_img.crop(
                (int(x * patch_size), int(y * patch_size), int((x + 1) * patch_size), int((y + 1) * patch_size)))
            patch_hr = hr_img.crop((int(x * patch_size * scale), int(y * patch_size * scale),
                                    int((x + 1) * patch_size * scale), int((y + 1) * patch_size * scale)))
            patch_hlr = su_img.crop((int(x * patch_size * scale), int(y * patch_size * scale),
                                    int((x + 1) * patch_size * scale), int((y + 1) * patch_size * scale)))
            patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:05d}.png')
            patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:05d}.png')
            patch_hlr.save(dir_patches_su + f'/{fname_su}_{i:05d}.png')
            i += 1

    # with mp.Pool(8) as p:
    #     p.map(partial(func_crop_patch, patch_size = patch_size, scale = scale), lr_imgs)

    ### grid patch end
    return


def func_img2patchNRMP(idx, dir, dbName, img_lr_lst, img_hr_lst, patch_size):
    # dir_patches_lr = dir + f"/patch/lr"
    # dir_patches_hr = dir + f"/patch/hr"
    dir_patches_lr = dir + f"/patch_{dbName}/lr"
    dir_patches_hr = dir + f"/patch_{dbName}/hr"

    img_lr = img_lr_lst[idx]
    img_hr = img_hr_lst[idx]

    lr_img = Image.open(img_lr)
    hr_img = Image.open(img_hr)

    fname_lr = os.path.splitext(os.path.basename(img_lr))[0]
    fname_hr = os.path.splitext(os.path.basename(img_hr))[0]
    print(img_lr)
    print(img_hr)

    # ### random patch start
    # for i in range(num_patches):
    #     if not i % 100:
    #         print(f'{i}_th patch done')
    #     patch_lr, patch_hlr, patch_hr = func_crop_patch(dir, lr_img, su_img, hr_img, patch_size=patch_size, scale=scale)
    #     patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:06d}.png')
    #     patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:06d}.png')
    #     patch_hlr.save(dir_patches_su + f'/{fname_su}_{i:06d}.png')
    # ### random patch end

    ### grid patch start
    w, h = lr_img.size
    dx = w // patch_size
    dy = h // patch_size
    i = 0
    for x in range(dx):
        for y in range(dy):
            patch_lr = lr_img.crop(
                (int(x * patch_size), int(y * patch_size), int((x + 1) * patch_size), int((y + 1) * patch_size)))
            patch_hr = hr_img.crop(
                (int(x * patch_size), int(y * patch_size), int((x + 1) * patch_size), int((y + 1) * patch_size)))
            patch_lr.save(dir_patches_lr + f'/{fname_lr}_{i:05d}.png')
            patch_hr.save(dir_patches_hr + f'/{fname_hr}_{i:05d}.png')
            i += 1
    ### grid patch end
    return


def sel_patch(patch_in):
    patch = patch_in.convert("YCbCr")
    psize = patch.size[0]
    patch = np.asarray(patch)
    sobel = cv2.Sobel(patch[:, :, 0], -1, 1, 1, ksize=3)
    # print(patch.shape)
    # print(f"{sobel.shape}, {np.min(sobel)}, {np.max(sobel)}, {np.count_nonzero(sobel>20)}")

    if np.count_nonzero(sobel > 10) > 0.1 * psize * psize:
        return True
    else:
        return False


def func_crop_patch(dir, img_lr, img_hlr, img_hr, patch_size, scale):
    # sel_edge = True
    sel_edge = False

    if sel_edge:
        flag_stop = False
        while not flag_stop:
            crop_x = random.randint(0, img_lr.width - patch_size)
            crop_y = random.randint(0, img_lr.height - patch_size)

            patch_hr = img_hr.crop(
                (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
            res_sel = sel_patch(patch_hr)

            if res_sel:
                flag_stop = True
    else:
        crop_x = random.randint(0, img_lr.width - patch_size)
        crop_y = random.randint(0, img_lr.height - patch_size)

    rot = random_rotate[random.randrange(0, 4)]

    patch_hr = img_hr.crop(
        (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    patch_hr = patch_hr #.rotate(rot)

    patch_hlr = img_hlr.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    patch_hlr = patch_hlr #.rotate(rot)

    patch_lr = img_lr.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    patch_lr = patch_lr #.rotate(rot)

    return patch_lr, patch_hlr, patch_hr


def CropPatch(dir, img_lr, img_hlr, img_hr, patch_size, scale):
    # sel_edge = True
    sel_edge = False

    if sel_edge:
        flag_stop = False
        while not flag_stop:
            crop_x = random.randint(0, img_lr.width - patch_size)
            crop_y = random.randint(0, img_lr.height - patch_size)

            patch_hr = img_hr.crop(
                (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
            res_sel = sel_patch(patch_hr)

            if res_sel:
                flag_stop = True
    else:
        crop_x = random.randint(0, img_lr.width - patch_size)
        crop_y = random.randint(0, img_lr.height - patch_size)

    rot = random_rotate[random.randrange(0, 4)]
    patch_hr = img_hr.crop(
        (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    patch_hr = patch_hr.rotate(rot)

    patch_hlr = img_hlr.crop(
        (crop_x * scale, crop_y * scale, (crop_x + patch_size) * scale, (crop_y + patch_size) * scale))
    patch_hlr = patch_hlr.rotate(rot)

    patch_lr = img_lr.crop((crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
    patch_lr = patch_lr.rotate(rot)

    return patch_lr, patch_hlr, patch_hr


def func_210308():
    dir_hr = "/ssd_data/sunguk.lim/x2_IV/hr_4k/"
    dir_lr = "/ssd_data/sunguk.lim/x2_IV/lr_hd/"

    out_lr = "/ssd_data/sunguk.lim/x2_IV/lr_2k/"
    out_hr = "/ssd_data/sunguk.lim/x2_IV/hr_1440p/"

    imgs_lr = np.sort(glob.glob(dir_lr + "*png"))
    imgs_hr = np.sort(glob.glob(dir_hr + "*png"))

    for i in range(len(imgs_lr)):
        img = Image.open(imgs_lr[i])
        fname = os.path.basename(imgs_lr[i]).split(".")[0]
        img = img.resize((1920, 1080), Image.BICUBIC)
        out_name = out_lr + fname + ".png"
        img.save(out_name)
        print(f"{out_name}")
        # break

    for i in range(len(imgs_hr)):
        img = Image.open(imgs_hr[i])
        fname = os.path.basename(imgs_hr[i]).split(".")[0]
        img = img.resize((2560, 1440), Image.BICUBIC)
        out_name = out_hr + fname + ".png"
        img.save(out_name)
        print(f"{out_name}")
        # break

    return


def genNRDB(dir_img, dbName, numCore, ratio, debug=0):
    # img_in = "/home/sunguk.lim/work/O22_sr_ci_pytorch/img/c_inputs/2k/FHD_2/*.png"
    # img_in = "/hdd/dns9_h0/sunguk.lim/share/210310_sr_dnshp/*.png"
    # img_in = "/ssd_data/sunguk.lim/x2_IV/div2k_b/hr/0002.png"

    dirs = np.sort(glob.glob(f"{dir_img}/*.png"))
    dir_out = os.path.dirname(dir_img) + f"/{dbName}"
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
        print(f"{dir_out} created")

    if debug == 0:
        with mp.Pool(numCore) as p:
            p.map(partial(pqeBlur, dir_out=dir_out, ratio=ratio), dirs)
    else:
        pqeBlur(dirs[int(debug - 1)], dir_out=dir_out, ratio=ratio)
        return

    return dir_out


def pqeBlur(img_dir, dir_out, ratio=1.):
    img = cv2.imread(img_dir)
    fname = os.path.splitext(os.path.basename(img_dir))[0]

    # ## median
    # img_out = cv2.medianBlur(img, 3)
    # out_dir = f"{dir_out}/{fname}_median.png"
    # cv2.imwrite(out_dir, img_out)
    # return

    #### bilateral
    # d = [3, 5, 7, 9]
    # sc = [3, 5, 10, 15, 20]
    # ss = [5, 25, 50, 75]
    # img_out = cv2.bilateralFilter(img, d=d[3], sigmaColor = sc[4], sigmaSpace = ss[3])

    # ### blur_r52_guided_210311 start
    # img_blur = Image.open(img_dir)
    # w,h = img_blur.size
    # ratio = 0.52
    # img_blur = img_blur.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC).resize((int(w), int(h)), Image.BICUBIC)
    # img_blur = np.array(img_blur)
    # img_blur = img_blur[:, :, (2, 1, 0)]
    # # print(img_blur.shape)
    # # print(img.shape)
    # # print((img_blur[:,:,(2,1,0)] == img).all())
    #
    # img_median = cv2.medianBlur(img, 5)
    #
    # img_src = img
    # img_guide = img_median
    #
    # EPS = 0.00001*255*255
    # img_out = cv2.ximgproc.guidedFilter(guide = img_guide, src = img_src, radius = 3, eps = EPS)
    # out_dir = f"{dir_out}/{fname}_pqeblur.png"
    # cv2.imwrite(out_dir, img_out)
    #
    # # EPS = [0.000001, 0.0001, 0.01, 1., 10., 100., 1000]
    # # for i in range(len(EPS)):
    # #     img_out = cv2.ximgproc.guidedFilter(guide = img_guide, src = img_src, radius = 15, eps = EPS[i])
    # #     out_dir = f"{dir_out}/{fname}_pqeblur_eps{EPS[i]}.png"
    # #     cv2.imwrite(out_dir, img_out)
    #
    # # try:
    # #     guide_dir = f"{dir_out}/{fname}_guide.png"
    # #     cv2.imwrite(guide_dir, img_guide)
    # #     src_dir = f"{dir_out}/{fname}_source.png"
    # #     cv2.imwrite(src_dir, img_src)
    # # except:
    # #     pass
    # ### blur_r52_guided_210311 end

    # ### blur_r52_guided_210312 start
    # img_out = cv2.medianBlur(img,9)
    # out_dir = f"{dir_out}/{fname}_pqeblur.png"
    # cv2.imwrite(out_dir, img_out)
    # # cv2.imwrite(f"/home/sunguk.lim/svn/adv/dev/SR/O22/branches/sunguk_perceptual_dns6/res/test/temp/result/{fname}_median.png", img_out)
    # # img_ll = Image.open(img_dir)
    # # w,h = img_ll.size
    # # ratio = 0.50
    # # img_50 = img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC).resize((int(w), int(h)), Image.NEAREST)
    #
    # ### blur_r52_guided_210312 end

    ## simple blur start
    img_blur = Image.open(img_dir)
    w, h = img_blur.size
    out_ratio = 1.
    w_out, h_out = (w * out_ratio, h * out_ratio)
    img_blur = img_blur.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC).resize((int(w_out), int(h_out)),
                                                                                       Image.BICUBIC)
    # img_blur = img_blur.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC).resize((int(w), int(h)), Image.BICUBIC)
    out_dir = f"{dir_out}/{fname}_{ratio}.png"
    img_blur.save(out_dir)
    ## simple blur end

    print(f"{out_dir} saved")

    return


if __name__ == '__main__':
    ## DEBUG=1
    ## dir_hr = "/ssd_data/sunguk.lim/x2_IV/srnr/hr_r100"
    ##
    ## dbName = f"blur_test"
    ## dir_out = genNRDB(dir_img = dir_hr, dbName = dbName, numCore = 6, ratio = 1., debug=DEBUG)
    ## exit(0)

    # ### make blurred NRDB start
    # DEBUG=0
    # dir_hr = "/ssd_data/sunguk.lim/x2_IV/srnr/hr_r100"
    #
    # ratio_lr = 0.5
    # ratio_hr = 1.0
    #
    # dbName = f"lr_{ratio_lr}"
    # dir_out = genNRDB(dir_img = dir_hr, dbName = dbName, numCore = 6, ratio = ratio_lr, debug=DEBUG)
    #
    # # dbName = f"hr_{ratio_hr}"
    # # dir_out = genNRDB(dir_img = dir_hr, dbName = dbName, numCore = 6, ratio = ratio_hr, debug=DEBUG)
    #
    # if not ratio_hr == 1.0:
    #     dbName = f"hr_{ratio_hr}"
    #     dir_out = genNRDB(dir_img = dir_hr, dbName = dbName, numCore = 6, ratio = ratio_hr, debug=DEBUG)
    # exit(0)

    ### make NR DB
    # ratio_lr = 0.5
    # ratio_hr = 1.0
    # dir_root="/ssd_data/sunguk.lim/x2_IV/srnr"
    # dir_lr=dir_root+f"/lr_{ratio_lr}"
    #
    # ### HR auto start
    # if not ratio_hr == 1.0:
    #     dir_hr=dir_root+f"/hr_{ratio_hr}"
    #     dbName = f"nrdb_lr{ratio_lr}_hr{ratio_hr}"
    # else:
    #     dir_hr=dir_root+f"/hr_r100"
    #     dbName = f"nrdb_lr{ratio_lr}_hr100"
    # ### HR auto end

    dir_lr = "/ssd_data/sunguk.lim/x2_IV/srnr/lr_0.5_jpg80"
    dir_hr = "/ssd_data/sunguk.lim/x2_IV/srnr/hr_r100"

    dir_out = "/ssd_data/sunguk.lim/x2_IV/srnr"
    dbName = "nrdb_lr50jpg80_hr100"
    img2patchesGridNR(dir_lr=dir_lr, dir_hr=dir_hr, dir_out=dir_out, dbName=dbName, num_patches=1100, patch_size=64)
    exit(0)
    ### make blurred NRDB end

    # img2patchesGridNR(dir_lr = dir_out, dir_hr = dir_hr, dir_out = dir_out, dbName = dbName, patch_size=64)
    # exit(0)

    # # ### make SR DB
    # # dir_db = '/ssd_data/sunguk.lim/x2_IV/div2k'
    # # dir_db = '/ssd_data/sunguk.lim/x2_IV/r52_biliteral'
    # dir_db = '/ssd_data/sunguk.lim/x2_IV/r100'
    # img2patchesGrid(dir_db, scale=2, num_patches=1100, patch_size=64, ratio=1., ratio_hr=1.)
    # # dir_out = genNRDB(dir_img = dir_hr, dbName = dbName, numCore = 6, debug=False)
    # # img2patchesGrid(dir_lr = dir_out, dir_hr = dir_hr, dir_out = dir_out, dbName = dbName, patch_size=64)
    # exit(0)

    # # ### make NR DB
    # dir_root="/ssd_data/sunguk.lim/x2_IV/srnr"
    # dir_lr=dir_root+f"/lr_0.6_jpg80"
    # dir_hr=dir_root+f"/hr_r100"
    # dbName = f"nrdb_lr60jpg80_hr100"
    # dir_out = "/ssd_data/sunguk.lim/x2_IV/srnr"
    # img2patchesGridNR(dir_lr = dir_lr, dir_hr = dir_hr, dir_out = dir_out, dbName = dbName, num_patches=1100, patch_size=64)
    # # dir_out = genNRDB(dir_img = dir_hr, dbName = dbName, numCore = 6, debug=False)
    # # img2patchesGrid(dir_lr = dir_out, dir_hr = dir_hr, dir_out = dir_out, dbName = dbName, patch_size=64)
    # exit(0)

    # dir_db = '/ssd_data/sunguk.lim/x2_IV/div2k'
    # img2patchesGrid(dir_db, scale=2, num_patches=1100, patch_size=64, ratio=0.52, ratio_hr=0.52)

    # dir_hr = "/ssd_data/sunguk.lim/x2_IV/div2k/hr/"
    # dir_lr = "/ssd_data/sunguk.lim/x2_IV/div2k/lr/"
    # dir_hrs = np.sort(glob.glob(dir_hr+"*.png"))
    # print(len(dir_hrs))
    # for i in range(len(dir_hrs)):
    #     hr = Image.open(dir_hrs[i])
    #     fname = os.path.basename(dir_hrs[i]).split(".")[0]
    #     w,h = hr.size
    #     lr = hr.resize((w//2, h//2), Image.BICUBIC)
    #     lr.save(dir_lr+fname+".png")
    #     print(i)
    # exit(0)