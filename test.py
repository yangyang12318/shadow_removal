#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms

import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from models import Generator_F2S

torch.cuda.set_device(3)
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='/mnt/data/YY/copy/demo3/ckpt/netG_S2F',
                    help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/mnt/data/YY/copy/demo3/ckpt/netG_refine',
                    help='B2A generator checkpoint file')
opt = parser.parse_args()

### ISTD
opt.dataroot_A = '/mnt/data/YY/datasets/Rome/img'  # 只有阴影区域
opt.dataroot_B = '/mnt/data/YY/datasets/Rome/mask/mask'  # 残差区域
opt.dataroot_C = '/mnt/data/YY/datasets/Rome/dilmk'  # 残差区域
opt.dataroot_D = '/mnt/data/YY/datasets/Rome/dilmk' #残差区域
opt.im_suf_A = '.png'
opt.im_suf_B = '.png'
opt.im_suf_C = '.png'
opt.im_suf_D = '.png'

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda')

print(opt)

img_transform1 = transforms.Compose([
    #    transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
img_transform3 = transforms.Compose([
    transforms.ToTensor(),
])
###### Definition of variables ######
# Networks
netG_S2F = Generator_F2S()
# style=StyleEncoder(16,norm_layer=nn.InstanceNorm2d)
netG_refine = Generator_F2S()
if opt.cuda:
    netG_S2F = netG_S2F.cuda()
    netG_refine = netG_refine.cuda()
#   style=style.cuda()
# Load state dicts
for i in range(40, 80, 1):
    if i == 0 :
        continue
    netG_S2F.load_state_dict(torch.load(opt.generator_A2B + '_' + str(i) + '.pth', map_location='cuda'))
    # style.load_state_dict(torch.load(opt.style,map_location='cuda'))
    netG_refine.load_state_dict(torch.load(opt.generator_B2A + '_' + str(i) + '.pth', map_location='cuda'))

    # Set model's test mode
    netG_S2F.eval()

    # style.eval()
    netG_refine.eval()

    # Dataset loader

    to_pil = transforms.ToPILImage()

    ###### Testing######

    # Create output dirs if they don't exist
    # if not os.path.exists('output/A'):
    #    os.makedirs('output/A')
    if not os.path.exists('ckpt/Bb' ):
        os.makedirs('ckpt/Bb' )

    ##################################### A to B // shadow to shadow-free
    gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

    # mask_queue = QueueMask(gt_list.__len__())

    # mask_non_shadow = Variable(Tensor(1, 1, opt.size, opt.size).fill_(-1.0), requires_grad=False)

    for idx, img_name in enumerate(gt_list):
        print('predicting: %d / %d' % (idx + 1, len(gt_list)))
        print(img_name)
        # Set model input
        # img_orignal=Image.open(os.path.join(opt.dataroot_D, img_name + opt.im_suf_A)).convert('RGB') #原图
        img = Image.open(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)).convert('RGB')  # 只有阴影区域的图像
        w, h = img.size
        img = (img_transform1(img).unsqueeze(0)).to(device)
        # img_res = Image.open(os.path.join(opt.dataroot_B, img_name + opt.im_suf_A)).convert('RGB')
        mask = Image.open(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B))
        """
        mask_res = Image.open(os.path.join(opt.dataroot_C, img_name + opt.im_suf_C))
        mask_res = (img_transform2(mask_res).unsqueeze(0)).to(device)
        mask_res = mask_res[:, 0, :, :].unsqueeze(0)
        """
        maskn = (img_transform2(mask).unsqueeze(0)).to(device)  ##用来合成最后的图像的掩模图

        mask1 = maskn[:, 0, :, :].unsqueeze(0)  # -1-1的阴影掩模图

        mask2 = (img_transform3(mask).unsqueeze(0)).to(device)
        mask22 = mask2[:, 0, :, :].unsqueeze(0)  # 0-1的阴影掩膜图

        mask_dil = Image.open(os.path.join(opt.dataroot_D, img_name + opt.im_suf_D))
        mask_dil = (img_transform2(mask_dil).unsqueeze(0)).to(device)
        mask_dill = mask_dil[:, 0, :, :].unsqueeze(0)

        # mask1=torch.cat((mask,mask,mask),dim=1)
        # mask_var=torch.cat((mask_var,mask_var,mask_var),dim=1)
        # img_flat=img_var.clone()
        # img_flatted = img_flat.view(-1)  # 正则化后的腐蚀图像
        # mask_flatted=mask_var.view(-1)
        # sd_pixel = (mask_flatted == -1).nonzero()  # 0的区域是非阴影区域

        # imgfinal = img_flatted.index_put_((sd_pixel,), torch.tensor(-1).type_as(img_flatted))  # 将非阴影区域都变成了-1 ，因为模型处理的都是正则化后的数据，所以为了统一，所以要都变为-1
        # imgfinal = imgfinal.resize_(1, 3, w, h)

        temp_B = netG_S2F(img, mask1)  # temp_B是只有阴影区域的

        temp_B = temp_B * mask2 + img * ((mask2 - 1) * (-1))
        all = netG_refine(temp_B, mask_dill)
        """
        input = torch.cat([temp_B, mask1], 1).to('cuda')
        nsd = style(img,((mask1 - 1) * (-1)) )

        nsd_sty_map = nsd.expand([1, 16, temp_B.shape[2], temp_B.shape[3]])
        inputs_c2r = torch.cat([input, nsd_sty_map], 1)
        all = netG_refine(inputs_c2r)
        """

        # 下面根据掩膜的图像来判断选择原始图像还是生成的阴影图吧

        imgfinal = img * ((mask2 - 1) * (-1)) + all * mask2

        imgfinal = 0.5 * (imgfinal.data + 1.0)
        # mask_queue.insert(mask_generator(img_var, temp_B))
        imgfinal = np.array((to_pil(imgfinal.squeeze(0).cpu())))
        Image.fromarray(imgfinal).save('ckpt/Bb/'+str(i)+  img_name + opt.im_suf_A)

        print('Generated images %04d of %04d' % (idx + 1, len(gt_list)))
