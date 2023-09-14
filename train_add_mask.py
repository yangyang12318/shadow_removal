#!/usr/bin/python3
#这个采用最基础的本身一致性和循环一致性，训练60个epoch后使用G_S2F，使用vgg提取和hist直方图来训练style和完善器
#这部分前半部分，既60个epoch是正确的，

#3.24更新：
#把hist和guass的统计进行更改，只计算掩膜区域的信息，
#
# 3.25
# 但好像改错了 进行调整检查

from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from models import Generator_S2F, Discriminator, UnetGenerator, Generator_F2S
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from vggloss import Perceptual_loss as PLloss, load_vgg19
import numpy as np
from Histogram_Loss import RGBuvHistBlock
from PIL import Image
# training set:
from datasets_shadow import ImageDataset
from gussian import *


def set_requires_grad(model, bool):
  for p in model.parameters():
    p.requires_grad = bool

def laplacian_op(x, kernel=None):
  if kernel is None:
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    channels = x.size()[1]
    kernel = torch.tensor(laplacian,
                          dtype=torch.float32).unsqueeze(0).expand(
      1, channels, 3, 3).to(device=torch.cuda.current_device())
  return F.conv2d(x, kernel, stride=1, padding=1)


def sobel_op(x, dir=0, kernel=None):
  if kernel is None:
    if dir == 0:
      sobel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]  # x
    elif dir == 1:
      sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # y
    channels = x.size()[1]
    kernel = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(
      1, channels, 3, 3).to(device=torch.cuda.current_device())
  return F.conv2d(x, kernel, stride=1, padding=1)


class reconstruction_loss(object):
  def __init__(self, loss):
    self.loss = loss
    if self.loss == '1st gradient':
      sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]  # x
      sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # y
      sobel_x = torch.tensor(
        sobel_x, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      sobel_y = torch.tensor(
        sobel_y, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      self.kernel1 = sobel_x
      self.kernel2 = sobel_y
    elif self.loss == '2nd gradient':
      laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
      self.kernel1 = torch.tensor(
        laplacian, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      self.kernel2 = None
    else:
      self.kernel1 = None
      self.kernel2 = None

  def compute_loss(self, input, target):
    if self.loss == 'L1':
      reconstruction_loss = torch.mean(torch.abs(input - target))  # L1

    elif self.loss == '1st gradient':
      input_dfdx = sobel_op(input, kernel=self.kernel1)
      input_dfdy = sobel_op(input, kernel=self.kernel2)
      target_dfdx = sobel_op(target, kernel=self.kernel1)
      target_dfdy = sobel_op(target, kernel=self.kernel2)
      input_gradient = torch.sqrt(torch.pow(input_dfdx, 2) +
                                  torch.pow(input_dfdy, 2))
      target_gradient = torch.sqrt(torch.pow(
        target_dfdx, 2) + torch.pow(target_dfdy, 2))
      reconstruction_loss = torch.mean(torch.abs(
        input_gradient - target_gradient))  # L1

    elif self.loss == '2nd gradient':
      input_lap = laplacian_op(input, kernel=self.kernel1)
      target_lap = laplacian_op(target, kernel=self.kernel1)
      reconstruction_loss = torch.mean(torch.abs(input_lap - target_lap))  # L1
    else:
      reconstruction_loss = None

    return reconstruction_loss
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=15,help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--dataroot', type=str, default='/mnt/data/YY/datasets/AISD',help='the data source path to read information')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--resume', action='store_true', help='resume')
parser.add_argument('--iter_loss', type=int, default=500, help='average loss for n iterations')
parser.add_argument('--pretrained_model', type=str, default='/mnt/data/yy/copy/demo1/ckpt1', help='pretrained data path')
opt = parser.parse_args()

if not os.path.exists('ckpt1'):
    os.mkdir('ckpt1')
opt.log_path = os.path.join('ckpt1', str(datetime.datetime.now()) + '.txt')  # 创建了一个以时间命名的txt来存储信息

if torch.cuda.is_available():
    opt.cuda = True

print(opt)

###### Definition of variables ######
# Networks
netG_S2F = Generator_F2S()  # 阴影到非阴影
netG_F2S = Generator_F2S()  # 非阴影到阴影 3, 3,8, 64, nn.InstanceNorm2d, True,use_attention=True
#style=StyleEncoder(16,norm_layer=nn.InstanceNorm2d)
netG_refine = Generator_F2S()  # 平滑阴影
netD_B = Discriminator()  # 鉴别是不是阴影区域
histBlock = RGBuvHistBlock(insz=400, h=64,method='inverse-quadratic',resizing='interpolation',sigma=0.02)
gaussKernel = get_gaussian_kernel(kernel_size=15, sigma=5, channels=3).to('cuda')
# netD_A = Discriminator()  #鉴别是不是非阴影区域
set_requires_grad(histBlock, True)
if opt.cuda:
    netG_S2F = netG_S2F.cuda()
    netG_F2S = netG_F2S.cuda()
    netG_refine = netG_refine.cuda()
    #style=style.cuda()
    netD_B = netD_B.cuda()


netG_S2F.apply(weights_init_normal)
netG_F2S.apply(weights_init_normal)
netG_refine.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
#style.apply(weights_init_normal)


# Lossess
criterion_GAN = torch.nn.MSELoss()  # lsgan 均方误差

criterion_cycle = torch.nn.L1Loss()  # L1 loss是创建一个标准来测量输入xx和目标yy中每个元素之间的平均绝对误差（MAE），用来让生成的图片和训练的目标图片尽量相似,而图像中高频的细节部分则交由GAN来处理,
# 图像中的低频部分有patchGAN处理
criterion_identity = torch.nn.L1Loss()
criterion_vgg = PLloss()
vgg = load_vgg19(20)
if torch.cuda.is_available():
    vgg.cuda()
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False
# Optimizers & LR schedulers
# 优化器的设置保证了只更新生成器或判别器，不会互相影响   Adam优化器会崩掉，这里使用了RMSprop好一点
optimizer_G = torch.optim.Adam(itertools.chain(netG_S2F.parameters(), netG_F2S.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))  # 但是这些对象在不同的容器中，你希望代码在不失可读性的情况下避免写重复的循环，就用itertools,chain
optimizer_R=torch.optim.Adam(itertools.chain(netG_refine.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))  #  这个十refine器，我觉得不应该和生成器一起更新，所以我把他分开了


optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# =========================================定义学习率更新方式
# 将每个参数组的学习率设置为给定函数的初始lr倍。 当last_epoch = -1时，将初始lr设置为lr
# lr_lambda：在给定整数参数epoch或此类函数列表的情况下计算乘法因子的函数，针对optimizer中的每个组一个函数.param_groups
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step,
                                                   last_epoch=-1)  # 60,0,30
lr_scheduler_R = torch.optim.lr_scheduler.LambdaLR(optimizer_R,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step,
                                                   last_epoch=-1)  # 60,0,30
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step,
                                                     last_epoch=-1)

####### resume the training process
if False:
    print('Loading pre-trained models')
    netG_S2F.load_state_dict(torch.load(opt.pretrained_model + '/netG_S2F.pth'))
    netG_F2S.load_state_dict(torch.load(opt.pretrained_model + '/netG_F2S.pth'))
    netD_B.load_state_dict(torch.load(opt.pretrained_model + '/netD_B.pth'))
    #netD_A.load_state_dict(torch.load(opt.pretrained_model + '/netD_A.pth'))

    optimizer_G.load_state_dict(
        torch.load(opt.pretrained_model + '/optimizer_G.pth'))  # optimizer_G包括netG_A2B 和netG_B2A
    optimizer_D_B.load_state_dict(torch.load(opt.pretrained_model + '/optimizer_D_B.pth'))


    lr_scheduler_G.load_state_dict(torch.load(opt.pretrained_model + '/lr_scheduler_G.pth'))
    lr_scheduler_D_B.load_state_dict(torch.load(opt.pretrained_model + '/lr_scheduler_D_B.pth'))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, 3, opt.size, opt.size)
#input_B1 = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_C = Tensor(opt.batchSize, 3, opt.size, opt.size)
#input_C1 = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_D = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
input_E = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
input_F = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
input_G = Tensor(opt.batchSize, 3, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0),requires_grad=False)  # tensor([1.])这个就是如果目标是真实的话，创造batchsize个用1代表的tensor对象
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)  # 同上面

# 首先定义好buffer   来源于工具包13行
# 是为了训练的稳定，采用历史生成的虚假样本来更新判别器，而不是当前生成的虚假样本
# 定义了一个buffer对象，有一个数据存储表data，大小预设为50，
# 它的运转流程是这样的：数据表未填满时，每次读取的都是当前生成的虚假图像，
# 当数据表填满时，随机决定 1. 在数据表中随机抽取一批数据，返回，并且用当前数据补充进来 2. 采用当前数据
fake_s_buffer = ReplayBuffer()
fake_Ann_buffer = ReplayBuffer()
# Dataset loader
transforms_1 = [
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # 归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型
]
transforms_2 = [
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5))
    # 归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型
]
transforms_3 = [
    transforms.ToTensor(),
    # 归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型
]
to_pil = transforms.ToPILImage()
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_1=transforms_1, transforms_2=transforms_2,transforms_3=transforms_3),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

######################################################################

curr_iter = 0

G_losses_temp = 0
R_losses_temp = 0
D_B_losses_temp = 0

G_losses = []
R_losses = []
D_B_losses = []

open(opt.log_path, 'w').write(str(opt) + '\n\n')



###### Training ######



for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input

        real_A = Variable(input_A.copy_(batch['A']))  #image
        real_B = Variable(input_B.copy_(batch['B']))  # mk
        real_B=real_B[:,0,:,:].unsqueeze(0)
        #real_B1 = Variable(input_B1.copy_(batch['B1'])) #mk 0-1分布的
        real_C = Variable(input_C.copy_(batch['C']))  # nmk
        real_C=real_C[:,0,:,:].unsqueeze(0)
        #real_C1 = Variable(input_C1.copy_(batch['C1'])) #nmk 0-1分布的
        real_D = Variable(input_D.copy_(batch['D']))  # nsd
        real_E = Variable(input_E.copy_(batch['E'])) #sd
        fake_F = Variable(input_F.copy_(batch['F'])) #fake
        real_G=Variable(input_G.copy_(batch['G']))  #res-mask
        real_G = real_G[:, 0, :, :].unsqueeze(0)
        ###### Generators A2B and B2A ######
        # 生成器损失函数：损失函数=身份损失+对抗损失+循环一致损失
        optimizer_G.zero_grad()

        """
        real_Aflatted1 =	real_A1.view(-1)#将原图展开，这里是方便根据阴影区域的坐标值制作全阴影和全非阴影图
        real_Aflatted2 =	real_A2.view(-1)
        real_Bflatted = real_B.view(-1)#这里是将没有正则化的膨胀掩模图展开，一一对应位置
        real_Dflatted = real_D.view(-1)#正则化后的腐蚀图像

        sd_pixel=(real_Dflatted==-1).nonzero()  #0的区域是非阴影区域
        real_As=real_Aflatted1.index_put_((sd_pixel,),torch.tensor(-1).type_as(real_Aflatted1))  #将非阴影区域都变成了-1 ，因为模型处理的都是正则化后的数据，所以为了统一，所以要都变为-1
        real_As=real_As.resize_(1, 3, 400 , 400)  #现在real_As 是一张图片，其中非阴影区域是-1，阴影区域是正常的值（这里都是正则化后的）  --全阴影图片

        fake_A = 0.5 * (real_As.data + 1.0)
        # mask_queue.insert(mask_generator(img_var, temp_B))
        fake_A = np.array(transforms.Resize((400, 400))(to_pil(fake_A.data.squeeze(0).cpu())))
        Image.fromarray(fake_A).save('/mnt/data/yy/newidea/output/2.jpg')

        nsd_pixel = (real_Bflatted == 1).nonzero()  # 0的区域是阴影区域
        real_An = real_Aflatted2.index_put_((nsd_pixel,), torch.tensor(-1).type_as(real_Aflatted2))  # 将阴影区域都变成了-1 ，因为模型处理的都是正则化后的数据，所以为了统一，所以要都变为-1
        real_An = real_An.resize_(1, 3, 400, 400)  # 现在real_An 是一张图片，其中y阴影区域是-1，非阴影区域是正常的值（这里都是正则化后的）---无阴影图片
        """

        #######这里应该进行本身监督，来进行一致性检验############先初始化一次，将阴影生成器初始化一次
        fake_As = netG_F2S(real_E,real_B)
        loss_identity_As = criterion_identity(fake_As, real_E) * 5.0 #+criterion_vgg(fake_As,real_EE)+criterion_identity(RGBuvHistBlock(fake_As*0.5+0.5),RGBuvHistBlock(real_EE))

        fake_An = netG_S2F(real_D,real_C)
        loss_identity_An = criterion_identity(fake_An,real_D) * 5.0  # ||Gb(b)-b||1



        #####GAN  loss#######
        fake_s = netG_F2S(real_D,real_C)
        pred_fake = netD_B(fake_s)  # netDB是鉴别阴影的
        loss_GAN_N = criterion_GAN(pred_fake, target_real)
        fake_Ann = netG_S2F(fake_s,real_C)  # real_As是转换后的图片，阴影区域都有值，非阴影区域都是-1（因为正则化过，所以这里非阴影区域都是-1）

        #        pred_fake1=netD_A(fake_Ann) #这里是对小影像进行处理的，
        #        loss_GAN_S=criterion_GAN(pred_fake1,target_real)
        loss_cycle = criterion_cycle(fake_Ann, real_D)*10
        loss_G = loss_identity_As + loss_identity_An + loss_GAN_N + loss_cycle
        loss_G.backward()
        G_losses_temp += loss_G.item()
        optimizer_G.step()



        optimizer_R.zero_grad()

        fake_nsd=netG_S2F(real_E,real_B)
        """
        print(real_E)
        print(fake_nsd)
        fake_A = fake_nsd.data
        # mask_queue.insert(mask_generator(img_var, temp_B))
        fake_A = np.array(transforms.Resize((400, 400))(to_pil(fake_A.data.squeeze(0).cpu())))
        Image.fromarray(fake_A).save('/mnt/data/YY/copy/demo1/result/tmp/img.png')
        """
        all_nsd = netG_refine(fake_nsd*real_B+real_D,real_B+real_G)

        fake_nnsd=all_nsd*real_B
        #loss_identity_sd=criterion_identity(fake_nnsd, fake_F)
        rec_loss_func = reconstruction_loss('2nd gradient')
        rec_loss = 1.5 * rec_loss_func.compute_loss(
          fake_F,  fake_nnsd)
        loss_feat=criterion_vgg(vgg,fake_nnsd,fake_F)



        input_hist=histBlock(real_D,torch.stack([real_G[:,0,:,:], real_G[:,0,:,:], real_G[:,0,:,:]], dim=1))
        input_gauss = gaussian_op(fake_F, kernel=gaussKernel)
        generated_gauss = gaussian_op(fake_nnsd, kernel=gaussKernel)
        output_hist=histBlock(fake_nnsd,torch.stack([real_B[:,0,:,:], real_B[:,0,:,:], real_B[:,0,:,:]], dim=1))
        loss_hist=(1 / np.sqrt(2.0)) * (torch.sqrt(
            torch.sum(torch.pow(torch.sqrt(
                input_hist) - torch.sqrt(
                output_hist), 2)))) / input_hist.shape[0]

        loss_var=-1 * (1.5/ 10) * torch.sum(torch.abs(input_hist - output_hist)) * torch.mean(torch.abs(torch.std(torch.std(input_gauss, dim=2), dim=2) -torch.std(torch.std(generated_gauss, dim=2), dim=2)))
        loss_color=loss_hist+loss_var


        loss_R = loss_color+rec_loss+loss_feat

        loss_R.backward()



        # G_losses.append(loss_G.item())
        R_losses_temp += loss_R.item()
        optimizer_R.step()



        #######Discriminator B############
        optimizer_D_B.zero_grad()
        pred_real = netD_B(real_E)
        loss_DB_real = criterion_GAN(pred_real, target_real)  # log(Db(b))

        # Fake loss
        # fake_Ann=fake_Ann_buffer.push_and_pop(fake_Ann)
        # pred_fake1=netD_A(fake_Ann.detach())
        # loss_DA_fake=criterion_GAN(pred_fake1,target_real)

        fake_s = fake_s_buffer.push_and_pop(fake_s)
        pred_fake = netD_B(fake_s.detach())
        loss_DB_fake = criterion_GAN(pred_fake, target_fake)  # log(1-Db(G(a)))

        # loss_D_A=loss_DA_fake+loss_DA_real
        loss_D_B = (loss_DB_fake + loss_DB_real)*0.5

        # Total loss
        # loss_D_A = loss_D_A
        # loss_D_A.backward()


        loss_D_B.backward()

        # D_B_losses.append(loss_D_B.item())
        # D_A_losses_temp += loss_D_A.item()
        D_B_losses_temp += loss_D_B.item()

        # optimizer_D_A.step()
        optimizer_D_B.step()
        ###################################
        curr_iter += 1

        if (i + 1) % opt.iter_loss == 0:
            log = '[iter %d], [loss_G %.5f],  [loss_D_B %.5f],  [loss_R %.5f]' %  (curr_iter, loss_G, loss_D_B, loss_R)
            print(log)
            open(opt.log_path, 'a').write(log + '\n')
            G_losses.append(G_losses_temp / opt.iter_loss)
            R_losses.append(R_losses_temp / opt.iter_loss)
            D_B_losses.append(D_B_losses_temp / opt.iter_loss)
            G_losses_temp = 0
            R_losses_temp = 0
            D_B_losses_temp = 0



            avg_log = '[the last %d iters], [loss_G_identity_As %.5f], [loss_G_identity_An %.5f],[loss_GAN_N %.5f],[loss_cycle %.5f],[loss_color %.5f],[rec_loss %.5f],[loss_feat %.5f],[D_B_losses %.5f],' \
                      % (
                      opt.iter_loss, loss_identity_As, loss_identity_An, loss_GAN_N,loss_cycle,loss_color,rec_loss,loss_feat,
                      D_B_losses[D_B_losses.__len__() - 1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

        # Progress report (http://137.189.90.150:8097)
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
        #             images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()

    lr_scheduler_R.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_S2F.state_dict(), 'ckpt1/netG_S2F.pth')
    torch.save(netG_F2S.state_dict(), 'ckpt1/netG_F2S.pth')
    torch.save(netG_refine.state_dict(), 'ckpt1/netG_refine.pth')
   # torch.save(style.state_dict(), 'ckpt/style.pth')
    torch.save(netD_B.state_dict(), 'ckpt1/netD_B.pth')

    torch.save(optimizer_G.state_dict(), 'ckpt1/optimizer_G.pth')
    torch.save(optimizer_R.state_dict(), 'ckpt1/optimizer_R.pth')
    torch.save(optimizer_D_B.state_dict(), 'ckpt1/optimizer_D_B.pth')

    torch.save(lr_scheduler_G.state_dict(), 'ckpt1/lr_scheduler_G.pth')
    torch.save(lr_scheduler_R.state_dict(), 'ckpt1/lr_scheduler_R.pth')
    torch.save(lr_scheduler_D_B.state_dict(), 'ckpt1/lr_scheduler_D_B.pth')

    if (epoch + 1) % opt.snapshot_epochs == 0:
        torch.save(netG_S2F.state_dict(), ('ckpt1/netG_S2F_%d.pth' % (epoch + 1)))
        torch.save(netG_F2S.state_dict(), ('ckpt1/netG_F2S_%d.pth' % (epoch + 1)))
        torch.save(netG_refine.state_dict(), ('ckpt1/netG_refine_%d.pth' % (epoch + 1)))
    #    torch.save(style.state_dict(), ('ckpt/style_%d.pth' % (epoch + 1)))
        torch.save(netD_B.state_dict(), ('ckpt1/netD_B_%d.pth' % (epoch + 1)))

    print('Epoch:{}'.format(epoch))



