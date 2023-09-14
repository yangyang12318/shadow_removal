"""
##### Copyright 2021 Mahmoud Afifi.

 If you find this code useful, please cite our paper:

 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
 Controlling Colors of GAN-Generated and Real Images via Color Histograms."
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
####
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1e-6

#改了，【0，1】--
class RGBuvHistBlock(nn.Module):
  def __init__(self, h=64, insz=400, resizing='interpolation',
               method='inverse-quadratic', sigma=0.02, intensity_scale=True,
               hist_boundary=None, green_only=False, device='cuda'):
    """ Computes the RGB-uv histogram feature of a given image.
    Args:
      h: histogram dimension size (scalar). The default value is 64.   直方图的维度，默认是64
      insz: maximum size of the input image; if it is larger than this size, the
        image will be resized (scalar). Default value is 150 (i.e., 150 x 150
        pixels).   最大的输入图像尺寸，默认是150，如果图片大于这个尺寸，就会被resize
      resizing: resizing method if applicable. Options are: 'interpolation' or
        'sampling'. Default is 'interpolation'.  resizing的方法，默认是interpolation（插值法）
      method: the method used to count the number of pixels for each bin in the
        histogram feature. Options are: 'thresholding', 'RBF' (radial basis
        function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
        #计算每个直方图特征中每个bin的像素数量，默认是inverse-quadratic
      sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
        the sigma parameter of the kernel function. The default value is 0.02.
        如果是那两种方法，就会需要sigma，这里默认的sigma参数核的大小是0.02
      intensity_scale: boolean variable to use the intensity scale (I_y in
        Equation 2). Default value is True. 用不用那个超参数Iy，默认是使用超参数
      hist_boundary: a list of histogram boundary values. Default is [-3, 3].直方图的范围，
      green_only: boolean variable to use only the log(g/r), log(g/b) channels.
        Default is False.  只有绿色通道作为分子那一个，默认是false

    Methods:
      forward: accepts input image and returns its histogram feature. Note that
        unless the method is 'thresholding', this is a differentiable function
        and can be easily integrated with the loss function. As mentioned in the
         paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
         training.
    """
    super(RGBuvHistBlock, self).__init__()
    self.h = h
    self.insz = insz
    self.device = device
    self.resizing = resizing
    self.method = method
    self.intensity_scale = intensity_scale
    self.green_only = green_only
    if hist_boundary is None:
      hist_boundary = [-3, 3]
    hist_boundary.sort() #这里是对列表进行升序排序
    self.hist_boundary = hist_boundary
    if self.method == 'thresholding':
      self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h  #就是将阈值平均分为h份，这里是64份
    else:
      self.sigma = sigma  #0.02

  def forward(self, x,m):

    x = torch.clamp(x, 0, 1)   #torch.clamp就是将x这个列表中的值都夹紧在0，1之间，如果x小于0就变为0，如果大于1就变为1
    if x.shape[2] > self.insz or x.shape[3] > self.insz:  #如果输入图像的长和宽有一个大于给定的尺寸，就给他resize了，下面就是选择resize的方法
      if self.resizing == 'interpolation':
        x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                  mode='bilinear', align_corners=False)
        m_sampled = m

      else:
        raise Exception(
          f'Wrong resizing method. It should be: interpolation or sampling. '
          f'But the given value is {self.resizing}.')
    else:
      x_sampled = x
      m_sampled = m
  #x_sampled就是x resize一下，尺寸还是（1，3，insz，insz）
    L = x_sampled.shape[0]  # size of mini-batch
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]
      m_sampled=m_sampled[:,:3,:,:]
    X = torch.unbind(x_sampled, dim=0) #这个就是如果batchsize大于1的时候，把x_sampled的第一个维度解绑，变成一个tuple，tuple的元素都是一张图片的RGB信息，batchsize是几，tuple（X）就有几个元素
    M = torch.unbind(m_sampled,dim=0)




    hists = torch.zeros((x_sampled.shape[0], 1 + int(not self.green_only) * 2,
                         self.h, self.h)).to(device=self.device)
    for l in range(L):

      X=torch.reshape(X[l], (3, -1))
      M=torch.reshape(M[l],(3,-1))
      Mc=M.clone().cpu()
      Mc = torch.BoolTensor(Mc > 0.5).cuda()

      X=X.masked_select(Mc)
      X=torch.reshape(X,(3,-1))

      I = torch.t(X)#这就是把每张图片都reshape成三行的，RGB各占一个通道，  torch.t就是转置的意思。
      II = torch.pow(I, 2) #这里是I的2次方的意思,注意是每个地方都平方
      if self.intensity_scale:  #这里是公式2中使不使用Iy这个超参数
        Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS),
                             dim=1) #Iy=根号下（Ir的平方+Ig的平方+Ib的平方）
      else:
        Iy = 1
      if not self.green_only:
        Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] +
                                                                   EPS), dim=1)
        Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] +
                                                                   EPS), dim=1)
        #这个是用公式1来计算IuR和IvR，Iu0的尺寸是（4096，1），就是64*64的


        diff_u0 = abs(Iu0 - torch.unsqueeze(torch.tensor(np.linspace(self.hist_boundary[0], self.hist_boundary[1], num=self.h)),dim=0).to(self.device))  #这个就是公式三中的|Iuc-u|
        diff_v0 = abs( Iv0 - torch.unsqueeze(torch.tensor(np.linspace(self.hist_boundary[0], self.hist_boundary[1], num=self.h)),dim=0).to(self.device))
        if self.method == 'thresholding':
          diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
          diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
        elif self.method == 'RBF':
          diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u0 = torch.exp(-diff_u0)  # Radial basis function
          diff_v0 = torch.exp(-diff_v0)
        elif self.method == 'inverse-quadratic':
          diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic  这里计算出来的就是公式三中的k（Iuc，Ivc，u，v）
          diff_v0 = 1 / (1 + diff_v0)
        else:
          raise Exception(
            f'Wrong kernel method. It should be either thresholding, RBF,'
            f' inverse-quadratic. But the given value is {self.method}.')
        diff_u0 = diff_u0.type(torch.float32)
        diff_v0 = diff_v0.type(torch.float32)
        a = torch.t(Iy * diff_u0) #然后计算完以后返回公式2计算H（u，v，c）
        hists[l, 0, :, :] = torch.mm(a, diff_v0) #然后这就算出来的最后的结果

      Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                            dim=1)
      Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                            dim=1)
      diff_u1 = abs(
        Iu1 - torch.unsqueeze(torch.tensor(np.linspace(
          self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
          dim=0).to(self.device))
      diff_v1 = abs(
        Iv1 - torch.unsqueeze(torch.tensor(np.linspace(
          self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
          dim=0).to(self.device))

      if self.method == 'thresholding':
        diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
        diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = torch.exp(-diff_u1)  # Gaussian
        diff_v1 = torch.exp(-diff_v1)
      elif self.method == 'inverse-quadratic':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
        diff_v1 = 1 / (1 + diff_v1)

      diff_u1 = diff_u1.type(torch.float32)
      diff_v1 = diff_v1.type(torch.float32)
      a = torch.t(Iy * diff_u1)
      if not self.green_only:
        hists[l, 1, :, :] = torch.mm(a, diff_v1)

      else:
        hists[l, 0, :, :] = torch.mm(a, diff_v1)
    #这里计算出来一个通道除以另外两个通道的值，
      if not self.green_only:
        Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] +
                                                                   EPS), dim=1)
        Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] +
                                                                   EPS), dim=1)
        diff_u2 = abs(
          Iu2 - torch.unsqueeze(torch.tensor(np.linspace(
            self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
            dim=0).to(self.device))
        diff_v2 = abs(
          Iv2 - torch.unsqueeze(torch.tensor(np.linspace(
            self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
            dim=0).to(self.device))
        if self.method == 'thresholding':
          diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
          diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
        elif self.method == 'RBF':
          diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u2 = torch.exp(-diff_u2)  # Gaussian
          diff_v2 = torch.exp(-diff_v2)
        elif self.method == 'inverse-quadratic':
          diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
          diff_v2 = 1 / (1 + diff_v2)
        diff_u2 = diff_u2.type(torch.float32)
        diff_v2 = diff_v2.type(torch.float32)
        a = torch.t(Iy * diff_u2)
        hists[l, 2, :, :] = torch.mm(a, diff_v2)

    # normalization
    hists_normalized = hists / (
        ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)
    #最后正则化一下，

    return hists_normalized  #(1,3,h,h)