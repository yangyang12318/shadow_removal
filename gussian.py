

from math import floor, log2, sqrt, pi
#from models import PartialConv2d
import torch
from torch import nn



def get_gaussian_kernel(kernel_size=15, sigma=3, channels=3):
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
  mean = (kernel_size - 1) / 2.
  variance = sigma ** 2.
  gaussian_kernel = (1. / (2. * pi * variance)) * torch.exp(
    -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
  gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
  gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
  gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                              kernel_size=kernel_size, groups=channels,
                              bias=False)
  gaussian_filter.weight.data = gaussian_kernel
  gaussian_filter.weight.requires_grad = False

  return gaussian_filter


def gaussian_op(x, kernel=None):
  if kernel is None:
    kernel = get_gaussian_kernel(kernel_size=15, sigma=15, channels=3).to(
      device=torch.cuda.current_device())
  return kernel(x)