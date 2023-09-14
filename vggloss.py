
import torch
import torch.nn as nn

from pd_vgg import pdvgg19
import torchvision
from torch.autograd import Variable
import os
import torch.nn.functional as F
def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor([0.5,0.5,0.5]).to('cuda') + torch.Tensor([0.5,0.5,0.5]).to('cuda')
    x = x.transpose(1, 3)
    return x

class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.criterion = nn.L1Loss()
    def forward(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return self.criterion(self.instancenorm(img_fea) , self.instancenorm(target_fea))


def load_vgg19(index):
    vgg = vgg_19(index)
    return vgg


def vgg_preprocess(batch):
    tensor_type = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = ((batch*0.5)+0.5) * 255  # * 0.5  [-1, 1] -> [0, 255]
    mean = tensor_type(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch




class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = pdvgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                # print(param, param.requires_grad)

    def forward(self, X, indices=None):
        out = []
        # indices = sorted(indices)
        for i in range(indices[-1] + 1):
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out


class vgg_19(nn.Module):
    def __init__(self, index):
        super(vgg_19, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:index])

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)


        return out
