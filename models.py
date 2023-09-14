import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_normal
import torch
import math
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)  # *????????????

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_S2F(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_S2F, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),  # ???????????
                                     nn.Conv2d(3, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(256),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(256))
        self.conv5_b = nn.Sequential(ResidualBlock(256))
        self.conv6_b = nn.Sequential(ResidualBlock(256))
        self.conv7_b = nn.Sequential(ResidualBlock(256))
        self.conv8_b = nn.Sequential(ResidualBlock(256))
        self.conv9_b = nn.Sequential(ResidualBlock(256))
        self.conv10_b = nn.Sequential(ResidualBlock(256))
        self.conv11_b = nn.Sequential(ResidualBlock(256))
        self.conv12_b = nn.Sequential(ResidualBlock(256))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_S2F(init_weights=True)
        return model

    def forward(self, xin):
        x = self.conv1_b(xin)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        x = F.interpolate(x, size=xin.size()[2:], mode='bilinear', align_corners=True)
        xout = x + xin
        return  xout.tanh()

class Generator_F2S(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_F2S, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(4, 64, 7),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(256),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(256))
        self.conv5_b = nn.Sequential(ResidualBlock(256))
        self.conv6_b = nn.Sequential(ResidualBlock(256))
        self.conv7_b = nn.Sequential(ResidualBlock(256))
        self.conv8_b = nn.Sequential(ResidualBlock(256))
        self.conv9_b = nn.Sequential(ResidualBlock(256))
        self.conv10_b = nn.Sequential(ResidualBlock(256))
        self.conv11_b = nn.Sequential(ResidualBlock(256))
        self.conv12_b = nn.Sequential(ResidualBlock(256))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = Generator_F2S(init_weights=True)
        return model

    def forward(self, xin, mask):
        x = torch.cat((xin, mask), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv6_b(x)
        x = self.conv7_b(x)
        x = self.conv8_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return  xout.tanh() # ????????????????Tanh??????????????????[0,255]???



def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        #???????
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        #???????
        self.input_conv.apply(weights_init('xavier'))
        #
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) ?C C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class Discriminator(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        self.conv1f = PartialConv(3, 64, kernel_size=kw, stride=2, padding=padw)
        self.relu1 = nn.LeakyReLU(0.2, True)

        self.conv2f = PartialConv(64, 128, kernel_size=kw, stride=2, padding=padw)
        self.norm2f = norm_layer(128)
        self.relu2 = nn.LeakyReLU(0.2, True)

        self.conv3f = PartialConv(128, 256, kernel_size=kw, stride=2, padding=padw)
        self.norm3f = norm_layer(256)
        self.relu3 = nn.LeakyReLU(0.2, True)

        self.conv4f = PartialConv(256, 512, kernel_size=kw, padding=padw)
        self.norm4f = norm_layer(512)
        self.relu4 = nn.LeakyReLU(0.2, True)

        self.conv5f = PartialConv(512, 1, kernel_size=kw, padding=padw)

        # self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, input, mask):
        """Standard forward."""
        xb = input
        mb = mask

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)

        # xb = self.avg_pooling(xb)
        s = F.avg_pool2d(xb, xb.size()[2:],divisor_override=torch.count_nonzero(mb)).view(xb.size()[0], -1).squeeze()
        return s

