import torch.nn as nn
import torch
import torch.nn.functional as F

class IdentityLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(IdentityLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        mask = self.mask.clone().expand_as(input)
        self.loss = torch.nn.L1Loss(input*mask, self.target) * self.weight
        return input

    def identity_hook(self, module, grad_input, grad_output):
        self.mask = self.mask[:, 0:1, :, :]
        mask = self.mask.clone().expand_as(grad_input[0])
        # print('Inside ' + module.__class__.__name__ + ' backward')
        #
        # print('grad_input size:', grad_input[0].size())
        # print('grad_output size:', grad_output[0].size())
        # assert grad_input[0].shape == self.mask.shape, \
        #     'grad_input:{} is not matchable with mask:{}'.format(grad_input[0].shape, self.mask.shape)

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input_1 = grad_input_1 * self.weight
        # grad_input_1 = grad_input_1 * self.mask
        # grad_input = tuple([grad_input_1])

        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1 * mask
        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])
        # grad_input_1 = grad_input_1 * self.weight
        grad_input = tuple([grad_input_1])
        return grad_input


class GANLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(GANLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = torch.nn.MSELoss(input, self.target) * self.weight
        return input

    def gan_hook(self, module, grad_input, grad_output):
        self.mask = self.mask[:, 0:1, :, :]
        mask = self.mask.clone().expand_as(grad_input[0])
        # print('Inside ' + module.__class__.__name__ + ' backward')
        #
        # print('grad_input size:', grad_input[0].size())
        # print('grad_output size:', grad_output[0].size())
        # assert grad_input[0].shape == self.mask.shape, \
        #     'grad_input:{} is not matchable with mask:{}'.format(grad_input[0].shape, self.mask.shape)

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input_1 = grad_input_1 * self.weight
        # grad_input_1 = grad_input_1 * self.mask
        # grad_input = tuple([grad_input_1])

        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1 * mask
        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])
        # grad_input_1 = grad_input_1 * self.weight
        grad_input = tuple([grad_input_1])
        return grad_input

class CycleLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(CycleLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = torch.nn.L1Loss(input, self.target) * self.weight
        return input

    def cycle_hook(self, module, grad_input, grad_output):
        self.mask = self.mask[:, 0:1, :, :]
        mask = self.mask.clone().expand_as(grad_input[0])
        # print('Inside ' + module.__class__.__name__ + ' backward')
        #
        # print('grad_input size:', grad_input[0].size())
        # print('grad_output size:', grad_output[0].size())
        # assert grad_input[0].shape == self.mask.shape, \
        #     'grad_input:{} is not matchable with mask:{}'.format(grad_input[0].shape, self.mask.shape)

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input_1 = grad_input_1 * self.weight
        # grad_input_1 = grad_input_1 * self.mask
        # grad_input = tuple([grad_input_1])

        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1 * mask
        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])
        # grad_input_1 = grad_input_1 * self.weight
        grad_input = tuple([grad_input_1])
        return grad_input



#mask是0，1的，图像是tensor以后进入losses才正则化的
def get_model_and_losses(G_S2F,G_F2S,D_B,
                         sd_img, nsd_img, smk_image, nsmk_image,
                         identity_weight=5, GAN_weight=1, cycle_weight=1,
                                ):
    identity_losses = []
    GAN_losses = []
    cycle_losses=[]


    i = 0

    print('-----Setting up identity_weight-----')
    fake_sd = G_F2S(sd_img).detach()
    fake_nsd=G_S2F(nsd_img).detach()
    identity_loss = IdentityLoss(sd_img, smk_image, identity_weight)+IdentityLoss(nsd_img,nsmk_image,identity_weight)
    identity_loss.register_backward_hook(identity_loss.identity_hook)
    identity_losses.append(identity_loss)

    print('-----Setting up GAN_weight AND  cycle_weight-----')
    fake_s = G_F2S(nsd_img).detach()
    pred_fake = D_B(fake_s)
    gan_loss = GANLoss(pred_fake,torch.tensor(1) ,GAN_weight)
    gan_loss.register_backward_hook(gan_loss.gan_hook)
    GAN_losses.append(gan_loss)


    fake_nsd = G_S2F(fake_s).detach()
    Cycle_loss = CycleLoss(sd_img, smk_image, cycle_weight)
    Cycle_loss.register_backward_hook(Cycle_loss.cycle_hook)
    cycle_losses.append(Cycle_loss)
    return identity_losses,GAN_losses,cycle_losses
###################################


