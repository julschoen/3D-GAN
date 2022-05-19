import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm
from utils import Attention as Self_Attn


class Generator(nn.Module):
    def __init__(self, params, im_size=128):
        super(Generator, self).__init__()
        z_dim = params.z_size
        conv_dim = params.filterG
        self.im_size = im_size
        Normalization = SpectralNorm
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.im_size)) - 2
        mult = 2 ** repeat_num # 32
        layer1.append(Normalization(nn.ConvTranspose3d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm3d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult # 2048
        layer2.append(Normalization(nn.ConvTranspose3d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm3d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2) # 1024
        layer3.append(Normalization(nn.ConvTranspose3d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm3d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        layer4 = []
        curr_dim = int(curr_dim / 2) # 512
        layer4.append(Normalization(nn.ConvTranspose3d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer4.append(nn.BatchNorm3d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())
        self.l4 = nn.Sequential(*layer4)

        self.attn1 = Self_Attn(int(curr_dim / 2))

        layer4 = []
        curr_dim = int(curr_dim / 2) # 256
        layer4.append(Normalization(nn.ConvTranspose3d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer4.append(nn.BatchNorm3d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())
        self.l5 = nn.Sequential(*layer4)

        curr_dim = int(curr_dim / 2) # 128
        last.append(nn.ConvTranspose3d(curr_dim, 1, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        

    def forward(self, z):
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.attn1(out)
        out = self.l5(out)
        out = self.last(out)
        return out


class Discriminator(nn.Module):
    ''' Discriminator '''
    def __init__(self, params, im_size=128):
        super(Discriminator, self).__init__()
        self.im_size = im_size
        conv_dim = params.filterD
        Normalization = SpectralNorm
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(Normalization(nn.Conv3d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim

        layer2.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        
        layer4 = []
        layer4.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        self.attn1 = Self_Attn(curr_dim*2)
        curr_dim = curr_dim * 2

        layer4 = []
        layer4.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l5 = nn.Sequential(*layer4)
        curr_dim = curr_dim * 2

        last.append(nn.Conv3d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.attn1(out)
        out = self.l5(out)
        out = self.last(out)
        return out.squeeze()