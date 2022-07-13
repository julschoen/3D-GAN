import torch 
import torch.nn as nn
import torch.nn.functional as F
from msl import RandomCrop3D
from utils import Attention as SelfAttention
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        nz = params.z_size
        ngf = params.filterG
        nc = 1
        self.ngpu = params.ngpu
        self.dim_z = nz
        if params.sagan:
            self.main = nn.Sequential(
                # in z x 1 x 1 x 1
                SpectralNorm(nn.ConvTranspose3d(nz, ngf*16, 4, stride=1)),
                nn.LayerNorm([ngf * 16, 4, 4, 4]),
                #nn.BatchNorm3d(ngf * 16),
                nn.ReLU(True),
                # state size (ngf*16) x 4 x 4 x 4
                SpectralNorm(nn.ConvTranspose3d(ngf*16, ngf*8, 4, stride=2, padding=1)),
                nn.LayerNorm([ngf * 8, 8, 8, 8]),
                #nn.BatchNorm3d(ngf * 8),
                nn.ReLU(True),
                # state size (ngf*8) x 8 x 8 x 8
                SpectralNorm(nn.ConvTranspose3d(ngf*8, ngf*4, 4, stride=2, padding=1)),
                nn.LayerNorm([ngf * 4, 16, 16, 16]),
                #nn.BatchNorm3d(ngf * 4),
                nn.ReLU(True),
                # state size (ngf*4) x 16 x 16 x 16
                SpectralNorm(nn.ConvTranspose3d(ngf*4, ngf*2, 4, stride=2, padding=1)),
                nn.LayerNorm([ngf * 2, 32, 32, 32]),
                #nn.BatchNorm3d(ngf * 2),
                nn.ReLU(True),
                SelfAttention(ngf*2),
                # state size (ngf*2) x 32 x 32 x 32
                SpectralNorm(nn.ConvTranspose3d(ngf*2, ngf, 4, stride=2, padding=1)),
                nn.LayerNorm([ngf, 64, 64, 64]),
                #nn.BatchNorm3d(ngf),
                nn.ReLU(True),
                # state size (ngf) x 64 x 64 x 64
                SpectralNorm(nn.ConvTranspose3d(ngf, nc, 4, stride=2, padding=1)),
                nn.Tanh()
                # state size nc x 128 x 128 x 128
            )
        else:
            self.main = nn.Sequential(
                # in z x 1 x 1 x 1
                nn.ConvTranspose3d(nz, ngf*16, 4, stride=1),
                #nn.LayerNorm([ngf * 16, 4, 4, 4]),
                nn.BatchNorm3d(ngf * 16),
                nn.ReLU(True),
                # state size (ngf*16) x 4 x 4 x 4
                nn.ConvTranspose3d(ngf*16, ngf*8, 4, stride=2, padding=1),
                #nn.LayerNorm([ngf * 8, 8, 8, 8]),
                nn.BatchNorm3d(ngf * 8),
                nn.ReLU(True),
                # state size (ngf*8) x 8 x 8 x 8
                nn.ConvTranspose3d(ngf*8, ngf*4, 4, stride=2, padding=1),
                #nn.LayerNorm([ngf * 4, 16, 16, 16]),
                nn.BatchNorm3d(ngf * 4),
                nn.ReLU(True),
                # state size (ngf*4) x 16 x 16 x 16
                nn.ConvTranspose3d(ngf*4, ngf*2, 4, stride=2, padding=1),
                #nn.LayerNorm([ngf * 2, 32, 32, 32]),
                nn.BatchNorm3d(ngf * 2),
                nn.ReLU(True),
                # state size (ngf*2) x 32 x 32 x 32
                nn.ConvTranspose3d(ngf*2, ngf, 4, stride=2, padding=1),
                #nn.LayerNorm([ngf, 64, 64, 64]),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),
                # state size (ngf) x 64 x 64 x 64
                nn.ConvTranspose3d(ngf, nc, 4, stride=2, padding=1),
                nn.Tanh()
                # state size nc x 128 x 128 x 128
            )

        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        nz = params.z_size
        ndf = params.filterD
        nc = 1
        self.ngpu=params.ngpu
        self.dim_z = nz
        
        if params.msl:
            nc = 128
            self.main = nn.Sequential(
                # input is nc x 128 x 128 x 128
                RandomCrop3D(device=params.device, n_crops=nc),
                # input is nc x 64 x 64 x 64
                SpectralNorm(nn.Conv3d(nc, ndf, 4, stride=2, padding=1, bias=False)), 
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf) x 32 x 32 x 32
                SpectralNorm(nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*2) x 16 x 16 x 16
                SpectralNorm(nn.Conv3d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*4) x 8 x 8 x 8
                SpectralNorm(nn.Conv3d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*8) x 4 x 4 x 4
                SpectralNorm(nn.Conv3d(ndf * 8, 1, (4,4,4), stride=1, padding=0, bias=False)),
                # state size. 1
            )
        elif params.sngan:
            self.main = nn.Sequential(
                # input is 128 x 128 x 128
                SpectralNorm(nn.Conv3d(nc, ndf, 4, stride=2, padding=1, bias=False)), 
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf) x 64 x 64 x 64
                SpectralNorm(nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*2) x 32 x 32 x 32
                SpectralNorm(nn.Conv3d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*4) x 16 x 16 x 16
                SpectralNorm(nn.Conv3d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*8) x 8 x 8 x 8
                SpectralNorm(nn.Conv3d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*16) x 4 x 4 x 4
                SpectralNorm(nn.Conv3d(ndf * 16, 1, (4,4,4), stride=1, padding=0, bias=False)),
            )
        elif params.sagan:
            self.main = nn.Sequential(
                # input is 128 x 128 x 128
                SpectralNorm(nn.Conv3d(nc, ndf, 4, stride=2, padding=1, bias=False)), 
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf) x 64 x 64 x 64
                SpectralNorm(nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                SelfAttention(ndf*2),
                # state size. (ndf*2) x 32 x 32 x 32
                SpectralNorm(nn.Conv3d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*4) x 16 x 16 x 16
                SpectralNorm(nn.Conv3d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*8) x 8 x 8 x 8
                SpectralNorm(nn.Conv3d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.1, inplace=True),
                # state size. (ndf*16) x 4 x 4 x 4
                SpectralNorm(nn.Conv3d(ndf * 16, 1, (4,4,4), stride=1, padding=0, bias=False)),
            )
        else:  
            self.main = nn.Sequential(
                # input is 128 x 128 x 128
                nn.Conv3d(nc, ndf, 4, stride=2, padding=1, bias=False), 
                nn.LayerNorm([ndf, 64, 64, 64]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 64 x 64 x 64
                nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf*2, 32, 32, 32]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 32 x 32 x 32
                nn.Conv3d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf*4, 16, 16, 16]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 16 x 16 x 16
                nn.Conv3d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf*8, 8, 8, 8]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 8 x 8 x 8
                nn.Conv3d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf*16, 4, 4, 4]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*16) x 4 x 4 x 4
                nn.Conv3d(ndf * 16, 1, (4,4,4), stride=1, padding=0, bias=False),
            )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        output = self.main(input)        
        return output
