import torch 
import torch.nn as nn
import torch.nn.functional as F
from msl import RandomCrop3D
from self_attention import SelfAttentionBlock
import torch.nn.utils.spectral_norm as SpectralNorm

class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_up, self).__init__()
        
        self.conv1 = nn.Conv3d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm3d(channel_out//2)
        self.conv2 = nn.Conv3d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm3d(channel_out)
        
        self.conv3 = nn.Conv3d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        nz = params.z_size
        ngf = params.filterG
        nc = 1

        self.ngpu = params.ngpu
        if params.res:
            self.main = nn.Sequential(
                Res_up(nz, ngf*16),
                Res_up(ngf*16, ngf*8),
                Res_up(ngf*8, ngf*8),
                Res_up(ngf*8, ngf*4),
                Res_up(ngf*4, ngf*2),
                Res_up(ngf*2, ngf),
                Res_up(ngf, ngf//2),
                nn.Conv3d(ngf//2, nc, 3, 1, 1),
                nn.Tanh()
            )
        elif params.sagan:
            self.main = nn.Sequential(
                nn.ConvTranspose3d(nz, ngf*16, 4, stride=1),
                SpectralNorm(),
                nn.BatchNorm3d(ngf*16),
                nn.ReLU(True),
                nn.ConvTranspose3d(ngf*16, ngf*8, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.BatchNorm3d(ngf*8),
                nn.ReLU(True),
                nn.ConvTranspose3d(ngf*8, ngf*4, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.BatchNorm3d(ngf*4),
                nn.ReLU(True),
                nn.ConvTranspose3d(ngf*4, ngf*2, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.BatchNorm3d(ngf*2),
                nn.ReLU(True),
                nn.ConvTranspose3d(ngf*2, ngf, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose3d(ngf, 1, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.Tanh()
            )
        else:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose3d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm3d(ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*16) x 4 x 4
                nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 8 x 8
                nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 16 x 16 
                nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 32 x 32
                nn.ConvTranspose3d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 64 x 64
                nn.ConvTranspose3d(    ngf,      nc, 4, 2, 1, bias=False),
                # state size. (nc) x 128 x 128 x 128
                nn.Tanh()
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        
        return output

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        nz = params.z_size
        ndf = params.filterD
        nc = 1
        self.ngpu=params.ngpu
        
        if params.msl:
            self.main = nn.Sequential(
                # input is 128 x 128 x 128
                RandomCrop3D(device=params.device),
                nn.Conv3d(nc, ndf, 4, stride=2, padding=1, bias=False), 
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 14 x 14
                nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf * 2, 16, 16, 16]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 6 x 6 
                nn.Conv3d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf * 4, 8, 8, 8]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 5 x 5
                nn.Conv3d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf * 8, 4, 4, 4]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*16) x 4 x 4
                nn.Conv3d(ndf * 8, 1, (4,4,4), stride=1, padding=0, bias=False),
                # state size. 1
            )
        elif params.sagan:
            self.main = nn.Sequential(
                nn.Conv3d(1, ndf, 4, stride=1, padding=1),
                SpectralNorm(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf, ndf*2, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf*2, ndf*4, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.LeakyReLU(0.2, inplace=True),
                SelfAttentionBlock(ndf*4),
                nn.Conv3d(ndf*4, ndf*8, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf*8, ndf*16, 4, stride=2, padding=1),
                SpectralNorm(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf*16, 1, 3, stride=1, padding=0),
                SpectralNorm(),
            )
        else:  
            self.main = nn.Sequential(
                # input is 128 x 128 x 128
                nn.Conv3d(nc, ndf, 4, stride=2, padding=1, bias=False), 
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 14 x 14
                nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf * 2, 32, 32, 32]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 7 x 7
                nn.Conv3d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf * 4, 16, 16, 16]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 6 x 6 
                nn.Conv3d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf * 8, 8, 8, 8]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 5 x 5
                nn.Conv3d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
                nn.LayerNorm([ndf * 16, 4, 4, 4]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*16) x 4 x 4
                nn.Conv3d(ndf * 16, 1, (4,4,4), stride=1, padding=0, bias=False),
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)

        if not sagan:
            output = output.mean(0)
            output = output.view(1)
        
        return output
