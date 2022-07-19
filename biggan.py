import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from utils import Attention, GBlockDeep, DBlockDeep, snconv3d, snlinear
from msl import RandomCrop3D

class Generator(nn.Module):
  def __init__(self, params):
    super(Generator, self).__init__()
    self.p = params
    self.dim_z = self.p.z_size
    
    self.arch = {'in_channels' :  [item * self.p.filterG for item in [16, 16, 8, 4, 2]],
             'out_channels' : [item * self.p.filterG for item in [16, 8, 4,  2, 1]],
             'resolution' : [8, 16, 32, 64, 128],
             'attention' : {2**i: (2**i in [int(item) for item in '32'.split('_')]) for i in range(3,8)}}
    print(self.arch)
    self.linear = snlinear(self.p.z_size, self.arch['in_channels'][0] * (4**3), sngan=self.p.sngan)
      
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      if self.p.biggan:
        self.blocks += [[GBlockDeep(in_channels=self.arch['in_channels'][index],
                               out_channels=self.arch['in_channels'][index] if g_index==0 else self.arch['out_channels'][index],
                               upsample=(functools.partial(F.interpolate, scale_factor=2)if g_index == 1 else None))]
                         for g_index in range(2)]
      else:
        self.blocks += [[GBlockDeep(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             upsample=functools.partial(F.interpolate, scale_factor=2),
                             sngan=self.p.sngan)]]
      if (self.p.sagan or self.p.biggan) and self.arch['attention'][self.arch['resolution'][index]]:
          self.blocks[-1] += [Attention(self.arch['out_channels'][index])]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

   
    self.output_layer = nn.Sequential(nn.BatchNorm3d(self.arch['out_channels'][-1]),
                                    nn.ReLU(inplace=True),
                                    snconv3d(self.arch['out_channels'][-1], 1, sngan=self.p.sngan))

    self.init_weights()

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv3d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        init.orthogonal_(module.weight)
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  def forward(self, z):
    # First linear layer
    h = self.linear(z.squeeze())
    # Reshape
    h = h.view(h.size(0), -1, 4, 4, 4)    
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    return torch.tanh(self.output_layer(h))

class Discriminator(nn.Module):
  def __init__(self, params):
    super(Discriminator, self).__init__()
    self.p = params
    # Architecture
    self.arch = {'in_channels' :  [item * self.p.filterD for item in [1, 2, 4,  8, 16]],
               'out_channels' : [item * self.p.filterD for item in [2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in '16'.split('_')]
                              for i in range(2,8)}}
    
    # Prepare model
    self.input_conv = snconv3d(1, self.arch['in_channels'][0], sngan=self.p.sngan)

    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      if self.p.biggan:
        self.blocks += [[DBlockDeep(in_channels=self.arch['in_channels'][index] if d_index==0 else self.arch['out_channels'][index],
                         out_channels=self.arch['out_channels'][index],
                         preactivation=True,
                         downsample=(nn.AvgPool3d(2) if self.arch['downsample'][index] and d_index==0 else None))
                         for d_index in range(2)]]
      else:
        self.blocks += [[DBlockDeep(in_channels=self.arch['in_channels'][index],
                         out_channels=self.arch['out_channels'][index],
                         preactivation=True,
                         downsample=(nn.AvgPool3d(2) if self.arch['downsample'][index] else None))]]
      if (self.p.sagan or self.p.biggan) and self.arch['attention'][self.arch['resolution'][index]]:
        self.blocks[-1] += [Attention(self.arch['out_channels'][index])]

    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    self.linear = snlinear(self.arch['out_channels'][-1], 1, sngan=self.p.sngan)

    self.activation = nn.ReLU(inplace=True)
    self.init_weights()

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv3d)
          or isinstance(module, nn.Linear)):
        init.orthogonal_(module.weight)
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x):
    # Run input conv
    h = self.input_conv(x)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3, 4])
    return self.linear(h)
