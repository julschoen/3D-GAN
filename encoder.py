import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import functools
from utils import Attention, DBlock, snconv3d, snlinear
from torch.distributions import Normal, Independent
from torch.distributions import kl_divergence as KLD
import numpy as np
from torch.nn.functional import softplus, sigmoid, softmax


class Encoder(nn.Module):
  def __init__(self, params):
    super(Encoder, self).__init__()
    self.p = params
    # Architecture
    self.arch = {'in_channels' :  [item * self.p.filterD for item in [1, 2, 4,  8, 16]],
               'out_channels' : [item * self.p.filterD for item in [2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in '16'.split('_')]
                              for i in range(2,8)}}
    
    # Prepare model
    self.input_conv = snconv3d(1, self.arch['in_channels'][0])

    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index] if d_index==0 else self.arch['out_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       preactivation=True,
                       downsample=(nn.AvgPool3d(2) if self.arch['downsample'][index] and d_index==0 else None))
                       for d_index in range(1)]]
      if self.p.att:
        if self.arch['attention'][self.arch['resolution'][index]]:
          self.blocks[-1] += [Attention(self.arch['out_channels'][index])]

    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    self.mu = nn.Linear(self.arch['out_channels'][-1], self.p.z_size)
    self.logvar = nn.Linear(self.arch['out_channels'][-1], self.p.z_size)
    self.activation = nn.ReLU(inplace=True)
    self.init_weights()

    self.mu_0 = torch.zeros((1,params.z_size))
    self.sigma_1 = torch.ones((1,params.z_size))
    self.cuda = (params.device == 'cuda')


  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv3d)
          or isinstance(module, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

  def forward(self, x):
    # Run input conv
    h = self.input_conv(x)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3, 4])

    mu = self.mu(h)
    log_var = self.logvar(h)

    log_var = softplus(log_var)
    sigma = torch.exp(log_var / 2)
    
    posterior = Independent(Normal(loc=mu,scale=sigma),1)
    z = posterior.rsample()

    # Instantiate a standard Gaussian with mean=mu_0, std=sigma_0
    # This is the prior distribution p(z)
    if self.cuda:
      prior = Independent(Normal(loc=self.mu_0.to('cuda:'+str(torch.cuda.current_device())),
                scale=self.sigma_1.to('cuda:'+str(torch.cuda.current_device()))),1)
    else:
      prior = Independent(Normal(loc=self.mu_0, scale=self.sigma_1),1)

    # Estimate the KLD between q(z|x)|| p(z)
    kl = KLD(posterior,prior).mean()
    return z, kl
