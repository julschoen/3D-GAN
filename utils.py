import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
import functools
from torch.nn import Parameter as P

def snconv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, sngan=False):
  if sngan:
    return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=bias)
  else:
    return SpectralNorm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias))

def snlinear(in_features, out_features, sngan=False):
  if sngan:
    return nn.Linear(in_features=in_features, out_features=out_features)
  else:
    return SpectralNorm(nn.Linear(in_features=in_features, out_features=out_features))

class Attention(nn.Module):
  def __init__(self, ch):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.ch_ = self.ch//8

    self.f = snconv3d(self.ch, self.ch_, kernel_size=1, padding=0, bias=False)
    self.g = snconv3d(self.ch, self.ch_, kernel_size=1, padding=0, bias=False)
    self.h = snconv3d(self.ch, self.ch_, kernel_size=1, padding=0, bias=False)
    self.v = snconv3d(self.ch_, self.ch, kernel_size=1, padding=0, bias=False)
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    f = self.f(x)
    g = F.max_pool3d(self.g(x), [2,2,2], stride=2)
    f = f.view(-1, self.ch_, x.shape[2] * x.shape[3] * x.shape[4])
    g = g.view(-1, self.ch_, x.shape[2] * x.shape[3] * x.shape[4]//8)
    beta = F.softmax(torch.bmm(f.permute(0,2,1), g), -1)

    h = F.max_pool3d(self.h(x), [2,2,2], stride=2)  
    h = h.view(-1, self.ch_, x.shape[2] * x.shape[3] * x.shape[4]//8)
    o = self.v(torch.bmm(h, beta.permute(0,2,1)).view(-1, self.ch_, x.shape[2], x.shape[3], x.shape[4]))
    return self.gamma * o + x

class GBlockDeep(nn.Module):
  def __init__(self, in_channels, out_channels, upsample=None, sngan=False, channel_ratio=4):
    super(GBlockDeep, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.hidden_channels = self.in_channels // channel_ratio
    
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0, sngan=sngan)
    self.conv2 = snconv3d(self.hidden_channels, self.hidden_channels, sngan=sngan)
    self.conv3 = snconv3d(self.hidden_channels, self.hidden_channels, sngan=sngan)
    self.conv4 = snconv3d(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0, sngan=sngan)
    # Batchnorm layers
    self.bn1 = nn.BatchNorm3d(self.in_channels)
    self.bn2 = nn.BatchNorm3d(self.hidden_channels)
    self.bn3 = nn.BatchNorm3d(self.hidden_channels)
    self.bn4 = nn.BatchNorm3d(self.hidden_channels)
    # upsample layers
    self.upsample = upsample
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
    # Project down to channel ratio
    h = self.conv1(self.activation(self.bn1(x)))
    # Apply next BN-ReLU
    h = self.activation(self.bn2(h))
    if self.in_channels != self.out_channels:
      x = x[:, :self.out_channels]   
    # Upsample both h and x at this point
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    # 3x3 convs
    h = self.conv2(h)
    h = self.conv3(self.activation(self.bn3(h)))
    # Final 1x1 conv
    h = self.conv4(self.activation(self.bn4(h)))
    return h + x

class DBlockDeep(nn.Module):
  def __init__(self, in_channels, out_channels, wide=True, preactivation=True,
               downsample=None, channel_ratio=4):
    super(DBlockDeep, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio

    self.preactivation = preactivation
    self.activation = nn.ReLU(inplace=True)
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv3 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv4 = snconv3d(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
                                 
    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = snconv3d(in_channels, out_channels - in_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)    
    return x
    
  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)     
    # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)

class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels, upsample=None, sngan=False):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.activation = nn.ReLU(inplace=True)
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.out_channels, sngan=sngan)
    self.conv2 = snconv3d(self.out_channels, self.out_channels, sngan=sngan)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = snconv3d(in_channels, out_channels, 
                                     kernel_size=1, padding=0, sngan=sngan)
    # Batchnorm layers
    self.bn1 = nn.BatchNorm3d(in_channels)
    self.bn2 = nn.BatchNorm3d(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x):
    h = self.activation(self.bn1(x))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    h = self.activation(self.bn2(h))
    h = self.conv2(h)
    if self.learnable_sc:       
      x = self.conv_sc(x)
    return h + x

class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, wide=True,
               preactivation=False, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.preactivation = preactivation
    self.activation = nn.ReLU(inplace=True)
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.hidden_channels)
    self.conv2 = snconv3d(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = snconv3d(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x
    
  def forward(self, x):
    if self.preactivation:
      h = F.relu(x)
    else:
      h = x    
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)     
    return h + self.shortcut(x)
