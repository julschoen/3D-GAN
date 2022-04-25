import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels=None, value_channels=None, out_channels=None, scale=2):
        super(SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        if key_channels is None:
            self.key_channels = in_channels // 8
        if value_channels is None:
            self.value_channels = in_channels // 8
        self.pool = nn.MaxPool3d(kernel_size=scale)
        self.f_key = nn.Sequential(
            SpectralNorm(nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0))
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv3d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv3d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size, c, d, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query,key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        return context