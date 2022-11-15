import numpy as np
import torch
import torch.nn as nn
from math import log2
import math
from kornia.filters import filter3d
import torch.nn.functional as F

#----------------------------------------------------------------------------
### Helpers ###
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 3, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        f = f.repeat((1,3,1)).reshape(1,3,3,3)
        return filter3d(x, f, normalized=True)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

#----------------------------------------------------------------------------
### Mapping Network ###
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = None, # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation is None and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.squeeze()
            x = F.linear(x, w, bias=b)
            x = self.activation(x)
        return x

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.training = True

        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=nn.LeakyReLU(0.2), lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if w_avg_beta is not None and num_ws is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = F.normalize(z.squeeze(), dim=1)

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg = self.w_avg.to(x.dtype)
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        return x

#----------------------------------------------------------------------------
### Synthesis Network ###
class Conv3DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel, kernel)))
        self.eps = eps

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w, d = x.shape
        out_channels, in_channels, kh, kw, kd = self.weight.shape
        
        w1 = y[:, None, :, None, None, None]
        w2 = self.weight[None, :, :, :, :, :]

        weights = w2 * w1

        if self.demod:
            demod = torch.rsqrt(weights.square().sum(dim=(2, 3, 4, 5)) + self.eps)
            weights = weights * demod.reshape(b, -1, 1, 1, 1, 1)

        x = x.reshape(1, -1, h, w, d)

        _, _, *ws = weights.shape
        weights = weights.reshape(-1, in_channels, kh, kw, kd)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv3d(x, weights, padding=padding, groups=b)

        x = x.reshape(b, -1, h, w, d)
        return x

class OutBlock(nn.Module):
    def __init__(self, latent_dim, input_channel):
        super().__init__()
        self.input_channel = input_channel
        self.affine = FullyConnectedLayer(latent_dim, input_channel)

        out_filters = 1
        self.conv = Conv3DMod(input_channel, out_filters, 1, demod=False)

    def forward(self, x, w):
        style = self.affine(w)
        x = self.conv(x, style)

        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, resolution, upsample = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) if upsample else None

        self.affine1 = FullyConnectedLayer(latent_dim, in_channels, bias_init=1)
        self.conv1 = Conv3DMod(in_channels, out_channels, 3)
        
        self.affine2 = FullyConnectedLayer(latent_dim, out_channels, bias_init=1)
        self.conv2 = Conv3DMod(out_channels, out_channels, 3)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.register_buffer('noise_const', torch.randn([resolution, resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, w):
        if self.upsample is not None:
            x = self.upsample(x)
        noise = self.noise_const * self.noise_strength

        style1 = self.affine1(w)
        x = self.conv1(x, style1)

        x = self.activation(x.add_(noise.to(x.dtype)))

        style2 = self.affine2(w)
        x = self.conv2(x, style2)
        x = self.activation(x + noise)

        return x

class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution, network_capacity = 16, fmap_max = 512):
        super().__init__()
        self.image_size = img_resolution
        self.latent_dim = w_dim
        self.num_layers = int(log2(self.image_size)-1)
        self.block_resolutions = [2**(i+2) for i in range(self.num_layers)]
        filters = [network_capacity * (2**i) for i in range(self.num_layers)][::-1]

        channels_dict = {res: min(filters[i], fmap_max) for i, res in enumerate(self.block_resolutions)}
        print(channels_dict)
        init_res = self.block_resolutions[0]
        init_channels = channels_dict[init_res]
        self.initial_block = nn.Parameter(torch.randn((1, init_channels, init_res, init_res, init_res)))
        self.block_resolutions = self.block_resolutions[1:]

        self.blocks = nn.ModuleList([])
        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res//2]
            out_channels = channels_dict[res]
            block = GeneratorBlock(
                self.latent_dim,
                in_channels,
                out_channels,
                res
            )
            self.blocks.append(block)

        out = OutBlock(self.latent_dim, out_channels)
        self.blocks.append(out)

    def forward(self, styles):
        x = self.initial_block.expand(styles.shape[0], -1, -1, -1, -1)
        styles = styles.transpose(0, 1)

        for style, block in zip(styles, self.blocks):
            x = block(x, style)

        return torch.tanh(x)

#----------------------------------------------------------------------------
### Generator ###
class Generator(torch.nn.Module):
    def __init__(self, params,
        w_dim = 512,                      # Intermediate latent (W) dimensionality.
        img_resolution=128,             # Output resolution.
        img_channels=1,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.p = params
        self.z_dim = self.p.z_size
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=self.w_dim, img_resolution=self.img_resolution, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_layers
        self.mapping = MappingNetwork(z_dim=self.z_dim, w_dim=self.w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img, ws

#----------------------------------------------------------------------------
### Discriminator ###
class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv3d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv3d(input_channels, filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(filters, filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv3d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Discriminator(nn.Module):
    def __init__(self, params, image_size=128, network_capacity = 16, fmap_max = 512):
        super().__init__()
        self.p = params
        num_layers = int(log2(image_size))
        num_init_filters = 1

        blocks = []
        filters = [num_init_filters] + [min(network_capacity * (2 ** i), fmap_max) for i in range(num_layers)]

        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)


        self.blocks = nn.ModuleList(blocks)

        chan_last = filters[-1]
        latent_dim = 2**3 * chan_last


        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.final_conv = nn.Conv3d(chan_last, chan_last, 1, padding=0)
        self.out = FullyConnectedLayer(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        for block in self.blocks:
            x = block(x)

        x = self.act(self.final_conv(x))
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x.squeeze()
