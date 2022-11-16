import numpy as np
import torch
import torch.nn as nn
from math import log2
import math
from kornia.filters import filter3d
import torch.nn.functional as F

activation_funcs = {
    'linear':   lambda x: x,
    'relu':     lambda x: torch.nn.functional.relu(x),
    'lrelu':    lambda x: torch.nn.functional.leaky_relu(x, 0.2),
    'tanh':     lambda x: torch.tanh(x),
    'sigmoid':  lambda x: torch.sigmoid(x),
    'elu':      lambda x: torch.nn.functional.elu(x),
    'selu':     lambda x: torch.nn.functional.selu(x),
    'softplus': lambda x: torch.nn.functional.softplus(x),
    'swish':    lambda x: torch.sigmoid(x) * x
}
#----------------------------------------------------------------------------

def bias_act(x, b=None, dim=1, act='linear'):
    act = activation_funcs[act]
    
    if b is not None:
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])
  
    return act(x)

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 3:
        padx, pady, padz = padding
        padding = [padx, padx, pady, pady, padz, padz]
    padx0, padx1, pady0, pady1, padz0, padz1 = padding
    return padx0, padx1, pady0, pady1, padz0, padz1

def _get_filter_size(f):
    if f is None:
        return 1, 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2, 3]
    fd = f.shape[2]
    fw = f.shape[1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
        fd = int(fd)
    misc.assert_shape(f, [fh, fw, fd][:f.ndim])
    assert fw >= 1 and fh >= 1 and fd >=1
    return fw, fh, fd

def _conv3d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    if not flip_weight: 
        w = w.flip([2, 3, 4])

    op = F.conv3d if transpose else F.conv_transpose3d
    return op(x, w, stride=stride, padding=padding, groups=groups)

def _upfirdn3d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    if f is None:
        f = torch.ones([1, 1, 1], dtype=torch.float32, device=x.device)

    batch_size, num_channels, in_height, in_width, in_depth = x.shape
    upx, upy, upz = _parse_scaling(up)
    downx, downy, downz = _parse_scaling(down)
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1, in_depth, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1, 0, upz -1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx, in_depth * upz])

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0), max(padz0, 0), max(padz1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    x = F.conv3d(x, f, groups=b)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx, ::downz]
    return x

def conv3d_resample(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    r"""3D convolution with optional up/downsampling.
    Padding is performed only once at the beginning, not between the operations.
    Args:
        x:              Input tensor of shape
                        `[batch_size, in_channels, in_height, in_width, in_depth]`.
        w:              Weight tensor of shape
                        `[out_channels, in_channels//groups, kernel_height, kernel_width, kernel_depth]`.
        f:              Low-pass filter for up/downsampling. Must be prepared beforehand by
                        calling upfirdn2d.setup_filter(). None = identity (default).
        up:             Integer upsampling factor (default: 1).
        down:           Integer downsampling factor (default: 1).
        padding:        Padding with respect to the upsampled image. Can be a single number
                        or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation (default: True).
        flip_filter:    False = convolution, True = correlation (default: False).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    # Validate arguments.
    out_channels, in_channels_per_group, kh, kw, kd = _get_weight_shape(w)
    fw, fh, fd = _get_filter_size(f)
    px0, px1, py0, py1, pd0, pd1 = _parse_padding(padding)

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = _upfirdn3d_ref(x=x, f=f, down=down, padding=[px0,px1,py0,py1,pz0,pz1], flip_filter=flip_filter)
        x = _conv3d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kw == 1 and kh == 1 and kd == 1 and (up > 1 and down == 1):
        x = _conv3d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = _upfirdn3d_ref(x=x, f=f, up=up, padding=[px0,px1,py0,py1,pz0,pz1], gain=up**2, flip_filter=flip_filter)
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = _upfirdn3d_ref(x=x, f=f, padding=[px0,px1,py0,py1,pz0,pz1], flip_filter=flip_filter)
        x = _conv3d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x

    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw, kd)
            w = w.transpose(1, 2)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kh, kw, kd)
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pz0 -= kd - 1
        pz1 -= kd - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        pzt = max(min(-pz0, -pz1), 0)
        x = _conv3d_wrapper(x=x, w=w, stride=up, padding=[pyt,pxt,pzt], groups=groups, transpose=True, flip_weight=(not flip_weight))
        x = _upfirdn3d_ref(x=x, f=f, padding=[px0+pxt,px1+pxt,py0+pyt,py1+pyt,pz0+pzt,pz1+pzt], gain=up**2, flip_filter=flip_filter)
        if down > 1:
            x = _upfirdn3d_ref(x=x, f=f, down=down, flip_filter=flip_filter)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and pz0 == pz1 and px0 >= 0 and py0 >= 0 and pz0 >= 0:
            return _conv2d_wrapper(x=x, w=w, padding=[pz1,py0,px0], groups=groups, flip_weight=flip_weight)

    # Fallback: Generic reference implementation.
    x = _upfirdn3d_ref(x=x, f=(f if up > 1 else None), up=up, padding=[px0,px1,py0,py1,pz0,pz1], gain=up**2, flip_filter=flip_filter)
    x = _conv3d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = _upfirdn3d_ref(x=x, f=f, down=down, flip_filter=flip_filter)
    return x

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
        self.weight = torch.nn.Parameter(torch.randn([in_features, out_features]) * lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        x = F.linear(
            x,
            self.weight.to(x.dtype) * self.weight_gain,
            bias=self.bias * self.bias_gain
        )

        if self.activation is not None:
            x = self.activation(x)
        return x

#----------------------------------------------------------------------------
### Mapping Network ###
def modulated_conv3d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw, kd = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw * kd) / weight.norm(float('inf'), dim=[1,2,3,4], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw, kd)
    x = conv3d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

class Conv3dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.resample_filter = torch.Tensor(resample_filter)
        self.resample_filter = self.resample_filter[None, None, :] * self.resample_filter [None, :, None]
        self.resample_filter = self.resample_filter.repeat((1,3,1)).reshape(1,3,4,4)
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 3))

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv3d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)
        x = bias_act(x, b, act=self.activation)
        return x

class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.resample_filter = torch.Tensor(resample_filter)
        self.resample_filter = self.resample_filter[None, None, :] * self.resample_filter [None, :, None]
        self.resample_filter = self.resample_filter.repeat((1,3,1)).reshape(1,3,4,4)
        self.padding = kernel_size // 2

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        in_resolution = self.resolution // self.up
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv3d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)
        x = bias_act(x, self.bias.to(x.dtype), act=self.activation)
        return x

class OutBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 3))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv3d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act(x, self.bias.to(x.dtype))
        return x

class GeneratorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels=1,                       # Number of output color channels.
        is_last=False,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.resample_filter = torch.Tensor(resample_filter)
        self.resample_filter = self.resample_filter[None, None, :] * self.resample_filter [None, :, None]
        self.resample_filter = self.resample_filter.repeat((1,3,1)).reshape(1,3,4,4)
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = OutBlock(out_channels, img_channels, w_dim=w_dim)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv3dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter)

    def forward(self, x, ws, img=None, force_fp32=False, fused_modconv=None, **layer_kwargs):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        if self.in_channels == 0:
            x = self.const
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, ws, fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, ws, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, ws, fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, ws, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, ws, fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            img = _upfirdn3d_ref(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, ws, fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32)
            img = img.add_(y) if img is not None else y

        return x, img

#----------------------------------------------------------------------------
### Synthesis Network ###
class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution, network_capacity = 16, fmap_max = 512):
        super().__init__()
        self.image_size = img_resolution
        self.latent_dim = w_dim
        self.num_layers = int(log2(self.image_size)-1)
        self.block_resolutions = [2**(i+2) for i in range(self.num_layers)]
        filters = [network_capacity * (2**i) for i in range(self.num_layers)][::-1]

        channels_dict = {res: min(filters[i], fmap_max) for i, res in enumerate(self.block_resolutions)}

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

        out = OutBlock(out_channels, 1, w_dim=self.latent_dim)
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

    def forward(self, z, **synthesis_kwargs):
        ws = self.mapping(z)
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
