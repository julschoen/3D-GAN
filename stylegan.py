import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleGAN2Loss():
    def __init__(self, params, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.p = params
        self.device = self.p.device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=self.device)

    def run_G(self, z):
        if self.p.stylegan2:
            ws = self.G_mapping(z)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), skip_w_avg_update=True)[:, cutoff:]
            img = self.G_synthesis(ws)
        else:
            img, ws = self.G_mapping(z)
        return img, ws

    def run_D(self, img):
        logits = self.D(img)
        return logits

    def step_D(self, step, real_img, gen_z, gain=1):
        loss_Dgen = 0
        do_Dr1   = (step % 16) == 0 and (self.r1_gamma != 0)
        
        with torch.autograd.profiler.record_function('Dgen_forward'):
            gen_img, _gen_ws = self.run_G(gen_z)
            gen_logits = self.run_D(gen_img)
            loss_Dgen = torch.nn.functional.softplus(gen_logits)
        with torch.autograd.profiler.record_function('Dgen_backward'):
            loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        name = 'Dreal_Dr1' if do_Dr1 else 'Dreal'
        with torch.autograd.profiler.record_function(name + '_forward'):
            real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp)

            loss_Dreal = 0
            loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))

            loss_Dr1 = 0
            if do_Dr1:
                with torch.autograd.profiler.record_function('r1_grads'):
                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

        with torch.autograd.profiler.record_function(name + '_backward'):
            (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        return (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain), loss_Dgen.mean().mul(gain)

    def step_G(self, step, gen_z, gain=1):
        do_Gpl   = (step % 16) == 0 and self.p.stylegan2

        with torch.autograd.profiler.record_function('Gmain_forward'):
            gen_img, _gen_ws = self.run_G(gen_z) # May get synced by Gpl.
            gen_logits = self.run_D(gen_img)
            loss_Gmain = torch.nn.functional.softplus(-gen_logits)
        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                loss_Gpl = pl_penalty * self.pl_weight
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        loss = (gen_img[:, 0, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain) + loss_Gmain.mean().mul(gain) if do_Gpl else loss_Gmain.mean().mul(gain)

        return loss, gen_img

#----------------------------------------------------------------------------

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

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

def fma(a, b, c): # => a * b + c
    return _FusedMultiplyAdd.apply(a, b, c)

class _FusedMultiplyAdd(torch.autograd.Function): # a * b + c
    @staticmethod
    def forward(ctx, a, b, c): # pylint: disable=arguments-differ
        out = torch.addcmul(c, a, b)
        ctx.save_for_backward(a, b)
        ctx.c_shape = c.shape
        return out

    @staticmethod
    def backward(ctx, dout): # pylint: disable=arguments-differ
        a, b = ctx.saved_tensors
        c_shape = ctx.c_shape
        da = None
        db = None
        dc = None

        if ctx.needs_input_grad[0]:
            da = _unbroadcast(dout * b, a.shape)

        if ctx.needs_input_grad[1]:
            db = _unbroadcast(dout * a, b.shape)

        if ctx.needs_input_grad[2]:
            dc = _unbroadcast(dout, c_shape)

        return da, db, dc

def _unbroadcast(x, shape):
    extra_dims = x.ndim - len(shape)
    assert extra_dims >= 0
    dim = [i for i in range(x.ndim) if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)]
    if len(dim):
        x = x.sum(dim=dim, keepdim=True)
    if extra_dims:
        x = x.reshape(-1, *x.shape[extra_dims+1:])
    assert x.shape == shape
    return x

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
    sx, sy, sz = scaling
    assert sx >= 1 and sy >= 1 and sz >= 1
    return sx, sy, sz

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
    
    fw = int(fw)
    fh = int(fh)
    fd = int(fd)
 
    return fw, fh, fd

def _get_weight_shape(w):
    shape = [int(sz) for sz in w.shape]
    return shape

def _conv3d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    if not flip_weight: 
        w = w.flip([2, 3, 4])

    op = F.conv_transpose3d if transpose else F.conv3d

    return op(x, w, stride=stride, padding=padding, groups=groups)

def _upfirdn3d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    if f is None:
        f = torch.ones([1, 1, 1], dtype=x.dtype, device=x.device)
    batch_size, num_channels, in_height, in_width, in_depth = x.shape
    upx, upy, upz = _parse_scaling(up)
    downx, downy, downz = _parse_scaling(down)
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)

    # Upsample
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1, in_depth, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1, 0, 0, 0, upz -1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx, in_depth * upz])

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0), max(padz0, 0), max(padz1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0), max(-padz0, 0) : x.shape[4] - max(-padz1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 3))
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))
    
    # Convolve with the filter.
    f = f[np.newaxis,np.newaxis].repeat([num_channels,1] + [1] * f.ndim)
    f = f.to(dtype=x.dtype, device=x.device)
    x = F.conv3d(x, f, groups=num_channels)
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
    px0, px1, py0, py1, pz0, pz1 = _parse_padding(padding)

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
        pz0 += (fd + up - 1) // 2
        pz1 += (fd - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2
        pz0 += (fd - down + 1) // 2
        pz1 += (fd - down) // 2

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
            return _conv3d_wrapper(x=x, w=w, padding=[pz1,py0,px0], groups=groups, flip_weight=flip_weight)

    # Fallback: Generic reference implementation.
    x = _upfirdn3d_ref(x=x, f=(f if up > 1 else None), up=up, padding=[px0,px1,py0,py1,pz0,pz1], gain=up**2, flip_filter=flip_filter)
    x = _conv3d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = _upfirdn3d_ref(x=x, f=f, down=down, flip_filter=flip_filter)
    return x

#----------------------------------------------------------------------------
### Mapping Network ###
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
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

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if layer_features is None:
            layer_features = w_dim

        features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=True):
        # Embed, normalize, and concat inputs.
        x = None
        z = z.squeeze()
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------
### Synthesis Blocks ###
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
        weight = weight * (1 / np.sqrt(in_channels * kh * kw * kd) / weight.norm(float('inf'), dim=[1,2,3,4], keepdim=True)) # max_Ikkk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I
    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4,5]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]
    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1, 1)
        x = conv3d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1, 1)
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
        self.resample_filter = self.resample_filter.repeat((1,4,1)).reshape(4,4,4)
        self.resample_filter /= self.resample_filter.sum()
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
        self.resample_filter = self.resample_filter.repeat((1,4,1)).reshape(4,4,4)
        self.resample_filter /= self.resample_filter.sum()
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
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution, self.resolution], device=x.device) * self.noise_strength
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

class SynthesisBlock(torch.nn.Module):
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
        self.resample_filter = self.resample_filter.repeat((1,4,1)).reshape(4,4,4)
        self.resample_filter /= self.resample_filter.sum()
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution, resolution]))

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
        w_iter = iter(ws.unbind(dim=1))

        if self.in_channels == 0:
            x = self.const
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1, 1])
        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        
        # ToRGB.
        if img is not None:
            up = 2
            padding = 0
            upx, upy, upz = _parse_scaling(up)
            padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)
            fw, fh, fd = _get_filter_size(self.resample_filter)
            p = [
                padx0 + (fw + upx - 1) // 2,
                padx1 + (fw - upx) // 2,
                pady0 + (fh + upy - 1) // 2,
                pady1 + (fh - upy) // 2,
                padz0 + (fd + upz - 1) // 2,
                padz1 + (fd - upz) // 2,
            ]
            img = _upfirdn3d_ref(img, self.resample_filter, up=up, padding=p, gain=1*upx*upy*upz)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32)
            img = img.add_(y) if img is not None else y

        return x, img

#----------------------------------------------------------------------------
### Synthesis Network ###
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution=128,         # Output image resolution.
        img_channels=1,             # Number of color channels.
        channel_base= 4096,         # Overall multiplier for the number of channels.
        channel_max= 512,           # Maximum number of channels in any layer.
        num_fp16_res= 0,            # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, cur_ws, img, **block_kwargs)
        return torch.tanh(img)

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
        self.synthesis = SynthesisNetwork(w_dim=self.w_dim, img_resolution=self.img_resolution, channel_base=params.filterG)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=self.z_dim, w_dim=self.w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, **synthesis_kwargs):
        ws = self.mapping(z)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img, ws

#----------------------------------------------------------------------------
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.resample_filter = torch.Tensor(resample_filter)
        self.resample_filter = self.resample_filter[None, None, :] * self.resample_filter [None, :, None]
        self.resample_filter = self.resample_filter.repeat((1,4,1)).reshape(4,4,4)
        self.resample_filter /= self.resample_filter.sum()

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv3dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter))

        self.conv0 = Conv3dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter))

        self.conv1 = Conv3dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter)

        if architecture == 'resnet':
            self.skip = Conv3dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            down = 2
            padding = 0
            downx, downy, downz = _parse_scaling(down)
            padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)
            fw, fh, fd = _get_filter_size(self.resample_filter)
            p = [
                padx0 + (fw - downx + 1) // 2,
                padx1 + (fw - downx) // 2,
                pady0 + (fh - downy + 1) // 2,
                pady1 + (fh - downy) // 2,
                padz0 + (fd - downz + 1) // 2,
                padz1 + (fd - downz) // 2,
            ]
            img = _upfirdn3d_ref(x, self.resample_filter, down=down, padding=p, flip_filter=False) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        return x, img

#----------------------------------------------------------------------------
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W, D = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
   
        y = x.reshape(G, -1, F, c, H, W, D)    # [GnFcHWD] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHWD] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHWD]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHWD]  Calc stddev over group.
        y = y.mean(dim=[2,3,4,5])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1, 1)          # [nF111]   Add missing dimensions.
        y = y.repeat(G, 1, H, W, D)            # [NFHWD]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHWD]   Append to input as new channels.

        return x

#----------------------------------------------------------------------------
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv3dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv3dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 3), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float32
        memory_format = torch.contiguous_format
        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        return x

#----------------------------------------------------------------------------
class Discriminator(torch.nn.Module):
    def __init__(self,
        params,
        img_resolution=128,                 # Input resolution.
        img_channels=1,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        channel_base = params.filterD
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)


        common_kwargs = dict(img_channels=img_channels, architecture=architecture)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        self.b4 = DiscriminatorEpilogue(channels_dict[4], resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, **block_kwargs):
        x = None
        for i,res in enumerate(self.block_resolutions):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
        
        x = self.b4(x, img)
        return x

#----------------------------------------------------------------------------
def meanAndStd(f, eps=1e-5):
    batch_size, num_channels = f.shape[:2]
    
    # calculate the standard deviation across all elements per instance per channel
    feat_var = f.view(batch_size, num_channels, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(batch_size, num_channels, 1, 1, 1) #re-expand tp the original size
    
    # calculate the mean of the feature maps per instance per channel
    feat_mean = f.view(batch_size, num_channels, -1).mean(dim=2).view(batch_size, num_channels, 1, 1 ,1)
    return feat_mean, feat_std

def AdaIN(f_content, f_style):
    
    # get the shape of the input content feature map
    f_content_shape = f_content.size()
    nc = f_content_shape[1]
    
    # calculate the instance mean and standard deviation for content
    content_mean, content_std = meanAndStd(f_content)
    
    # split the f_style into the mean and std
    style_mean = f_style[:,0:nc][:,:,None,None,None].repeat((1,1,f_content_shape[2],f_content_shape[3],f_content_shape[4]))
    style_std = f_style[:,nc::][:,:,None,None,None].repeat((1,1,f_content_shape[2],f_content_shape[3],f_content_shape[4]))

    # normallise the content features
    f_content_norm = (f_content - content_mean.expand(f_content_shape)) / content_std.expand(f_content_shape)
    
    
    return f_content_norm * style_std + style_mean

def conditionalSplit(w,swapPoint,layerCtr,alreadySplit):
    
    if (layerCtr==swapPoint) & (alreadySplit!=True):
        
        w = w[torch.randperm(w.shape[0])] # shuffle along the batch dimension
        
    return w

class Self_Attention(nn.Module):
    def __init__(self,nc_in):
        super(Self_Attention,self).__init__()
        self.CBar = nc_in//8
        
        self.Wf = nn.Conv3d(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1)
        self.Wg = nn.Conv3d(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1)
        self.Wh = nn.Conv3d(in_channels = nc_in , out_channels = self.CBar , kernel_size= 1)
        self.Wv = nn.Conv3d(in_channels = nc_in//8 , out_channels = nc_in , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))        

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x,singleOutput=True):
#        print('Self attention!')
        batch_size, C, W ,H, D = x.size()
        f  = self.Wf(x).view(batch_size,-1,W*H*D)
        g =  self.Wg(x).view(batch_size,-1,W*H*D)
        s =  torch.bmm(f.permute(0,2,1),g)
        bta = self.softmax(s) 
        h = self.Wh(x).view(batch_size,-1,W*H*D)

        out = self.Wv(torch.bmm(h,bta.permute(0,2,1)).view(batch_size,self.CBar,W ,H, D))
        
        out = self.gamma*out + x
        
        if singleOutput == True:
            return out
        else:
            return out, bta
        

class styleGAN_gen(nn.Module):
    def __init__(self, params, styleMixing=True, mappingNet=True, SAflag=False):
        super(styleGAN_gen, self).__init__()
        self.p = params
        self.nz = self.p.z_size
        self.styleMixing = styleMixing
        self.SAflag = SAflag
        self.mappingNet = mappingNet

        if self.mappingNet==True:
            self.latentMapping = nn.Sequential(
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.nz,self.nz),
                    nn.LeakyReLU(0.2, inplace=True)
                )
        
        self.C1 = nn.Sequential(
            nn.Conv3d(512, 512, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C2 = nn.Sequential(
            nn.Conv3d(512, 256, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C3 = nn.Sequential(
            nn.Conv3d(256, 256, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C4 = nn.Sequential(
            nn.Conv3d(256, 128, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C5 = nn.Sequential(
            nn.Conv3d(128, 128, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C6 = nn.Sequential(
            nn.Conv3d(128, 64, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C7 = nn.Sequential(
            nn.Conv3d(64, 64, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        if (self.SAflag == True):
            self.SA = Self_Attention(64)
        
        self.C8 = nn.Sequential(
            nn.Conv3d(64, 32, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))

        self.C9 = nn.Sequential(
            nn.Conv3d(32, 32, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C10 = nn.Sequential(
            nn.Conv3d(32, 16, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.LeakyReLU(0.2,inplace=True))
        
        self.C_out = nn.Sequential(
            nn.Conv3d(16, 1, (3,3,3), (1,1,1), (1,1,1), bias=False),
            nn.Tanh())
        
        # affine mappings produce style stds and means (so, need 2x the output channels)
        self.A1 = nn.Linear(512,2*512)
        self.A2 = nn.Linear(512,2*512)
        self.A3 = nn.Linear(512,2*256)
        self.A4 = nn.Linear(512,2*256)
        self.A5 = nn.Linear(512,2*128)
        self.A6 = nn.Linear(512,2*128)
        self.A7 = nn.Linear(512,2*64)
        self.A8 = nn.Linear(512,2*64)
        self.A9 = nn.Linear(512,2*32)
        self.A10 = nn.Linear(512,2*32)
        self.A11 = nn.Linear(512,2*16)

    def forward(self, z_in, w_passed=False):
        batch_size = z_in.shape[0]
        num_channels = z_in.shape[1]
        assert num_channels == 512
        
        if (self.styleMixing == True) & (self.training == True):
            swapPoint = torch.randint(6,(1,1,1)).detach().item()
            alreadySplit = False
            layerCtr = 0
        
        if (self.mappingNet == False) or (w_passed == True):
            w = torch.squeeze(z_in)
        else: # if mapping network is in use and w_passed is false, use the mapping network
            w = self.latentMapping(torch.squeeze(z_in))
        
        constIm = torch.ones((batch_size,num_channels,4,4,4)).to(z_in.dtype).to(z_in.device)
        
        h = AdaIN(constIm,self.A1(w))        
       
        h = self.C1(h)
        
        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A2(w))        
        
        h = F.interpolate(h,scale_factor=2)
        
        h = self.C2(h)
        
        h = AdaIN(h,self.A3(w))  
        
        h = self.C3(h)

        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A4(w))  

        h = F.interpolate(h,scale_factor=2)

        h = self.C4(h)

        h = AdaIN(h,self.A5(w))  
        
        h = self.C5(h)

        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A6(w))  

        h = F.interpolate(h,scale_factor=2)
 
        h = self.C6(h)

        h = AdaIN(h,self.A7(w))  

        h = self.C7(h)
        
        if (self.SAflag == True):
            h, attn = self.SA(h)

        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A8(w)) 
        
        h = F.interpolate(h,scale_factor=2)
        
        h = self.C8(h)
        
        h = AdaIN(h,self.A9(w))  
        
        h = self.C9(h)
        
        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A10(w)) 
        
        h = F.interpolate(h,scale_factor=2)
        
        h = self.C10(h)
        
        if (self.styleMixing == True) & (self.training == True):
            w = conditionalSplit(w,swapPoint,layerCtr,alreadySplit)
            layerCtr += 1
        
        h = AdaIN(h,self.A11(w))  
        
        h = self.C_out(h)
        
        return h, w