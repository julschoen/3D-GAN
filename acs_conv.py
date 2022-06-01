import torch
import torch.nn as nn
from torch.nn import init
import math
from utils import _to_triple, _triple_same, _pair_same, acs_conv_f

class ACSConv(_ACSConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, acs_kernel_split=None, 
                 bias=True, padding_mode='zeros'):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, 0, groups, bias, padding_mode)
        if acs_kernel_split is None:
            if self.out_channels%3==0:
                self.acs_kernel_split = (self.out_channels//3,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==1:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==2:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3+1,self.out_channels//3)
        else:
            self.acs_kernel_split = acs_kernel_split


    def forward(self, x):
        return acs_conv_f(x, self.weight, self.bias, self.kernel_size, self.dilation, self.padding, self.stride, 
                            self.groups, self.out_channels, self.acs_kernel_split)


    def extra_repr(self):
        s = super().extra_repr() + ', acs_kernel_split={acs_kernel_split}'
        return s.format(**self.__dict__)

class _ACSConv(nn.Module):
    """
    Base class for ACS Convolution
    Basically the same with _ConvNd in torch.nn.
    Warnings:
        The kernel size should be the same in the three directions under this implementation.
    """
    def __init__(self,  in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super().__init__()

        assert padding_mode!='circular', 'circular padding is not supported yet.'
        stride = _to_triple(stride)
        padding = _to_triple(padding)
        dilation = _to_triple(dilation)
        output_padding = _to_triple(output_padding)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        
        if self.transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *_pair_same(kernel_size) ))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *_pair_same(kernel_size) ))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _triple_same(kernel_size) 

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'