import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
import collections.abc
from itertools import repeat
import math

def conv3D_output_shape_f(i, input_shape, kernel_size, dilation, padding, stride):
    """
    Calculate the original output size assuming the convolution is nn.Conv3d based on 
    input size, kernel size, dilation, padding and stride.
    """
    return math.floor((input_shape[i]-kernel_size[i]-(dilation[i]-1)*
                                        (kernel_size[i]-1)+2*padding[i])
                                    /stride[i])+1
    
def acs_conv_f(x, weight, bias, kernel_size, dilation, padding, stride, groups, out_channels, acs_kernel_split):
    B, C_in, *input_shape = x.shape
    C_out = weight.shape[0]
    assert groups==1 or groups==C_in==C_out, "only support standard or depthwise conv"

    conv3D_output_shape = (conv3D_output_shape_f(0, input_shape, kernel_size, dilation, padding, stride), 
                            conv3D_output_shape_f(1, input_shape, kernel_size, dilation, padding, stride), 
                            conv3D_output_shape_f(2, input_shape, kernel_size, dilation, padding, stride))
            
    weight_a = weight[0:acs_kernel_split[0]].unsqueeze(2)
    weight_c = weight[acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])].unsqueeze(3)
    weight_s = weight[(acs_kernel_split[0]+acs_kernel_split[1]):].unsqueeze(4)
    if groups==C_in==C_out:
        # depth-wise
        x_a = x[:, 0:acs_kernel_split[0]]
        x_c = x[:, acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])]
        x_s = x[:, (acs_kernel_split[0]+acs_kernel_split[1]):]
        group_a = acs_kernel_split[0]
        group_c = acs_kernel_split[1]
        group_s = acs_kernel_split[2]
    else:
        # groups=1
        x_a = x_c = x_s = x
        group_a = group_c = group_s = 1

    f_out = []
    if acs_kernel_split[0]>0:
        a = F.conv3d(x_a if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,padding[0],padding[0]),'constant',0)[:,:,
                                            kernel_size[0]//2:kernel_size[0]//2+(conv3D_output_shape[0]-1)*stride[0]+1,
                                            :,:], 
                                            weight=weight_a, bias=None, 
                                            stride=stride,
                                            padding=(0,padding[1],padding[2]),
                                            dilation=dilation,
                                            groups=group_a)                
        f_out.append(a)
    if acs_kernel_split[1]>0:
        c = F.conv3d(x_c if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,padding[1],padding[1]),'constant',0)[:,:,:,
                                            kernel_size[1]//2:kernel_size[1]//2+stride[1]*(conv3D_output_shape[1]-1)+1,
                                            :], 
                                            weight=weight_c, bias=None,                                     
                                            stride=stride,
                                            padding=(padding[0],0,padding[2]),
                                            dilation=dilation,
                                            groups=group_c)
        f_out.append(c)
    if acs_kernel_split[2]>0:
        s = F.conv3d(x_s if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (padding[2],padding[2]),'constant',0)[:,:,:,:,
                                            kernel_size[2]//2:kernel_size[2]//2+stride[2]*(conv3D_output_shape[2]-1)+1
                                            ], 
                                            weight=weight_s, 
                                            bias=None, 
                                            stride=stride,
                                            padding=(padding[0],padding[1],0),
                                            dilation=dilation,
                                            groups=group_s)
        f_out.append(s)
    f = torch.cat(f_out, dim=1)
    
    if bias is not None:
        f += bias.view(1,out_channels,1,1,1)

    return f

def _ntuple_same(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            assert len(set(x))==1, 'the size of kernel must be the same for each side'
            return tuple(repeat(x[0], n))
    return parse

def _to_ntuple(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            if len(set(x))==1:
                return tuple(repeat(x[0], n))
            else:
                assert len(x)==n , 'wrong format'
                return x
    return parse

_pair_same = _ntuple_same(2)
_triple_same = _ntuple_same(3)

_to_pair = _to_ntuple(2)
_to_triple = _to_ntuple(3)


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
