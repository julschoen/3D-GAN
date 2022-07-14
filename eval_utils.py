import numpy as np
from scipy.linalg import sqrtm
from collections import OrderedDict
import torch
from FID_ResNet import resnet50
import pytorch_fid_wrapper as FID
from pytorch_msssim import MS_SSIM
from torch.cuda.amp import autocast



def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i)
                      for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd(self, source, target):
    batch_size = int(source.size()[0])
    kernels = self.guassian_kernel(
        source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def mmd_(real, fake):
    x,y = real.squeeze(), fake.squeeze()
    xx = torch.matmul(x, torch.permute(x,(0, 3, 2, 1)))
    yy = torch.matmul(y, torch.permute(y,(0, 3, 2, 1)))
    zz = torch.matmul(x, torch.permute(y,(0, 3, 2, 1)))
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5*dxx/a)
        YY += torch.exp(-0.5*dyy/a)
        XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX+YY-2.*XY)

def psnr(real, fake):
    with torch.no_grad():
        real, fake = real+1, fake+1 
        mse = torch.mean(torch.square((real - fake)))
        if(mse == 0):
            return 100
        psnr_ = 10 * (torch.log(4/mse)/torch.log(torch.Tensor([10]))).item()
    return psnr_

def ssim(real, fake):
    with torch.no_grad():
        real = (real+1)/2
        fake = (fake+1)/2
        ms_ssim_module = MS_SSIM(data_range=1, win_size=7, size_average=True, channel=1, spatial_dims=3)
        ms_ssim_ = ms_ssim_module(real.cpu().to(torch.float32), fake.cpu().to(torch.float32)).item()
    return ms_ssim_
 
def fid_3d(model, real, fake):
    with torch.no_grad():
        act1 = model(real.cuda()).mean(dim=(2,3,4)).detach().cpu().numpy()
        act2 = model(fake.cuda()).mean(dim=(2,3,4)).detach().cpu().numpy() 
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid_ = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_

def get_fid_model(path):
    fid_model = resnet50()
    state = torch.load(path)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    fid_model.load_state_dict(new_state_dict)
    return fid_model

def fid(real, fake, device):
    FID.set_config(device=device)
    real.to(device)
    fake.to(device)
    with torch.no_grad():
        fid_ax = FID.fid(
                torch.reshape(fake.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1), 
                real_images=torch.reshape(real.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1)
                )

        fid_cor = FID.fid(
                torch.reshape(fake.to(torch.float32).transpose(2,3), (-1,1,128,128)).expand(-1,3,-1,-1), 
                real_images=torch.reshape(real.to(torch.float32).transpose(2,3), (-1,1,128,128)).expand(-1,3,-1,-1)
                )
        fid_sag = FID.fid(
                torch.reshape(fake.to(torch.float32).transpose(4,2), (-1,1,128,128)).expand(-1,3,-1,-1), 
                real_images=torch.reshape(real.to(torch.float32).transpose(4,2), (-1,1,128,128)).expand(-1,3,-1,-1)
                )
    return fid_ax, fid_cor, fid_sag
