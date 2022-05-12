import numpy as np
from scipy.linalg import sqrtm
from collections import OrderedDict
import torch
from FID_ResNet import resnet50
import pytorch_fid_wrapper as FID
from pytorch_msssim import ms_ssim

def psnr(real, fake):
    mse = torch.mean(torch.square((real - fake)))
    if(mse == 0):
        return 100
    return 10 * (torch.log(1/mse)/torch.log(torch.Tensor([10]))).item()

def ssim(real, fake):
    real = (real+1)/2
    fake = (fake+1)/2
    return ms_ssim(real, fake, data_range=1, size_average=True, channel=1)
 
# calculate frechet inception distance
def fid_3d(model, real, fake):
    # calculate activations
    model.cuda()
    act1 = model(real.cuda()).mean(dim=(2,3,4)).detach().cpu().numpy()
    act2 = model(fake.cuda()).mean(dim=(2,3,4)).detach().cpu().numpy() 
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_fid_model(path):
    fid_model = resnet50()
    state = torch.load(path)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    fid_model.load_state_dict(new_state_dict)
    return fid_model

def fid(real, fake):
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














