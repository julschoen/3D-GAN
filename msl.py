import torch
from torch.nn.functional import interpolate

class RandomCrop3D(torch.nn.Module):
    def __init__(self, img_sz=128, n_crops=64, device='cuda'):
        super().__init__()
        self.img_sz  = tuple((img_sz, img_sz, img_sz))
        self.device=device
        self.n_crops = n_crops
        
    def __call__(self, x):
        x_ = self.crop(x[0].clone()).unsqueeze(0)
        for xi in x[1:]:
            xi_ = self.crop(xi.clone()).unsqueeze(0)
            x_ = torch.concat((x_, xi_))
        return x_

    def crop(self, x):
        crop_size = int(torch.rand(1) * (self.img_sz[0]-15))+15
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, (crop_size, crop_size, crop_size))]
        x_ = self._crop(x.squeeze(), *slice_hwd)
        x_ = interpolate(
                x_.unsqueeze(0).unsqueeze(0),
                size=(64,64,64)
        ).squeeze(0)
        for _ in range(self.n_crops-1):
            int(torch.rand(1) * (self.img_sz[0]-15))+15
            slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, (crop_size, crop_size, crop_size))]
            xi = self._crop(x.squeeze(), *slice_hwd)

            xi = interpolate(
                xi.unsqueeze(0).unsqueeze(0),
                size=(64,64,64)
            ).squeeze(0)
            x_ = torch.cat((x_, xi))
        return x_
        
    @staticmethod
    def _get_slice(sz, crop_sz):
        try : 
            lower_bound = torch.randint(sz-crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except: 
            return (None, None)
    
    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]
