import torch
from torch.nn.functional import grid_sample

from torch.nn.functional import interpolate

if crop_size<15: crop_size = 15

class RandomCrop3D(torch.nn.Module):
    def __init__(self, img_sz=128, n_crops=64, device='cuda'):
        super().__init__()
        self.img_sz  = tuple((img_sz, img_sz, img_sz))
        self.device=device
        self.n_crops = n_crops
        
    def __call__(self, x):
        x_ = self.crop(x[0].clone()).unsqueeze(1)
        for xi in x[1:]:
            xi_ = self.crop(xi.clone()).unsqueeze(1)
            x_ = torch.concat((x_, xi_))
        return x_

    def crop(self, x):
        crop_size = int(torch.rand(1) * self.img_sz[0])
        if crop_size<15: crop_size = 15
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, (crop_size, crop_size, crop_size))]
        x_ = self._crop(x.squeeze(), *slice_hwd)


        #d = torch.linspace(-1,1,64)
        #meshz, meshy, meshx = torch.meshgrid((d, d, d), indexing='xy')
        #grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0).to(self.device)
        #x_ = grid_sample(x_.unsqueeze(0).unsqueeze(0), grid).squeeze(0)

        x_ = interpolate(
                x_.unsqueeze(0).unsqueeze(0),
                size=(64,64,64),
                mode='trilinear'
        ).squeeze(0)
        for _ in range(self.n_crops-1):
            crop_size = int(torch.rand(1) * self.img_sz[0])
            if crop_size<15: crop_size = 15
            slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, (crop_size, crop_size, crop_size))]
            xi = self._crop(x.squeeze(), *slice_hwd)

            xi = interpolate(
                xi.unsqueeze(0).unsqueeze(0),
                size=(64,64,64),
                mode='trilinear'
            ).squeeze(0)

            #d = torch.linspace(-1,1,64)
            #meshz, meshy, meshx = torch.meshgrid((d, d, d), indexing='xy')
            #grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0).to(self.device)
            #xi = grid_sample(xi.unsqueeze(0).unsqueeze(0), grid).squeeze(0)
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
