import numpy as np
import os
import argparse
import pickle

from eval_utils import *
from model import Discriminator, Generator
from biggan import Discriminator as BigD
from biggan import Generator as BigG
from data_handler import DATA

def load_gen(path):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
    		params = pickle.load(file)
	if params.hybrid or params.biggan:
		netG = BigG(params)
	else:
		netG = Generator(params)
	state = torch.load(os.path.join(path, 'models/checkpoint.pt'))['modelG_state_dict']
    netG.load_state_dict(state)
    return netG

def eval(params):
	fid_model = get_fid_model(params.fid_checkpoint).to(params.device)
    os.makedirs(params.log_dir, exist_ok=True)
    for model_path in params.models_dir:
    	print(model_path)
    	netG = load_gen(model_path).to(params.device)
    	ssims = []
		psnrs = []
		fids = []
		for i, data in enumerate(generator_train):
		    x1 = data.unsqueeze(dim=1)
		    noise = torch.randn(4, netG.dim_z, 1, 1,1,dtype=torch.float, device=params.device)
		    x2 = netG(noise)
		    s,p,f = ssim(x1,x2), psnr(x1.cpu(),x2.cpu()),fid_3d(fid_model, x1, x2)
		    ssims.append(s)
		    psnrs.append(p)
		    fids.append(f)
		    if i == 1:
		        break

		ssims = np.array(ssims)
		psnrs = np.array(psnrs)
		fids = np.array(fids)
		print(f'SSIM: {ssims.mean()}+-{ssims.std(ddof=1)}  PSNR: {psnrs.mean()}+-{psnrs.std(ddof=1)}  FID: {fids.mean()}+-{fids.std(ddof=1)}')
		np.savez_compressed(f'{model_path}_stats.npz', ssim = ssims, psnr = psnrs, fid = fids)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='train_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--model_logs', type=list, default=[], help='List of model log directories to evaluate')
	parser.add_argument('--fid_checkpoint', type=str, default='resnet_50.pth', help='Path to pretrained MedNet')
	params = parser.parse_args()

	eval(params)

if __name__ == '__main__':
	main()

