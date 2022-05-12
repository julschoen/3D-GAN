import numpy as np
import os
import argparse
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn

from eval_utils import *
from model import Discriminator, Generator
from biggan import Discriminator as BigD
from biggan import Generator as BigG
from data_handler import DATA

def load_gen(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	if params.hybrid or params.biggan:
		netG = BigG(params)
	else:
		netG = Generator(params)

	if ngpu > 1:
		netG = nn.DataParallel(netG)
	state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
	netG.load_state_dict(state['modelG_state_dict'])

	return netG

def eval(params):
	dataset = DATA(path=params.data_path)
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	fid_model = get_fid_model(params.fid_checkpoint).to(params.device)
	if params.ngpu > 1:
		fid_model = nn.DataParallel(fid_model)
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path)
		netG = load_gen(model_path, params.ngpu).to(params.device)
		ssims = []
		psnrs = []
		fids = []
		fids_ax = []
		fids_cor = []
		fids_sag = []
		for i, data in enumerate(generator):
			x1 = data.unsqueeze(dim=1).to(params.device)
			if params.ngpu > 1:
				noise = torch.randn(data.shape[0], netG.module.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			else:
				noise = torch.randn(data.shape[0], netG.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			x2 = netG(noise)
			s,p,f = ssim(x1,x2), psnr(x1.cpu(),x2.cpu()),fid_3d(fid_model, x1, x2)
			#if i<3:
			#	fa, fc, fs = fid(x1, x2)
			#	fids_ax.append(fa)
				#fids_cor.append(fc)
				#fids_sag.append(fs)
			ssims.append(s)
			psnrs.append(p)
			fids.append(f)
			

		ssims = np.array(ssims)
		psnrs = np.array(psnrs)
		fids = np.array(fids)
		fids_ax = np.array(fids_ax)
		fids_cor = np.array(fids_cor)
		fids_sag = np.array(fids_sag)
		print(f'SSIM: {ssims.mean():.6f}+-{ssims.std(ddof=1):.6f}'+ 
			f'\tPSNR: {psnrs.mean():.6f}+-{psnrs.std(ddof=1):.6f}'+
			#f'\tFID ax: {fids_ax.mean():.6f}+-{fids_ax.std(ddof=1):.6f}'+
			#f'\tFID cor: {fids_cor.mean():.6f}+-{fids_cor.std(ddof=1):.6f}'+
			#f'\tFID sag: {fids_sag.mean():.6f}+-{fids_sag.std(ddof=1):.6f}'+
			f'\t3d-FID: {fids.mean():.6f}+-{fids.std(ddof=1):.6f}')
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}_stats.npz'),
			ssim = ssims, psnr = psnrs, fid = fids, fid_ax=fids_ax, fid_cor=fids_cor, fid_sag=fids_sag)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='train_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--model_log', nargs='+', type=str, required=True, help='Model log directories to evaluate')
	parser.add_argument('--fid_checkpoint', type=str, default='resnet_50.pth', help='Path to pretrained MedNet')
	params = parser.parse_args()
	eval(params)

if __name__ == '__main__':
	main()


