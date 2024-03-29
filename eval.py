import numpy as np
import os
import argparse
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast

from eval_utils import *
from dcgan import Discriminator, Generator
from biggan import Discriminator as BigD
from biggan import Generator as BigG
from data_handler import DATA

def load_gen(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	if params.dcgan:
		netG = Generator(params)
	else:
		netG = BigG(params)
		

	if ngpu > 1:
		netG = nn.DataParallel(netG)
	state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
	netG.load_state_dict(state['modelG_state_dict'])

	return netG

def eval(params):
	dataset = DATA(path=params.data_path)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True)
	fid_model = get_fid_model(params.fid_checkpoint).to(params.device)
	if params.ngpu > 1:
		fid_model = nn.DataParallel(fid_model)
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path, flush=True)
		ssims = []
		mmds = []
		fids = []
		fids_ax = []
		fids_cor = []
		fids_sag = []
		for j in range(3):
			model_run = model_path+f'{j}'
			netG = load_gen(model_run, params.ngpu).to(params.device)
			with torch.no_grad():
				for i, data in enumerate(generator):
					x1 = data.unsqueeze(dim=1)
					if params.ngpu > 1:
						noise = torch.randn(data.shape[0], netG.module.dim_z,
								1, 1, 1, dtype=torch.float, device=params.device)
					else:
						noise = torch.randn(data.shape[0], netG.dim_z,
								1, 1, 1, dtype=torch.float, device=params.device)
					x2 = netG(noise)
					
					s,f = ssim(x1.cpu(),x2.cpu()) ,fid_3d(fid_model, x1, x2)
					m = mmd(x1.cpu(), x2.cpu())
					ssims.append(s)
					fids.append(f)
					mmds.append(m)

					fa, fc, fs = fid(x1, x2, params.device)
					fids_ax.append(fa)
					fids_cor.append(fc)
					fids_sag.append(fs)

					if i == 3:
						np.savez_compressed(f'{model_run}_ims.npz', x2[:6].cpu().numpy())

		ssims = np.array(ssims)
		mmds = np.array(mmds)
		fids = np.array(fids)
		fids_ax = np.array(fids_ax)
		fids_cor = np.array(fids_cor)
		fids_sag = np.array(fids_sag)

		print(f'SSIM: {ssims.mean():.2f}+-{ssims.std():.2f}'+ 
			f'\tMMD: {mmds.mean():.2f}+-{mmds.std():.2f}'+
			f'FID ax: {fids_ax.mean():.1f}+-{fids_ax.std():.1f}'+
			f'\tFID cor: {fids_cor.mean():.1f}+-{fids_cor.std():.1f}'+
			f'\tFID sag: {fids_sag.mean():.1f}+-{fids_sag.std():.1f}'#+
			f'\t3d-FID: {fids.mean():.2f}+-{fids.std():.2f}', flush=True
			)
		p = model_path.split('/')[1]
		#np.savez_compressed(os.path.join(params.log_dir,f'{p}_stats.npz'), fid_ax=fids_ax, fid_cor=fids_cor, fid_sag=fids_sag)
		np.savez_compressed(os.path.join(params.log_dir,f'{p}_stats.npz'),
				ssim = ssims, mmds=mmds, fid = fids, fid_ax=fids_ax, fid_cor=fids_cor, fid_sag=fids_sag)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='test_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	parser.add_argument('--fid_checkpoint', type=str, default='resnet_50.pth', help='Path to pretrained MedNet')
	params = parser.parse_args()
	eval(params)

if __name__ == '__main__':
	main()
