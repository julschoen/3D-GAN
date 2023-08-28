import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

from eval_utils import *
from data_handler import DATA

def eval(params):
	dataset = DATA(path=params.data_path)
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	print(dataset.__len__())
	fid_model = get_fid_model(params.fid_checkpoint).to(params.device)
	if params.ngpu > 1:
		fid_model = nn.DataParallel(fid_model)


	ssims = []
	mmds = []
	fas = []
	fcs = []
	fss = []
	f3 = []

	for _ in range(3):
		for i, data in enumerate(generator):
			if i == 0:
				x1 = data.unsqueeze(dim=1)
			elif i == 1:
				x2 = data.unsqueeze(dim=1)
			else:
				break


		s,f = ssim(x1.cpu(),x2.cpu()), fid_3d(fid_model, x1, x2)
		m = mmd(x1.cpu(), x2.cpu())
		fa, fc, fs = fid(x1, x2, params.device)

		ssims.append(s)
		mmds.append(m)
		fas.append(fa)
		fcs.append(fc)
		fss.append(fs)
		f3.append(f)

	ssims = np.array(ssims)
	mmds = np.array(mmds)
	fas = np.array(fas)
	fcs = np.array(fcs)
	fss = np.array(fss)
	f3 = np.array(f3)
		
	print('Metrics vs 2 Data Batches')
	print(f'SSIM: {ssims.mean():.2f}+-{ssims.std():.2f}'+ 
			f'\tMMD: {mmds.mean():.2f}+-{mmds.std():.2f}'+
			f'FID ax: {fas.mean():.1f}+-{fas.std():.1f}'+
			f'\tFID cor: {fcs.mean():.1f}+-{fcs.std():.1f}'+
			f'\tFID sag: {fss.mean():.1f}+-{fss.std():.1f}'#+
			f'\t3d-FID: {f3.mean():.2f}+-{f3.std():.2f}', flush=True
			)


	ssims = []
	mmds = []
	fas = []
	fcs = []
	fss = []
	f3 = []

	for _ in range(3):
		x1 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)
		x2 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)

		s,f = ssim(x1.cpu(),x2.cpu()), fid_3d(fid_model, x1, x2)
		m = mmd(x1.cpu(), x2.cpu())
		fa, fc, fs = fid(x1, x2, params.device)

		ssims.append(s)
		mmds.append(m)
		fas.append(fa)
		fcs.append(fc)
		fss.append(fs)
		f3.append(f)

	ssims = np.array(ssims)
	mmds = np.array(mmds)
	fas = np.array(fas)
	fcs = np.array(fcs)
	fss = np.array(fss)
	f3 = np.array(f3)
		
	print('Metrics vs 2 RandN')
	print(f'SSIM: {ssims.mean():.2f}+-{ssims.std():.2f}'+ 
			f'\tMMD: {mmds.mean():.2f}+-{mmds.std():.2f}'+
			f'FID ax: {fas.mean():.1f}+-{fas.std():.1f}'+
			f'\tFID cor: {fcs.mean():.1f}+-{fcs.std():.1f}'+
			f'\tFID sag: {fss.mean():.1f}+-{fss.std():.1f}'#+
			f'\t3d-FID: {f3.mean():.2f}+-{f3.std():.2f}', flush=True
			)


	ssims = []
	mmds = []
	fas = []
	fcs = []
	fss = []
	f3 = []

	for _ in range(3)
		x1 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)
		x2 = torch.rand(32, 1,128, 128, 128, dtype=torch.float, device=params.device)

		s,f = ssim(x1.cpu(),x2.cpu()), fid_3d(fid_model, x1, x2)
		m = mmd(x1.cpu(), x2.cpu())
		fa, fc, fs = fid(x1, x2, params.device)

		ssims.append(s)
		mmds.append(m)
		fas.append(fa)
		fcs.append(fc)
		fss.append(fs)
		f3.append(f)

	ssims = np.array(ssims)
	mmds = np.array(mmds)
	fas = np.array(fas)
	fcs = np.array(fcs)
	fss = np.array(fss)
	f3 = np.array(f3)
		
	print('Metrics vs RandN/RandU')
	print(f'SSIM: {ssims.mean():.2f}+-{ssims.std():.2f}'+ 
			f'\tMMD: {mmds.mean():.2f}+-{mmds.std():.2f}'+
			f'FID ax: {fas.mean():.1f}+-{fas.std():.1f}'+
			f'\tFID cor: {fcs.mean():.1f}+-{fcs.std():.1f}'+
			f'\tFID sag: {fss.mean():.1f}+-{fss.std():.1f}'#+
			f'\t3d-FID: {f3.mean():.2f}+-{f3.std():.2f}', flush=True
			)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--data_path', type=str, default='test_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--fid_checkpoint', type=str, default='resnet_50.pth', help='Path to pretrained MedNet')
	params = parser.parse_args()
	eval(params)

if __name__ == '__main__':
	main()
