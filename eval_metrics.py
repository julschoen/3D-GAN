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
		
	print('Metrics vs 2 Data Batches')
	print(f'SSIM: {s:.2f} MMD: {m:.2f} FID ax: {fa:.1f} cor: {fc:.1f} sag: {fs:.1f} 3D: {f:.2f}')

	x1 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)
	x2 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)

	s,f = ssim(x1.cpu(),x2.cpu()), ,fid_3d(fid_model, x1, x2)
	m = mmd(fid_model, x1.cpu(), x2.cpu())
	fa, fc, fs = fid(x1, x2, params.device)
		
	print('Metrics vs 2 RandN')
	print(f'SSIM: {s:.2f} MMD: {m:.2f} FID ax: {fa:.1f} cor: {fc:.1f} sag: {fs:.1f} 3D: {f:.2f}')

	x1 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)
	x2 = torch.rand(32, 1,128, 128, 128, dtype=torch.float, device=params.device)

	s,f = ssim(x1.cpu(),x2.cpu()), ,fid_3d(fid_model, x1, x2)
	m = mmd(fid_model, x1.cpu(), x2.cpu())
	fa, fc, fs = fid(x1, x2, params.device)
		
	print('Metrics vs RandN/RandU')
	print(f'SSIM: {s:.2f} MMD: {m:.2f} FID ax: {fa:.1f} cor: {fc:.1f} sag: {fs:.1f} 3D: {f:.2f}')

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
