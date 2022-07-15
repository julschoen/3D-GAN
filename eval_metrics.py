import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

from eval_utils import *
from data_handler import DATA

def eval(params):
	dataset = DATA(path=params.data_path)
	print(dataset.__len__())
	fid_model = get_fid_model(params.fid_checkpoint).to(params.device)
	if params.ngpu > 1:
		fid_model = nn.DataParallel(fid_model)
	os.makedirs(params.log_dir, exist_ok=True)

	for i, data in enumerate(generator):
		if i == 0:
			x1 = data.unsqueeze(dim=1)
		elif i == 1:
			x2 = data.unsqueeze(dim=1)
		else:
			break


	s,p,f = ssim(x1.cpu(),x2.cpu()), psnr(x1.cpu(),x2.cpu()),fid_3d(fid_model, x1, x2)
	m = mmd(fid_model, x1.cpu(), x2.cpu())
	fa, fc, fs = fid(x1_, x2_, params.device)
		
	print('Metrics vs 2 Data Batches')
	print(f'SSIM: {s:.4f} PSNR: {p:.4f} MMD: {m:.4f} FID ax: {fa:.4f} cor: {fc:.4f} sag: {fs:.4f} 3D: {f:.4f}')

	x1 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)
	x2 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)

	s,p,f = ssim(x1.cpu(),x2.cpu()), psnr(x1.cpu(),x2.cpu()),fid_3d(fid_model, x1, x2)
	m = mmd(fid_model, x1.cpu(), x2.cpu())
	fa, fc, fs = fid(x1_, x2_, params.device)
		
	print('Metrics vs 2 RandN')
	print(f'SSIM: {s:.4f} PSNR: {p:.4f} MMD: {m:.4f} FID ax: {fa:.4f} cor: {fc:.4f} sag: {fs:.4f} 3D: {f:.4f}')

	x1 = torch.randn(32, 1,128, 128, 128, dtype=torch.float, device=params.device)
	x2 = torch.rand(32, 1,128, 128, 128, dtype=torch.float, device=params.device)

	s,p,f = ssim(x1.cpu(),x2.cpu()), psnr(x1.cpu(),x2.cpu()),fid_3d(fid_model, x1, x2)
	m = mmd(fid_model, x1.cpu(), x2.cpu())
	fa, fc, fs = fid(x1_, x2_, params.device)
		
	print('Metrics vs RandN/RandU')
	print(f'SSIM: {s:.4f} PSNR: {p:.4f} MMD: {m:.4f} FID ax: {fa:.4f} cor: {fc:.4f} sag: {fs:.4f} 3D: {f:.4f}')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='test_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--fid_checkpoint', type=str, default='resnet_50.pth', help='Path to pretrained MedNet')
	params = parser.parse_args()
	eval(params)

if __name__ == '__main__':
	main()
