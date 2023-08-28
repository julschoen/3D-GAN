import numpy as np
import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dcgan import Discriminator, Generator
from biggan import Discriminator as BigD
from biggan import Generator as BigG
from data_handler import DATA

def load_model(path, ngpu):
    with open(os.path.join(path, 'params.pkl'), 'rb') as file:
        params = pickle.load(file)

    if params.dcgan:
        netG = Generator(params)
        netD = Discriminator(params)
    else:
        netG = BigG(params)
        netD = BigD(params)

    if ngpu > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
    netG.load_state_dict(state['modelG_state_dict'])
    netD.load_state_dict(state['modelD_state_dict'])

    return netD, netG

def get_decision_bound(disc, gen, data, params):
	for x in data:
		x = x.unsqueeze(1).to(params.device)
		rs, fs = torch.tensor([]), torch.tensor([])
		with torch.no_grad():
			disc = disc.to(params.device)
			gen = gen.to(params.device)
			r = disc(x)
			if params.ngpu > 1:
				noise = torch.randn(x.shape[0], gen.module.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			else:
				noise = torch.randn(x.shape[0], gen.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			f = disc(gen(noise))

			rs = torch.concat((rs, r.detach().cpu().squeeze()))
			fs = torch.concat((fs, f.detach().cpu().squeeze()))

	disc, gen = disc.cpu(), gen.cpu()
	return ((rs.mean()+fs.mean())/2).item()

def round(disc, gen, bound, params):
	disc = disc.to(params.device)
	gen = gen.to(params.device)
	wrt = 0
	for i in range(2):
		with torch.no_grad():
			if params.ngpu > 1:
				noise = torch.randn(params.batch_size, gen.module.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			else:
				noise = torch.randn(params.batch_size, gen.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			f = disc(gen(noise))
			wrt += (f > bound).sum().item()

	disc, gen = disc.cpu(), gen.cpu()

	wrt =wrt/(params.batch_size*2)
	return wrt

def tournament(data, params):
	names = params.model_log
	res = {}
	for n in names:
		res[n] = []
	for i, name_d in enumerate(names):
		for k in range(3):
			d,g_d = load_model(name_d+f'{k}', params.ngpu)
			bound = get_decision_bound(d, g_d, data, params)
			for j, name_g in enumerate(names):
				if name_d == name_g:
						continue
				for l in range(3):
					_, g = load_model(name_g+f'{l}', params.ngpu)
					wr = round(d, g, bound, params)
					res[name_g].append(wr)

	print('------------- Tournament Results -------------')
	for n in names:
		g = res[n]
		wr = np.mean(g)
		print(f'G of {n} with Mean Win Rate of {wr:.4f}')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='test_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	params = parser.parse_args()

	dataset = DATA(path=params.data_path)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)

	tournament(generator, params)

if __name__ == '__main__':
	main()
