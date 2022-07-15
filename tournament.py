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

def inf_train_gen(generator_train):
        while True:
            for data in generator_train:
                yield data

def load_model(path, ngpu):
    with open(os.path.join(path, 'params.pkl'), 'rb') as file:
        params = pickle.load(file)

    if params.dcgan:
        netG = Generator(params)
        netD = Discriminator(params)
    elif params.hybrid:
        netG = BigG(params)
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

def round(disc, gen, x, bound, params):
	disc = disc.to(params.device)
	gen = gen.to(params.device)
	with torch.no_grad():
		r = disc(x)
		if params.ngpu > 1:
			noise = torch.randn(x.shape[0], gen.module.dim_z,
					1, 1, 1, dtype=torch.float, device=params.device)
		else:
			noise = torch.randn(x.shape[0], gen.dim_z,
					1, 1, 1, dtype=torch.float, device=params.device)
		f = disc(gen(noise))

	disc, gen = disc.cpu(), gen.cpu()

	wrg = (r < bound).sum() + (f > bound).sum()
	wrg = wrg/(x.shape[0]*2)  

	return wrg.item()

def tournament(discs, gens, data, params):
	names = params.model_log
	res = {}
	decision_boundaries = []
	x = next(data).unsqueeze(1)
	for i, disc in enumerate(discs):
		with torch.no_grad():
			disc = disc.to(params.device)
			gen = gens[i].to(params.device)
			r = disc(x).mean()
			if params.ngpu > 1:
				noise = torch.randn(x.shape[0], gen.module.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			else:
				noise = torch.randn(x.shape[0], gen.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			f = disc(gen(noise)).mean()

		disc, gen = disc.cpu(), gen.cpu()

		decision_boundaries.append((f+r)/2)

	for n in names:
		res[n] = []
	for i, d in enumerate(discs):
		for j, g in enumerate(gens):
			if i == j:
				continue
			x = next(data).unsqueeze(1)
			wr = round(d,g,x, decision_boundaries[i], params)
			res[names[j]].append(wr)

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

	discs, gens = [], []
	for model in params.model_log:
		d,g = load_model(model, params.ngpu)
		discs.append(d)
		gens.append(g)

	dataset = DATA(path=params.data_path)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	data = inf_train_gen(generator)

	tournament(discs, gens, data, params)

if __name__ == '__main__':
	main()
