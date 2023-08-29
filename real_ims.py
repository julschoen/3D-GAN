import numpy as np
import argparse
from torch.utils.data import DataLoader

from data_handler import DATA


def make_ims(params):
	dataset = DATA(path=params.data_path)
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True)
	
	for i, data in enumerate(generator):
		x1 = data.unsqueeze(dim=1)
		data_name = params.data_path.split('_')[1]
		np.savez_compressed(f'{data_name}_real.npz', x1.cpu().numpy())
		break


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='test_lidc_128.npz',help='Path to data.')
	params = parser.parse_args()
	make_ims(params)

if __name__ == '__main__':
	main()
