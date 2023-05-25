import numpy as np
import os
import argparse

def make_dir(params):
	os.makedirs(params.log_dir, exist_ok=True)
	data = np.load(params.data_path)['X']
	for i, x in enumerate(data):
		np.savez_compressed(os.path.join(params.log_dir, f'{i}.npz'), X=x)
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default='train_lidc_128.npz',help='Path to data.')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	params = parser.parse_args()
	make_dir(params)

if __name__ == '__main__':
	main()
