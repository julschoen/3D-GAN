from data_handler import DATA
from trainer import Trainer
import argparse
import os
import pickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--niters', type=int, default=5000, help='Number of training iterations')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--z_size', type=int, default=512, help='Latent space dimension')
	parser.add_argument('--filterG', type=int, default=128, help='Number of filters G')
	parser.add_argument('--filterD', type=int, default=128, help='Number of filters D')
	parser.add_argument('--iterD', type=int, default=5, help='Number of D iters per iter')
	parser.add_argument('--lrG', type=float, default=1e-4, help='Learning rate G')
	parser.add_argument('--lrD', type=float, default=1e-4, help='Learning rate D')
	parser.add_argument('--data_path', type=str, default='train_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
	parser.add_argument('--steps_per_img_log', type=int, default=50, help='Image Save Iterations')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--msl', type=bool, default=False, help='Use MSL Module in Discriminator')
	parser.add_argument('--biggan', type=bool, default=False, help='Use BigGAN')
	parser.add_argument('--biggan_deep', type=bool, default=False, help='Use BigGAN-deep')
	parser.add_argument('--att', type=bool, default=False, help='Use Attention in BigGAN')
	parser.add_argument('--hybrid', type=bool, default=False, help='Use BigGAN generator with DCGAN discriminator')
	parser.add_argument('--hinge', type=bool, default=False, help='Use Hinge Loss or Wasserstein loss')
	parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')
	params = parser.parse_args()
	#print(params)
	with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)
	#dataset_train = DATA(path=params.data_path)

	#trainer = Trainer(dataset_train, params=params)
	#trainer.train()

if __name__ == '__main__':
	main()
