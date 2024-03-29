from data_handler import DATA_DIR
from trainer import Trainer
import argparse

def main():
	parser = argparse.ArgumentParser()
	## MISC & Hyper
	parser.add_argument('--niters', type=int, default=5000, help='Number of training iterations')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--z_size', type=int, default=512, help='Latent space dimension')
	parser.add_argument('--filterG', type=int, default=128, help='Number of filters G')
	parser.add_argument('--filterD', type=int, default=128, help='Number of filters D')
	parser.add_argument('--iterD', type=int, default=2, help='Number of D iters per iter')
	parser.add_argument('--lrG', type=float, default=5e-5, help='Learning rate G')
	parser.add_argument('--lrD', type=float, default=1e-4, help='Learning rate D')
	parser.add_argument('--data_path', type=str, default='lidc_train',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
	parser.add_argument('--steps_per_img_log', type=int, default=50, help='Image Save Iterations')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')

	## Model Type
	parser.add_argument('--dcgan', type=bool, default=False, help='Use DCGAN Base Model else ResBlock')
	parser.add_argument('--hybrid', type=bool, default=False, help='Res G with DCGAN D (Overwrites --dcgan)')
	parser.add_argument('--stylegan2', type=bool, default=False, help='Use StyleGAN2')
	parser.add_argument('--stylegan', type=bool, default=False, help='Use StyleGAN')
	parser.add_argument('--msl', type=bool, default=False, help='Use MSL Module in Discriminator (Currently only implemented with DCGAN Base Model)')
	parser.add_argument('--sngan', type=bool, default=False, help='Use SNGAN')
	parser.add_argument('--sagan', type=bool, default=False, help='Use SAGAN')
	parser.add_argument('--biggan', type=bool, default=False, help='Use BigGAN-Deep')
	
	## Loss
	parser.add_argument('--hinge', type=bool, default=False, help='Use Hinge Loss or Wasserstein loss')
	
	params = parser.parse_args()
	print(params)
	
	dataset_train = DATA_DIR(path=params.data_path)

	trainer = Trainer(dataset_train, params=params)
	trainer.train()

if __name__ == '__main__':
	main()
