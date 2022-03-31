from data_handler import BRATS, LIDC
from trainer import Trainer
import argparse


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--z_size', type=int, default=100, help='Latent space dimension')
	parser.add_argument('--filterG', type=int, default=128, help='Number of filters G')
	parser.add_argument('--filterD', type=int, default=128, help='Number of filters D')
	parser.add_argument('--lrG', type=float, default=2e-4, help='Learning rate G')
	parser.add_argument('--lrD', type=float, default=2e-4, help='Learning rate D')
	parser.add_argument('--lam', type=int, default=10, help='Parameter for gradient penalty')
	parser.add_argument('--data_path', type=str, default='train_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--lidc', type=bool, default=True, help='Using LIDC or BRATS')
	parser.add_argument('--steps_per_log', type=int, default=50, help='Output Iterations')
	parser.add_argument('--steps_per_img_log', type=int, default=100, help='Image Save Iterations')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	params = parser.parse_args()

	if params.lidc:
		dataset_train = LIDC(path=params.data_path)
	else:
		dataset_train = BRATS(path=params.data_path)

	trainer = Trainer(dataset_train, params=params)
	trainer.train()

if __name__ == '__main__':
	main()
