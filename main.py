from data_handler import BRATS
from trainer import Params, Trainer


def main():
	dataset_train = BRATS(path='./brats_128.npz')

	conf = {
		'batch_size':16,
		'filterG': 168,
		'z_size': 512
	}
	params = Params(**conf)
	trainer = Trainer(dataset_train, log_dir='./test', params=params)
	trainer.train()

if __name__ == '__main__':
	main()
