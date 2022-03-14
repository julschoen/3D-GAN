from data_handler import BRATS
from trainer import Params, Trainer


def main():
	dataset_train = BRATS()

	conf = {
		
	}
	params = Params(**conf)
	trainer = Trainer(dataset_train, log_dir='drive/MyDrive/3DGan/test', params=params)
	trainer.train()

if __name__ == '__main__':
	main()
