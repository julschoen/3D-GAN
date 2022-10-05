import numpy as np

path =  'brats_128.npz'
data = np.load(path)['X']

test_ind = np.random.choice(data, size=int(len(dirs)*0.1))
train = []
test = []
for i, d in enumerate(data):
	if i in test_ind:
		test.append(d)
	else:
		train.append(d)

train, test = np.array(train), np.array(test)

np.savez_compressed('train_brats.npz', X=train)
np.savez_compressed('test_brats.npz', X=test)