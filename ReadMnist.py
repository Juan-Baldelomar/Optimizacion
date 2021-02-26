
import numpy, pickle, gzip
import matplotlib.pyplot as plt



def loadData():
	with gzip.open('mnist.pkl.gz', 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		train_set, val_set, test_set = u.load()
		f.close()

		# idx = 1
		# im = train_set[0][idx].reshape(28, -1)
		# plt.imshow(im, cmap=plt.cm.gray)
		# plt.show()
		# print(train_set[0][0])
		# print(train_set[0].shape)
		# print(test_set[0].shape)

	return train_set, val_set, test_set



loadData()


