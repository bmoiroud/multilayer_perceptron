from preprocessing import shuffle_data, feature_scaling
from pandas import read_csv
from Model import Model
from sys import argv
from loss import handler

import numpy as np
import time
np.random.seed(838881564)

if __name__ == "__main__":
	try:
		data = read_csv(argv[1], header=None)	
	except:
		print("usage: python3 train.py <dataset.csv>")
		exit(-1)

	Y = data.iloc[:,1]
	Y = np.array([[1, 0] if y == 'M' else [0, 1] for y in Y])
	
	data.drop(data.columns[[0, 1, 2, 5, 7, 12, 15, 22, 25, 27]], axis=1, inplace=True)

	X = feature_scaling(data.to_numpy())
		
	X, Y = shuffle_data(X, Y)
	lim = int(len(X) * .8)
	val_X = X[lim:]
	val_Y = Y[lim:]
	X = X[:lim]
	Y = Y[:lim]

	model = Model(data.shape[1])
	model.dense(30, "relu")
	model.dense(30, "relu")
	model.dense(30, "relu")
	model.dense(2, "softmax")
	model.early_stop(delta=1e-7, monitor="val_loss", keep_best=True)
	model.train(X, Y, val_X, val_Y, loss="binary_crossentropy", optimizer="adam", learning_rate=2e-4, epochs=5000, batch_size=64, plot=True)
	model.save("model.json")