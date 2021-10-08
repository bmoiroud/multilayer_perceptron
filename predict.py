from preprocessing import shuffle_data, feature_scaling
from pandas import read_csv
from Model import Model, evaluate
from sys import argv
from loss import binary_crossentropy as bc

import numpy as np

def		calc_score(X, Y):
	X = ['M' if x[0] >= 0.5 else 'B' for x in X]
	Y = ['M' if y[0] == 1 else 'B' for y in Y]
	valid = 0
	for i in range(len(X)):
		if (X[i] == Y[i]):
			valid += 1
	return (valid / len(X) * 100, valid, len(X))

def		calc_class_score(X, Y):
	scores = []

	X = [1 if x[0] >= .5 else 0 for x in X]
	Y = [1 if y[0] == 1 else 0 for y in Y]
	for c in np.unique(Y):
		s = 0
		j = 0
		for i in range(len(X)):
			if (c == Y[i]):
				j += 1
				if (Y[i] == X[i]):
					s += 1
		scores.append((s / j * 100))
	return (scores)

if __name__ == "__main__":
	model = Model()
	
	if (len(argv) < 3):
		print("usage: python3 predict.py <dataset.csv> <model.json>")
	try:
		data = read_csv(argv[1], header=None)
	except:
		print("invalid data")
		exit(-1)
	try:
		model.load(argv[2])
	except:
		print("invalid model")
		exit(-1)

	Y = data.iloc[:,1]
	Y = np.array([[1, 0] if y == 'M' else [0, 1] for y in Y])
	data.drop(data.columns[[0, 1, 2, 5, 7, 12, 15, 22, 25, 27]], axis=1, inplace=True)
	X = feature_scaling(data.to_numpy())
	model.predict(X)
	res = model.output
	if (res is not None and res.shape[1] == 2):
		for i in range(len(res)):
			print("predicted: {0}\ttarget: {1}\t raw:[{2:.3f}], [{3:.3f}]".format('M' if (res[i][0] >= 0.5) else 'B', 'M' if (Y[i][0] == 1) else 'B', res[i][0], res[i][1]))
		a, b, c = calc_score(res, Y)
		print("\nscore: {0:.3f}%  ({1}/{2})".format(a, b, c))
		cs = calc_class_score(res, Y)
		print("class 'M' score: {0:.3f}%\tclass 'B' score: {1:.3f}%".format(cs[0], cs[1]))
		print("binary cross entropy: {0}".format(bc(res, Y)))
