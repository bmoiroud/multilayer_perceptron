import numpy as np

def		shuffle_data(x, y):
	p = np.random.permutation(len(x))
	return (x[p], y[p])

def		feature_scaling(x):
	return ((x - np.mean(x)) / np.std(x))
