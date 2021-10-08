import numpy as np

def		error(x, y):
	return (x - y)

def		binary_crossentropy(x, y):
	return (-(1 / y.shape[0]) * np.sum(y * np.log(x + 1e-50) + (1 - y) * np.log((1 - x) + 1e-50)))

def		mean_squared_error(x, y):
	return (np.sum((x - y)**2) / len(x))

def		mean_absolute_error(x, y):
	return (np.sum(np.abs(x - y)) / len(x))

handler = {
	"binary_crossentropy" : binary_crossentropy,
	"mean_squared_error" : mean_squared_error,
	"mean_absolute_error" : mean_absolute_error,
	"MSE" : mean_squared_error,
	"MAE" : mean_absolute_error
}