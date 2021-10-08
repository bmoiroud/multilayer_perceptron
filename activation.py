import numpy as np

def		relu(x):
	return (np.maximum(x, 0))

def		sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def		identity(x):
	return (x)

def		heaviside(x):
	return (1 * (x > 0))

def		tanh(x):
	ex1 = np.exp(x)
	ex2 = np.exp(-x)
	return ((ex1 - ex2) / (ex1 + ex2))

def		softmax(x):
	e = np.exp(x - np.max(x, axis=(0 if x.ndim == 1 else 1), keepdims=True))
	return(e / np.sum(e, axis=(0 if x.ndim == 1 else 1), keepdims=True))

def		softplus(x):
	return(np.log(1 + np.exp(x)))

def		arctan(x):
	return(np.arctan(x))

handler = {
	"relu": relu,
	"sigmoid" : sigmoid,
	"identity" : identity,
	"heaviside" : heaviside,
	"tanh" : tanh,
	"softmax" : softmax,
	"softplus" : softplus,
	"arctan" : arctan
}