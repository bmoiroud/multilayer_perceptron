import numpy as np
import activation as act

def		relu(x):
	return (act.heaviside(x))

def		sigmoid(x):
	return (act.sigmoid(x) * (1 - act.sigmoid(x)))

def		identity(x):
	return (np.ones(len(x)))

def		heaviside(x):
	return (np.array([0 if i != 0 else float('Inf') for i in x]))

def		tanh(x):
	return (1 - act.tanh(x)**2)

def		softmax(x):
	return (act.softmax(x) * (1 - act.softmax(x)))
	
def		softplus(x):
	return (act.sigmoid(x))

def		arctan(x):
	return (1 / (x**2 + 1))

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