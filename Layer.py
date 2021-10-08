import math
import loss
import copy
import activation as act_func
import optimizers as optimizers
import derivatives as der

import numpy as np

class	Layer:
	def		__init__(self, size, activation="sigmoid", l=[]):
		self.g_w = []
		self.g_b = []
		self.data = None
		self.inputs = None
		self.error = None
		self.opti = None
		self.opti2 = None
		self.act = None
		self.weights = None
		self.biases = None

		if (len(l) >= 1):
			self.act = act_func.handler["sigmoid" if activation == None else activation]
			self.weights = np.random.randn(l[-1].size, size) * np.sqrt(2 / l[-1].size)
			self.biases = np.random.randn(size) * np.sqrt(2 / size)
		self.outputs = np.zeros(size)
		self.size = size

	def		calc(self, X):
		self.data = X
		self.inputs = np.dot(X, self.weights) + self.biases
		self.outputs = np.array(self.act(self.inputs))
	
	def		get_results(self):
		return (self.outputs)

	def		get_error(self):
		return (self.error)

	def		get_weights(self):
		return (self.weights)

	def		derivative(self):
		return (der.handler[self.act.__name__](self.inputs.copy()))

	def		calc_error(self, prev_w, error):
		self.error = self.derivative() * np.dot(prev_w, error.T).T

	def		calc_gradients(self, m):
		self.g_w.append(np.dot(self.error.T, self.data) / m)
		self.g_b.append(np.sum(self.error, axis=0) / m)

	def		update(self, alpha, optimizer):
		if (self.opti == None):
			if (optimizer == "adam"):
				self.opti = optimizers.Adam()
				self.opti2 = optimizers.Adam()
			else:
				self.opti = optimizers.Gradient_descent()
				self.opti2 = optimizers.Gradient_descent()
		self.weights -= self.opti.update(np.sum(self.g_w, axis=0).T, alpha)
		self.biases -= self.opti2.update(np.sum(self.g_b, axis=0), alpha)
		self.g_w.clear()
		self.g_b.clear()

	def		debug(self):
		try:
			print("\tactivation: ", self.act.__name__)
		except:
			pass

		print("\tsize: ", self.size)
		
		try:
			print("\tweight size: ", self.weights.T.shape)
			for i in range(self.weights.T.shape[0]):
				print("\t\t", self.weights.T[i])
		except:
			pass

		try:
			print("\tg_w size:", np.array(self.g_w).shape)
			for i in range(np.array(self.g_w).shape[0]):
				print("\t\t", self.g_w[i])
		except:
			pass
		
		try:
			print("\tbias size: ", self.biases.shape, end="\n\t\t")
			for i in range(self.biases.shape[0]):
				print(self.biases[i], end=' ')
				if ((i + 1) % 10 == 0):
					print("\n\t\t", end='')
			print("")
		except:
			pass
			
		try:
			print("\tg_b size:", np.array(self.g_b).shape)
			print("\t\t", self.g_b)
		except:
			pass

		try:
			print("\tdata size: ", self.data.shape)
			for i in range(self.data.shape[0]):
				print("\t\t", self.data[i])
		except:
			pass
	
		try:
			print("\tinput size: ", self.inputs.shape)
			for i in range(self.inputs.shape[0]):
				print("\t\t", self.inputs[i])
		except:
			pass

		try:
			print("\terror size: ", self.error.shape)
			for i in range(self.error.shape[0]):
				print("\t\t", self.error[i])
		except:
			pass

		try:
			print("\toutput size: ", self.outputs.shape, end="\n\t\t")
			for i in range(self.outputs.shape[0]):
				print(self.outputs[i], end=' ')
				if ((i + 1) % 10 == 0):
					print("\n\t\t", end='')
			print("")
		except:
			pass
