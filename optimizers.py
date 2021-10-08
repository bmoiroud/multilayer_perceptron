import numpy as np

class	Adam:
	def		__init__(self):
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 1e-8
		self.t = 0
		self.m = 0
		self.v = 0

	def		update(self, g, alpha):
		self.t += 1
		self.m = self.beta1 * self.m + (1 - self.beta1) * g
		self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
		a = alpha * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
		return (a * self.m / (np.sqrt(self.v) + self.epsilon))

class	Gradient_descent:

	def		update(self, g, alpha):
		return (alpha * g)
