from Layer import Layer

import pickle

class	Early_stop():
	def 	__init__(self, monitor, delta, patience, keep_best, min_epochs):
		self.monitor = monitor
		self.delta = delta
		self.patience = patience
		self.keep_best = keep_best
		self.min_epochs = min_epochs
		self.prev = None
		self.layers = None
		self.i = 0
		self.best = 100000

	def		get_best(self):
		return (pickle.loads(self.layers))

	def		test(self, val, layers):
		if (self.min_epochs > 0):
			self.min_epochs -= 1
			return (False)

		if (val < self.best):
			self.best = val
			if (self.keep_best):
				self.layers = pickle.dumps(layers)

		if (self.prev == None):
			self.prev = val
			return (False)
		else:
			delta = self.prev - val
			self.prev = val
			
			if (delta < self.delta):
				self.i += 1
			else:
				self.i = 0

			if (self.i >= self.patience):
				return (True)

		return (False)
