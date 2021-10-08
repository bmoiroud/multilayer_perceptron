import json
import loss as l
import numpy as np
import derivatives as der
import matplotlib.pyplot as plt

from Layer import Layer
from Early_stop import Early_stop
from preprocessing import shuffle_data

def		evaluate(X, Y):
	max_x = np.argmax(X, axis=1)
	max_y = np.argmax(Y, axis=1)
	res = [1 if max_x[i]== max_y[i] else 0 for i in range(len(X))]
	return (sum(res) / len(X) * 100)

class	Model:
	def		__init__(self, input_shape=0):
		self.output = None
		self.layers = []
		self.size = 0
		self.stop = None
		self.layers.append(Layer(abs(input_shape), activation=None))

	def		dense(self, shape, activation="sigmoid"):
		self.layers.append(Layer(abs(shape), activation=activation, l=self.layers))
	
	def		backpropagation(self, Y, loss):
		self.layers[-1].error = l.error(self.output, Y)
		for i in reversed(range(1, self.size - 1)):
			self.layers[i].calc_error(self.layers[i + 1].get_weights(), self.layers[i + 1].get_error())
		m = len(Y)
		for i in range(1, self.size):
			self.layers[i].calc_gradients(m)

	def		update_layers(self, alpha, optimizer):
		for i in range(1, self.size):
			self.layers[i].update(alpha, optimizer)

	def		predict(self, X):
		if (X.shape[1] != self.layers[0].size):
			print("Wrong data shape expected: {0} got: {1}".format(self.layers[0].size, X.shape[1]))
			return (None)
		self.layers[0].outputs = X
		for i in range(1, self.size):
			self.layers[i].calc(self.layers[i - 1].get_results())
		self.output = self.layers[-1].get_results()

	def		early_stop(self, monitor="val_loss", delta=1e-5, patience=2, keep_best=False, min_epochs=0):
		if (monitor not in {"loss", "val_loss"}):
			monitor = "val_loss"
		if (delta <= 0):
			delta =1e-5
		if (patience < 0):
			patience = 2
		self.stop = Early_stop(monitor, delta, patience, keep_best, min_epochs)

	def		train(self, X, Y, val_X=None, val_Y=None, loss='binary_crossentropy', optimizer="gradient_descent", learning_rate=0.001, epochs=100, batch_size=32, plot=False, shuffle=False):
		pred = []
		val_pred = []
		val_cost = None
		cost_hist = []
		val_cost_hist = []

		self.size = len(self.layers)

		if (X.shape[1] != self.layers[0].size):
			print("Wrong data shape expected: {0} got: {1}".format(self.layers[0].size, X.shape[1]))
			return (None)

		for e in range(epochs):
			i = 0
			pred.clear()
			val_pred.clear()
			if (shuffle == True):
				X, Y = shuffle_data(X, Y)

			while (i < len(X)):
				if (isinstance(val_X, np.ndarray) and isinstance(val_Y, np.ndarray) and i < len(val_X)):
					self.predict(val_X[i : i + batch_size])
					val_pred += self.output.tolist()

				self.predict(X[i : i + batch_size])
				pred += self.output.tolist()
				self.backpropagation(Y[i : i + batch_size], loss)
				i += batch_size

			self.update_layers(learning_rate, optimizer)
			cost = l.handler[loss](np.array(pred).reshape(Y.shape), Y)

			if (isinstance(val_X, np.ndarray) and isinstance(val_Y, np.ndarray)):
				val_cost = l.handler[loss](np.array(val_pred).reshape(val_Y.shape), val_Y)
				val_acc = evaluate(val_pred, val_Y)

			if (plot == True):
				cost_hist.append(cost)
				if (isinstance(val_cost, float)):
					val_cost_hist.append(val_cost)

			if (isinstance(val_cost, float)):
				s = str("\tval_loss: {0:.6f}".format(val_cost))
				s2 = str("\tval_acc: {0:.3f}%".format(val_acc))
			else:
				s = s2 = ''
			prec = evaluate(pred, Y)
			print("epoch: {0}/{1}\tloss: {2:.6f}\t{3}\tacc: {4:.3f}%\t{5}".format(e + 1, epochs, cost, s, prec, s2))

			if (self.stop != None):
				if (self.stop.monitor == "val_loss" and not isinstance(val_cost, float)):
					self.stop.monitor = "loss"
				val = cost if self.stop.monitor == "loss" else None
				val = val_cost if self.stop.monitor == "val_loss" else val
				if (self.stop.test(val, self.layers)):
					if (self.stop.keep_best == True):
						self.layers = self.stop.get_best()
					break
		
		if (plot == True):
			x = np.linspace(0, e + 1, e + 1)
			plt.plot(x, cost_hist)
			if (isinstance(val_X, np.ndarray) and isinstance(val_Y, np.ndarray)):
				plt.plot(x, val_cost_hist)
				plt.legend(("loss", "val_loss"), loc='upper right')
			else:
				plt.legend("loss", loc='upper right')
			plt.xlabel("epochs")
			plt.ylabel("loss")
			plt.show()

	def		save(self, filename):
		data = {}
		
		self.size = len(self.layers)
		for i in range(self.size):
			s = "layer" + str(i)
			data[s] = []
			data[s].append({
				"size": self.layers[i].size,
				"activation": self.layers[i].act.__name__ if self.layers[i].act is not None else None,
				"weights": self.layers[i].weights.tolist() if self.layers[i].weights is not None else None,
				"biases": self.layers[i].biases.tolist() if self.layers[i].biases is not None else None
			})
		with open(filename, "w") as file:
			json.dump(data, file)

	def		load(self, filename):
		self.layers.clear()

		with open(filename, "r") as file:
			data = json.load(file)

		self.size = len(data)
		self.layers.append(Layer(data["layer0"][0]["size"], activation=None))
		for i in range(1, len(data)):
			s = "layer" + str(i)
			self.layers.append(Layer(data[s][0]["size"], activation=data[s][0]["activation"], l=self.layers))
			self.layers[i].weights = np.array(data[s][0]["weights"])
			self.layers[i].biases = np.array(data[s][0]["biases"])

	def		debug(self):
		print("self.layers size: ", len(self.layers))
		for i in range(len(self.layers)):
			print("\nLayer ", str(i))
			self.layers[i].debug()
