import numpy as np
from time import time
import pandas as pd

class SoftmaxRegression(object):
	def __init__(self, eta=0.01, epochs=50, l2=0.0, minibatches=1, n_classes=None, random_seed=None):
		self.eta = eta
		self.epochs = epochs
		self.l2 = l2
		self.minibatches = minibatches
		self.n_classes = n_classes
		self.random_seed = random_seed

	def _net_input(self, X, W, b):
		return (X.dot(W)+b)

	def _softmax(self, z):
		return (np.exp(z.T)/np.sum(np.exp(z), axis=1)).T

	def _cross_entropy(self, output, y_target):
		return -np.sum(np.log(output)*(y_target), axis=1)

	def _cost(self, cross_entropy):
		L2_term = self.l2 * np.sum(self.w_ **2)
		cross_entropy = cross_entropy + L2_term
		return 0.5 * np.mean(cross_entropy)

	def _to_classlabels(self, z):
		return z.argmax(axis=1)

	def _init_params(self, weights_shape, bias_shape=(1, ), dtype='float64',
					scale=0.01, random_seed=None):
		# initialize weight coefficients
		if random_seed:
			np.random.seed(random_seed)
		w=np.random.normal(loc=0.0, scale=scale, size=weights_shape)
		b=np.zeros(shape=bias_shape)
		return b.astype(dtype), w.astype(dtype)

	def _one_hot(self, y, n_labels, dtype):
		# returns a matrix where each sample in y is represented as a row, 
		# and each column represents the class label in the one-hot encoding scheme
		mat = np.zeros((len(y), n_labels))
		for i, val in enumerate(y):
			mat[i, val]=1
		return mat.astype(dtype)

	def _yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
		indices = np.arange(data_ary.shape[0])
		if shuffle:
			indices = np.arange(data_ary.shape[0])
		if n_batches > 1:
			remainder = data_ary.shape[0] % n_batches
			if remainder:
				minis = np.array_split(indices[:-remainder], n_batches)
				minis[-1] = np.concatenate((minis[-1], indices[-remainder:]), axis=0)
			else:
				minis = np.array_split(indices, n_batches)
		else:
			minis = (indices, )
		for idx_batch in minis:
			yield idx_batch

	def _shuffle_arrays(self, arrays):
		# shuffle arrays in unison
		r=np.random.permutation(len(arrays[0]))
		return [ary[r] for ary in arrays]


	def _fit(self, X, y, init_params=True):
		if init_params:
			if self.n_classes is None:
				self.n_classes = np.max(y)+1
			self._n_features = X.shape[1]

			self.b_, self.w_ = self._init_params(weights_shape=(self._n_features, self.n_classes),
												bias_shape=(self.n_classes,),
												random_seed=self.random_seed)
			self.cost_ = []

		y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)
		for i in range(self.epochs):
			for idx in self._yield_minibatches_idx(n_batches=self.minibatches,
													data_ary=y,
													shuffle=True):
				net = self._net_input(X[idx], self.w_, self.b_)
				softm = self._softmax(net)
				diff = softm - y_enc[idx]
				mse = np.mean(diff, axis=0)

				# gradient -> n_features times n_classes
				grad = np.dot(X[idx].T, diff)

				# update in opp. direction of the cost gradient
				self.w_ -= (self.eta * grad + self.eta * self.l2 * self.w_)
				self.b_ -= (self.eta * np.sum(diff, axis=0))

			# compute cost of the whole epoch
			net = self._net_input(X, self.w_, self.b_)
			softm = self._softmax(net)
			cross_ent = self._cross_entropy(output=softm, y_target=y_enc)
			cost = self._cost(cross_ent)
			self.cost_.append(cost)
		return self


	def fit(self, X, y, init_params=True):
		if self.random_seed != None:
			np.random.seed(self.random_seed)
		self._fit(X=X, y=y, init_params=init_params)
		self._is_fitted=True
		return self

	def _predict(self, X):
		probas = self.predict_proba(X)
		return self._to_classlabels(probas)

	def predict(self, X):
		if not self._is_fitted:
			raise AttributeError('Model is not fitted yet')
		return self._predict(X)

	def predict_proba(self, X):
		net = self._net_input(X, self.w_, self.b_)
		softm = self._softmax(net)
		return softm


def main():
	from mlxtend.data import iris_data
	from mlxtend.plotting import plot_decision_regions
	import matplotlib.pyplot as plt

	# Loading Data

	X, y = iris_data()
	X = X[:, [0, 3]] # sepal length and petal width

	# standardize
	X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

	lr = SoftmaxRegression(eta=0.01, epochs=10, minibatches=1, random_seed=0)
	lr.fit(X, y)

	plot_decision_regions(X, y, clf=lr)
	plt.title('Softmax Regression - Gradient Descent')
	plt.show()

	plt.plot(range(len(lr.cost_)), lr.cost_)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()

if __name__ == "__main__":
	main()