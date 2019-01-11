import numpy as np


class Preceptron(object):

	def __init__(self, batch_size, learning_rate,
				 max_iter=200, shuffle=True,
				 seed=None, tol=1e-4,
				 verbose=False, momentum=0.9,
				 nesterovs_momentum=True, early_stopping=False,
				 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
				 epsilon=1e-8, n_iter_no_change=10):
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.shuffle = shuffle
		self.seed = seed
		self.tol = tol
		self.verbose = verbose
		self.momentum = momentum
		self.nesterovs_momentum = nesterovs_momentum
		self.early_stopping = early_stopping
		self.validation_fraction = validation_fraction
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.n_iter_no_change = n_iter_no_change

	def _fit(self, X, y):
		n_sample, self._n_features = X.shape

		if y.ndim != 1:
			y = y.reshape(-1)
		self._n_outputs = 1
		self._random_state = np.random.RandomState(self.seed)
		if n_sample != y.shape[0]:
			raise ValueError('0 dimension of X and y must be equal. '
							 'got X of %d, y of %d.' % (X.shape[0], y.shape[0]))

		self._initialize()
		shuffled_idx = list(range(n_sample))
		self._random_state.shuffle(shuffled_idx)
		X, y = X[shuffled_idx], y[shuffled_idx]
		y = y * 2 - 1  # map to -1/1
		p_split = int(n_sample * self.validation_fraction)
		X_val, y_val = X[:p_split], y[:p_split]
		X_train, y_train = X[p_split:], y[p_split:]

		n_train = X_train.shape[0]

		batch_size = min(self.batch_size, n_train)
		batch_num = (n_train // batch_size)
		print(batch_num)
		for i in range(self.max_iter):
			batch_s = i % batch_num
			batch_e = min(batch_s + batch_size, n_train)
			X_batch, y_batch = X_train[batch_s:batch_e], y_train[batch_s: batch_e]

			bs = batch_e - batch_s

			#  miss classified sample
			d = (-y_batch * (np.dot(X_batch, self._w) + self._b))
			M_mask = d > 0


			# grad_w = - 0.5 * (((M_mask * y_batch)[:, None] * X_batch) / np.sqrt(np.abs(d))[:, None]).sum(axis=0) / bs
			# grad_b = - 0.5 * ((M_mask * y_batch) / np.sqrt(np.abs(d))).sum() / bs

			grad_w = - ((M_mask * y_batch)[:, None] * X_batch).sum(axis=0) / bs
			grad_b = - (M_mask * y_batch).sum() / bs

			self._w -= self.learning_rate * grad_w
			self._b -= self.learning_rate * grad_b

			loss = (-y_val * (np.dot(X_val, self._w) + self._b))
			M_mask_val = loss > 0
			loss = (loss * M_mask_val).sum() / bs
			self.loss_curve_.append(loss)
			if loss < self.best_loss_:
				self.best_loss_ = loss
			else:
				self._no_improvement_count += 1

			if self._no_improvement_count > self.n_iter_no_change:
				break
			error = M_mask_val.sum() / y_val.shape[0]
			print('Iter:%d, val loss:%.5f, val error:%.5f' % (i, loss, error))

			self.w = self._w
			self.b = self._b

	def _initialize(self):
		factor = 1
		self.n_iter_ = 0

		init_bound = np.sqrt(factor / (self._n_features + self._n_outputs))
		self._w = self._random_state.uniform(-init_bound, init_bound,
											 self._n_features)

		self._b = self._random_state.uniform(-init_bound, init_bound,
											 self._n_outputs)

		self.loss_curve_ = []
		self._no_improvement_count = 0
		if self.early_stopping:
			self.validation_scores_ = []
			self.best_validation_score_ = -np.inf
		else:
			self.best_loss_ = np.inf

		self.w_init = self._w.copy()
		self.b_init = self._b.copy()

	def fit(self, X, y):
		self._fit(X, y)

if __name__ == '__main__':
	from sklearn.datasets import make_classification
	import matplotlib.pyplot as plt

	X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, n_classes=2)

	pp = Preceptron(500, 0.001, max_iter=10000)




	pp.fit(X,y)

	plt.scatter(X[:, 0], X[:,1], c=y, s=2, marker='o')
	x1 = np.array([X[:, 0].min(), X[:, 0].max()])
	x2 = -(pp.w[0] * x1 + pp.b) / pp.w[1]
	plt.plot(x1, x2)

	# plt.scatter(X[:, 0], X[:, 1], c=y, s=2, marker='o')
	# x1 = np.array([X[:, 0].min(), X[:, 0].max()])
	# x2 = -(pp.w_init[0] * x1 + pp.b_init) / pp.w[1]
	# plt.plot(x1, x2)

	plt.show()