import numpy as np


class Perceptron(object):

	def __init__(self, batch_size, learning_rate,
				 max_iter=200, shuffle=True,
				 seed=None, validation_fraction=0.1,
				 n_iter_no_change=10):
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.shuffle = shuffle
		self.seed = seed
		self.validation_fraction = validation_fraction
		self.n_iter_no_change = n_iter_no_change

	def _fit(self, X, y, X_val=None, y_val=None):
		n_sample, self._n_features = X.shape

		if y.ndim != 1:
			y = y.reshape(-1)
		self._n_outputs = 1
		self._random_state = np.random.RandomState(self.seed)
		if n_sample != y.shape[0]:
			raise ValueError('0 dimension of X and y must be equal. '
							 'got X of %d, y of %d.' % (X.shape[0], y.shape[0]))

		# get train val sets
		shuffled_idx = list(range(n_sample))
		self._random_state.shuffle(shuffled_idx)
		X, y = X[shuffled_idx], y[shuffled_idx]
		y = y * 2 - 1  # map to -1/1

		if X_val is None or y_val is None:
			p_split = int(n_sample * self.validation_fraction)
			X_val, y_val = X[:p_split], y[:p_split]
			X_train, y_train = X[p_split:], y[p_split:]
		else:
			X_train, y_train = X, y

		n_train = X_train.shape[0]

		# init
		self._initialize(n_train)

		#  Gram matrix
		gm = np.dot(X_train, X_train.T)

		self.w_init = (self.alpha_init[:, None] * y_train[:, None] * X_train).sum(axis=0)

		#  prepare batch
		batch_size = min(self.batch_size, n_train)
		batch_num = (n_train // batch_size)
		# print(batch_num)

		for i in range(self.max_iter):
			batch_s = i % batch_num
			batch_e = min(batch_s + batch_size, n_train)
			# X_batch, y_batch = X_train[batch_s:batch_e], y_train[batch_s: batch_e]
			gm_batch, y_batch = gm[batch_s:batch_e], y_train[batch_s: batch_e]

			bs = batch_e - batch_s

			alpha_y = self._alpha * y_train
			#  miss classified sample mask
			d = -y_batch * ((alpha_y[None, :] * gm_batch).sum(axis=1) + self._b)
			M_mask = d > 0
			train_loss = (d * M_mask).sum() / bs
			M_idx = np.array(range(batch_s, batch_e))[M_mask]

			self._alpha[M_idx] += self.learning_rate
			self._b += self.learning_rate * (M_mask * y_batch).sum()

			loss = -y_val * np.dot(X_val, (self._alpha[:, None] * y_train[:, None] * X_train).sum(axis=0)) + self._b
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
			print('Iter:%d, train_loss: %.5f, val loss:%.5f, val error:%.5f' % (i, train_loss, loss, error))

		self.w = (self._alpha[:, None] * y_train[:, None] * X_train).sum(axis=0)
		self.b = self._b

	def _initialize(self, n_train_sample):
		factor = 1
		self.n_iter_ = 0

		init_bound = np.sqrt(factor / (self._n_features + self._n_outputs))
		self._alpha = self._random_state.uniform(-init_bound, init_bound,
												 n_train_sample)

		self._b = self._random_state.uniform(-init_bound, init_bound,
											 self._n_outputs)

		self.loss_curve_ = []
		self._no_improvement_count = 0

		self.best_loss_ = np.inf

		self.alpha_init = self._alpha.copy()
		self.b_init = self._b.copy()

	def fit(self, X, y, X_val=None, y_val=None):
		self._fit(X, y, X_val, y_val)


if __name__ == '__main__':
	from sklearn.datasets import make_classification
	from sklearn.model_selection import train_test_split
	import matplotlib.pyplot as plt

	SEED = 1229
	X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1,
							   n_classes=2, random_state=SEED)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=SEED)

	pp = Perceptron(500, 0.001, max_iter=10000, seed=SEED)
	pp.fit(X_train, y_train)  # , X_test, y_test)

	plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=2, marker='o')

	x1 = np.array([X_test[:, 0].min(), X_test[:, 0].max()])
	x2 = -(pp.w[0] * x1 + pp.b) / pp.w[1]  # trained weight
	x3 = -(pp.w_init[0] * x1 + pp.b_init) / pp.w_init[1]  # initial weight
	plt.plot(x1, x2)
	plt.plot(x1, x3)
	plt.legend(('trained weight', 'random initial weight'), loc='best')
	plt.title('Perceptron dual training')
	plt.savefig('./result_dual.png')
	plt.show()
