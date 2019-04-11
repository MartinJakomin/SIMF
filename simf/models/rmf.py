import numpy as np
import time

from simf.models.base import BaseFactorization


class RMF(BaseFactorization):
    """
    Regularized matrix factorization (RMF)
    implementation based on Y. Koren et al.: Matrix factorization techniques for recommender systems (2009)
    """

    def __init__(self, rank, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.P = None
        self.Q = None
        self.b_u = None
        self.b_i = None

    def name(self):
        return "RMF"

    def fit(self, data, verbose=False):
        st = time.time()
        self.init_relations(data)
        self.init_factors_and_biases(data)
        if verbose:
            self.log.info("Initialization complete in %s seconds" % (time.time() - st))
        if verbose:
            self.log.info(
                "Factorizing (SGD) %s relations: bias=%s, max_iter=%s, bias=%s, regularization=%s, learning rate=%s"
                " epsilon=%s" % (len(self.relations), self.bias, self.max_iter, self.bias,
                                 self.regularization, self.learning_rate, self.epsilon))
        self.train_sgd(verbose)
        if verbose:
            self.log.info("Factorization complete in %s seconds" % (time.time() - st))

    def fit_update(self, data, verbose=False):
        if not self.update:
            return
        st = time.time()
        self.init_relations(data)
        self.update_factors(data)
        if verbose:
            self.log.info("Updating (SGD) %s relations: max_iter=%s, bias=%s, regularization=%s, learning rate=%s"
                          " epsilon=%s" % (len(self.relations), self.max_iter, self.bias,
                                           self.regularization, self.learning_rate, self.epsilon))
        self.train_sgd(verbose)
        if verbose:
            self.log.info("Update complete in %s seconds" % (time.time() - st))

    def init_factors_and_biases(self, data):
        R = self.relations[0].get_matrix()
        n, m = R.shape
        self.P = self.construct_factor(R, n, self.rank)
        self.Q = self.construct_factor(R.T, m, self.rank)
        bu, bi = self.construct_bias(R)
        self.b_u = bu.reshape(1, n)[0]
        self.b_i = bi.reshape(1, m)[0]

    def update_factors(self, data):
        R = data[0].get_matrix()
        n, m = R.shape
        self.expand_factors_and_biases(None, n, m)

    def expand_factors_and_biases(self, r, n, m):
        if self.P is not None and n > self.P.shape[0]:
            self.P = self.vstack_factor(self.P, n)
        if self.Q is not None and m > self.Q.shape[0]:
            self.Q = self.vstack_factor(self.Q, m)
        if self.bias:
            self.b_u = self.resize_matrix(self.b_u, n)
            self.b_i = self.resize_matrix(self.b_i, m)

    def train_sgd(self, verbose):
        alpha = self.learning_rate
        beta = self.regularization
        for n in range(self.max_iter):
            cs = self.relations[0].get_matrix().tocoo()
            batches = list(zip(cs.row, cs.col, cs.data))
            np.random.shuffle(batches)
            for i, j, r in batches:
                prediction = self.predict(None, i, j)
                e = (r - prediction)
                # Update biases
                if self.bias:
                    self.b_u[i] += alpha * (e - beta * self.b_u[i])
                    self.b_i[j] += alpha * (e - beta * self.b_i[j])
                # Update user and item latent feature matrices
                self.P[i, :] += alpha * (e * self.Q[j, :] - beta * self.P[i, :])
                self.Q[j, :] += alpha * (e * self.P[i, :] - beta * self.Q[j, :])

    def predict(self, r, i, j):
        prediction = self.P[i].dot(self.Q[j].T)
        if self.bias:
            b = self.data_averages[self.relations[0]]
            prediction += b + self.b_u[i] + self.b_i[j]
        return prediction

    def predict_stream(self, relation=None, s=None, verbose=False):
        relation = self.relations[0]
        maxu = max(list(zip(*s))[0])
        maxi = max(list(zip(*s))[1])
        self.expand_factors_and_biases(relation, maxu + 1, maxi + 1)
        p = np.array([float(self.predict(relation, i, j)) for i, j, _ in s])
        return np.clip(p, *self.data_ranges[relation])
