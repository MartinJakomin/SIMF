import logging
import sys

import numpy as np
import scipy.sparse as sps

from simf.initialization import a_col, random_normal, bias_from_data, bias_zero


class BaseFactorization(object):

    def __init__(self, max_iter=20, epsilon=0, regularization=0.02, learning_rate=0.01, init_method='random', bias=True,
                 precompute_bias=(20, 15), update=True, logger=None):

        self.log = logger
        if not logger:
            self.log = logging.getLogger('default_logger')
            if len(self.log.handlers) < 1:
                self.log.setLevel(logging.DEBUG)
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter("%(asctime)s: %(message)s")
                handler.setFormatter(formatter)
                self.log.addHandler(handler)

        self.init_method = init_method
        self.bias = bias
        self.precompute_bias = precompute_bias
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.update = update
        self.object_types = None
        self.relations = None
        self.data_ranges = None
        self.data_averages = None
        self.factors = None
        self.biases = None

    def __str__(self):
        print(self.__class__.__name__)

    def name(self):
        return "Base"

    def fit(self, data, verbose):
        pass

    def fit_update(self, data, verbose):
        pass

    def predict(self, r, i, j):
        pass

    def predict_stream(self, r, s, verbose):
        pass

    def init_factors_and_biases(self, data):
        pass

    def init_relations(self, data):
        self.object_types = set()
        self.relations = []
        self.data_ranges = {}
        self.data_averages = {}
        for relation in data:
            self.object_types.add(relation.ot1)
            self.object_types.add(relation.ot2)
            self.relations.append(relation)
            R = relation.get_matrix()
            self.data_averages[relation] = float(np.average(R.data))
            self.data_ranges[relation] = (float(min(R.data)), float(max(R.data)))

    def construct_factor(self, M, n, m):
        if self.init_method == "random":
            return random_normal(n, m)
        elif self.init_method == 'a_col':
            return a_col(M, n, m)

    def vstack_factor(self, F, n):
        if self.init_method == 'random':
            return np.vstack([F, random_normal(n - F.shape[0], F.shape[1], loc=0, scale=1. / F.shape[1])])
        elif self.init_method == 'a_col':
            return np.vstack([F, a_col(F, n - F.shape[0], F.shape[1])])

    def construct_bias(self, R):
        if not self.precompute_bias:
            return bias_zero(R)
        return bias_from_data(R, self.precompute_bias[0], self.precompute_bias[1])

    def expand_factors_and_biases(self, r, n, m):
        ot1, ot2 = r.get_object_types()
        if self.factors and n > self.factors[ot1].shape[0]:
            self.factors[ot1] = self.vstack_factor(self.factors[ot1], n)
        if self.bias and n > len(self.biases[r][ot1]):
            self.biases[r][ot1] = self.resize_matrix(self.biases[r][ot1], n)
        if self.factors and m > self.factors[ot2].shape[0]:
            self.factors[ot2] = self.vstack_factor(self.factors[ot2], m)
        if self.bias and m > len(self.biases[r][ot2]):
            self.biases[r][ot2] = self.resize_matrix(self.biases[r][ot2], m)

    def resize_matrix(self, M, shape):
        if isinstance(shape, int):
            B = np.copy(M)
            B.resize(shape)
            return B
        n, m = M.shape
        p, k = shape
        if sps.issparse(M):
            M = sps.coo_matrix(M)
            return sps.csr_matrix((np.append(M.data, 0), (np.append(M.row, p - 1), np.append(M.col, k - 1))),
                                  shape=shape)
        return np.pad(M, [(0, p - n), (0, k - m)], mode='constant', constant_values=0)

    def rmse(self, real, pred):
        if len(pred) < 1 or np.isnan(pred).any():
            return -1
        return np.sqrt(np.average((real - pred) ** 2, axis=0))

    def mae(self, real, pred):
        if len(pred) < 1 or np.isnan(pred).any():
            return -1
        return np.average(np.abs(pred - real), axis=0)

    def get_train_error(self, verbose=False):
        errors = {}
        for rel in self.relations:
            cx = rel.get_matrix().tocoo()
            stream = [(int(i), int(j), float(v)) for i, j, v in zip(cx.row, cx.col, cx.data)]
            values = list(zip(*stream))[2]
            pred = self.predict_stream(rel, stream, verbose=verbose)
            errors[rel] = (self.rmse(values, pred), self.mae(values, pred))
        return errors

    def get_test_error(self, relation, test_set, verbose=False):
        errors = {}
        values = list(zip(*test_set))[2]
        pred = self.predict_stream(relation, test_set, verbose=verbose)
        errors[relation] = (self.rmse(values, pred), self.mae(values, pred))
        return errors
