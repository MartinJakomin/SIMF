import time
from functools import reduce

import numpy as np

from simf.initialization.factors import random_normal as random_init
from simf.models.base import BaseFactorization

POSSIBLE_FACTORIZATIONS = ['SGD']


class SIMF(BaseFactorization):

    def __init__(self, factorization_method='SGD', combine_bias=False, combine_initial_factors=True, **kwargs):
        self.factorization_method = factorization_method
        self.combine_bias = combine_bias
        self.combine_initial_factors = combine_initial_factors
        super().__init__(*kwargs)

    def name(self):
        return "SIMF"

    def factorize(self, verbose):
        if self.factorization_method == 'SGD':
            return self.train_sgd(verbose)

    def fit(self, data, verbose=False):
        st = time.time()
        self.init_relations(data)
        self.init_factors_and_biases(data)
        if verbose:
            self.log.info("Initialization complete in %s seconds" % (time.time() - st))
        if self.factorization_method not in POSSIBLE_FACTORIZATIONS:
            raise NotImplemented(
                "Other factorization methods not available. Please choose from: %s" % (str(POSSIBLE_FACTORIZATIONS)))
        if verbose:
            self.log.info(
                "Factorizing (%s) %s relations: bias=%s, max_iter=%s, bias=%s, regularization=%s, learning rate=%s"
                " epsilon=%s" % (self.factorization_method, len(self.relations), self.bias, self.max_iter, self.bias,
                                 self.regularization, self.learning_rate, self.epsilon))
        self.factorize(verbose)
        if verbose:
            self.log.info("Factorization complete in %s seconds" % (time.time() - st))

    def fit_update(self, data, verbose=False):
        if not self.update:
            return
        st = time.time()
        self.update_relations(data)
        self.update_factors(data)
        if self.factorization_method not in POSSIBLE_FACTORIZATIONS:
            raise NotImplemented(
                "Other factorization methods not available. Please choose from: %s" % (str(POSSIBLE_FACTORIZATIONS)))
        if verbose:
            self.log.info("Updating (%s) %s relations: max_iter=%s, bias=%s, regularization=%s, learning rate=%s"
                          " epsilon=%s" % (self.factorization_method, len(self.relations), self.max_iter, self.bias,
                                           self.regularization, self.learning_rate, self.epsilon))
        self.factorize(verbose)
        if verbose:
            self.log.info("Update complete in %s seconds" % (time.time() - st))

    def init_factors_and_biases(self, data):
        self.factors = {}
        self.biases = {}
        # Factor matrices
        for ot in self.object_types:
            initial_factors = []
            for r in data:
                R = r.get_matrix()
                ot1, ot2 = r.get_object_types()
                if ot == ot1:
                    initial_factors.append(self.construct_factor(R, R.shape[0], ot.get_rank()))
                if ot == ot2:
                    initial_factors.append(self.construct_factor(R.T, R.shape[1], ot.get_rank()))
            if self.combine_initial_factors:
                self.factors[ot] = self.combine_factors(initial_factors)
            else:
                self.factors[ot] = initial_factors[0]
        # Middle matrices and biases
        self.biases['combined'] = {}
        for o in self.object_types:
            self.biases['combined'][o] = []
        for r in data:
            ot1, ot2 = r.get_object_types()
            self.factors[(ot1, ot2)] = self.construct_middle_factor(ot1.get_rank(), ot2.get_rank())
            if self.bias:
                bi, bj = self.construct_bias(r.get_matrix())
                self.biases[r] = {}
                self.biases[r][ot1] = bi
                self.biases[r][ot2] = bj
                self.biases['combined'][ot1].append(bi)
                self.biases['combined'][ot2].append(bj)
        if self.combine_bias:
            self.combine_biases()

    def construct_middle_factor(self, n, m):
        return random_init(n, m)

    def combine_factors(self, factors):
        if len(factors) is 1:
            return factors[0]
        # Make all factors of same size
        M = max([f.shape[0] for f in factors])
        r = factors[0].shape[1]
        for k, f in enumerate(factors):
            factors[k] = np.vstack([f, np.zeros((M - f.shape[0], f.shape[1]))])
        # Take the non-zero average
        F = np.zeros((M, r))
        for i in range(M):
            row = np.sum([f[i] for f in factors], axis=0)
            nonzero = np.sum(np.array([np.sum(f[i]) != 0 for f in factors]).astype(int))
            F[i] = row / nonzero
        return F

    def combine_biases(self):
        for o, bl in self.biases['combined'].items():
            max_l = max([len(b) for b in bl])
            # Resize all biases to same length
            cb = [self.resize_matrix(b, max_l) for b in bl]
            cb_nonzero = np.array([b != 0 for b in cb], dtype=int)
            cb = np.sum(cb, axis=0)
            cb_nonzero = np.sum(cb_nonzero, axis=0)
            self.biases['combined'][o] = np.divide(cb, cb_nonzero, where=cb_nonzero != 0)

    def update_relations(self, data):
        self.relations = []
        for relation in data:
            self.relations.append(relation)
            self.data_averages[relation] = float(np.average(relation.get_matrix().data))

    def update_factors(self, data):
        for relation in data:
            n, m = relation.get_matrix().shape
            self.expand_factors_and_biases(relation, n, m)

    def expand_factors_and_biases(self, r, n, m):
        super().expand_factors_and_biases(r, n, m)
        if self.bias and self.combine_bias:
            ot1, ot2 = r.get_object_types()
            if n > len(self.biases['combined'][ot1]):
                self.biases['combined'][ot1] = self.resize_matrix(self.biases['combined'][ot1], n)
            if m > len(self.biases['combined'][ot2]):
                self.biases['combined'][ot2] = self.resize_matrix(self.biases['combined'][ot2], m)

    def predict(self, relation, i, j):
        ot1, ot2 = relation.get_object_types()
        drd, dru = self.data_ranges[relation]
        self.expand_factors_and_biases(relation, i, j)
        prediction = reduce(np.dot, [self.factors[ot1][i], self.factors[(ot1, ot2)], self.factors[ot2][j].T])
        if self.bias:
            prediction += self.data_averages[relation]
            if self.combine_bias:
                relation = 'combined'
            prediction += self.biases[relation][ot1] + self.biases[relation][ot2]
        return np.clip(prediction, drd, dru)

    def predict_stream(self, relation, stream, verbose=False):
        o1, o2 = relation.get_object_types()
        maxu = max(list(zip(*stream))[0])
        maxi = max(list(zip(*stream))[1])
        self.expand_factors_and_biases(relation, maxu + 1, maxi + 1)
        Gi = self.factors[o1]
        Gj = self.factors[o2]
        Sij = self.factors[(o1, o2)]
        if self.bias:
            avg = self.data_averages[relation]
            bu = self.biases[relation][o1]
            bi = self.biases[relation][o2]
            if self.combine_bias:
                bu = self.biases['combined'][o1]
                bi = self.biases['combined'][o2]
            p = np.array([reduce(np.dot, [Gi[i], Sij, Gj[j].T]) + avg + bu[i] + bi[j] for i, j, _ in stream])
        else:
            p = np.array([reduce(np.dot, [Gi[i], Sij, Gj[j].T]) for i, j, _ in stream])
        return np.clip(p, *self.data_ranges[relation])

    def train_sgd(self, verbose):
        beta = self.regularization
        for n in range(self.max_iter):
            # Create batches
            batches = []
            for k, r in enumerate(self.relations):
                R = r.get_matrix().tocoo()
                batches += list(zip([k for _ in range(len(R.row))], R.row, R.col, R.data))
            np.random.shuffle(batches)

            for k, i, j, v in batches:
                rel = self.relations[k]
                oti, otj = rel.get_object_types()
                alpha = self.learning_rate * rel.weight
                p = reduce(np.dot, [self.factors[oti][i], self.factors[(oti, otj)], self.factors[otj][j].T])

                if self.bias:
                    p += self.data_averages[rel]
                    if self.combine_bias:
                        rel = 'combine'
                    p += self.biases[rel][oti][i] + self.biases[rel][otj][j]
                e = float(v - p)

                # Update biases
                if self.bias:
                    self.biases[rel][oti][i] += alpha * (e - beta * self.biases[rel][oti][i])
                    self.biases[rel][otj][j] += alpha * (e - beta * self.biases[rel][otj][j])

                # Update factor matrices
                self.factors[(oti, otj)] += alpha * (e * (self.factors[oti][i, :].T.reshape(
                    self.factors[oti].shape[1], 1)).dot((self.factors[otj][j, :]).reshape(
                    1, self.factors[otj].shape[1])) - beta * self.factors[(oti, otj)])
                self.factors[oti][i, :] += alpha * (e * self.factors[otj][j, :].dot(self.factors[(oti, otj)].T)
                                                    - beta * self.factors[oti][i, :])
                self.factors[otj][j, :] += alpha * (e * self.factors[oti][i, :].dot(self.factors[(oti, otj)])
                                                    - beta * self.factors[otj][j, :])
