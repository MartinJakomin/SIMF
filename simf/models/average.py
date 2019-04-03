import time

import numpy as np

from simf.models.base import BaseFactorization


class Average(BaseFactorization):
    """
    Average model
    """

    def __init__(self, **kwargs):
        super().__init__(*kwargs)

    def name(self):
        return "Average"

    def fit(self, data, verbose=False):
        if verbose:
            self.log.info("Fitting the average model: bias=%s, update=%s" % (self.bias, self.update))
        st = time.time()
        self.init_relations(data)
        self.init_factors_and_biases(data)
        if verbose:
            self.log.info("Fit complete in %s seconds" % (time.time() - st))
        return

    def fit_update(self, data, verbose=False):
        if not self.update:
            return
        self.fit(data, verbose)

    def init_factors_and_biases(self, data):
        self.biases = {}
        for r in data:
            ot1, ot2 = r.get_object_types()
            bu, bi = self.construct_bias(r.get_matrix())
            self.biases[r] = {}
            self.biases[r][ot1] = bu
            self.biases[r][ot2] = bi

    def predict(self, relation, i, j):
        ot1, ot2 = relation.get_object_types()
        prediction = self.data_averages[relation]
        if self.bias:
            prediction += self.biases[relation][ot1] + self.biases[relation][ot2]
        return prediction

    def predict_stream(self, relation, s, verbose=False):
        o1, o2 = relation.get_object_types()
        avg = self.data_averages[relation]
        maxu = max(list(zip(*s))[0])
        maxi = max(list(zip(*s))[1])
        self.expand_factors_and_biases(relation, maxu + 1, maxi + 1)
        if self.bias:
            bu = self.biases[relation][o1]
            bi = self.biases[relation][o2]
            return np.array([avg + bu[i] + bi[j] for i, j, _ in s]).clip(*self.data_ranges[relation])
        return np.array([avg for _ in s]).clip(*self.data_ranges[relation])
