import scipy.sparse as sps


class Relation(object):
    def __init__(self, ot1, ot2, matrix, weight=1):
        self.matrix = matrix
        self.ot1 = ot1
        self.ot2 = ot2
        if matrix is not None:
            self.matrix = sps.csr_matrix(matrix)
            if not self.ot1.length or ot1.length < self.matrix.shape[0]:
                self.ot1.length = self.matrix.shape[0]
            if not self.ot2.length or ot2.length < self.matrix.shape[1]:
                self.ot2.length = self.matrix.shape[1]
        self.weight = weight

    def __str__(self):
        return str(self.ot1) + '/' + str(self.ot2)

    def get_matrix(self, copy=False):
        if copy:
            return self.matrix.copy()
        return self.matrix

    def set_matrix(self, matrix):
        self.matrix = sps.csr_matrix(matrix)
        if not self.ot1.length or self.ot1.length < self.matrix.shape[0]:
            self.ot1.length = self.matrix.shape[0]
        if not self.ot2.length or self.ot2.length < self.matrix.shape[1]:
            self.ot2.length = self.matrix.shape[1]

    def set_weight(self, w):
        self.weight = w

    def get_object_types(self, k=0):
        if k is 1:
            return self.ot1
        elif k is 2:
            return self.ot2
        return self.ot1, self.ot2

    def get_shape(self):
        return self.matrix.shape
