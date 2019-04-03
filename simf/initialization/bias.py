import numpy as np
import scipy.sparse as sps


def bias_zero(R):
    n, m = R.shape
    return np.zeros((1, n)).reshape(1, n)[0], np.zeros((1, m)).reshape(1, m)[0]


def bias_from_data(R, alpha, beta):
    n, m = R.shape
    avg = R.sum() / float(R.nnz)
    W = sps.csr_matrix(R > 0, dtype=int)
    # Calculate bi
    nnzi = R.getnnz(axis=0)
    bi = ((R.transpose() * sps.csr_matrix(np.ones((n, 1)))).toarray().transpose() - (
            nnzi * avg).reshape(1, m)) / (nnzi + alpha).reshape(1, m)
    # Calculate bu
    RB = R - W.multiply(bi[0] + avg)
    bu = (RB * sps.csr_matrix(np.ones((m, 1)))).toarray() / (R.getnnz(axis=1) + beta).reshape(n, 1)
    return bu.reshape(1, n)[0], bi.reshape(1, m)[0]
