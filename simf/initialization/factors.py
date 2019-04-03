import numpy as np


def random_normal(n, m, loc=0.2, scale=0.2):
    return np.random.normal(loc=loc, scale=scale, size=(n, m))


def a_col(M, n, m, p=1):
    W = np.zeros((n, m))
    p_c = int(1. / p * M.shape[1])
    for i in range(m):
        S = M[:, np.random.randint(low=0, high=M.shape[1], size=p_c)]
        SN = (S > 0).astype(int)
        sS = S.sum(axis=1)
        sSN = SN.sum(axis=1)
        B = np.divide(sS, sSN, where=sS != 0)
        W[:, i] = list(B)
    W = W.clip(min=0.00001)
    return W
