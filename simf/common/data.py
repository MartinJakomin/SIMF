import numpy as np
import scipy.sparse as sps


def build_sparse_matrix(stream, clazz=sps.csr_matrix):
    rows = np.array(stream[0])
    cols = np.array(stream[1])
    vals = np.array(stream[2])
    return clazz((vals.astype(float), (rows.astype(int), cols.astype(int))))


def split_streams(streams, t=-1):
    length = len(streams)
    if t is -1:
        return streams, [None for _ in range(length)]
    for k, s in enumerate(streams):
        streams[k] = s[:, s[3, :].argsort()]
    train = []
    test = []
    for k, s in enumerate(streams):
        split_point = np.argmax(s[3] > t)
        train.append(s[:, :split_point])
        test.append(s[:, split_point:])
    return train + test


def build_test_set(stream, ts=False):
    rows = [int(x) for x in stream[0]]
    cols = [int(x) for x in stream[1]]
    vals = [float(x) for x in stream[2]]
    if ts:
        return list(zip(rows, cols, vals, [int(x) for x in stream[3]]))
    return list(zip(rows, cols, vals))
