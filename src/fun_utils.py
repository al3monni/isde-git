import numpy as np
from pandas import read_csv



def load_data(filename):
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    x = z[:, 1:]
    return x, y


def split_data(x, y, tr_fraction=0.5):
    n_samples = x.shape[0]

    n_tr = int(tr_frac * n_samples)
    n_ts = n_samples - n_tr

    idx = np.linspace(0, n_samples, num=n_samples, endpoint=False, dtype='int')
    np.random.shuffle(idx)
    tr_idx = idx[:n_tr]
    ts_idx = idx[n_tr:]

    assert (n_tr == tr_idx.size)
    assert (n_ts == ts_idx.size)

    xtr = x[tr_idx, :]
    ytr = y[tr_idx]
    xts = x[ts_idx, :]
    yts = y[ts_idx]
    return xtr, ytr, xts, yts
