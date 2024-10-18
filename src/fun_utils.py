import numpy as np
from pandas import read_csv



def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    x = z[:, 1:]
    return x, y


def split_data(x, y, tr_frac=0.6):
    n_samples = x.shape[0]

    n_tr = int(tr_frac * n_samples)
    n_ts = n_samples - n_tr

    idx = np.linspace(0, n_samples, num=n_samples, endpoint=False, dtype='int')
    np.random.shuffle(idx)
    tr_idx = idx[:n_tr]
    ts_idx = idx[n_tr:]

    assert (n_tr == tr_idx.size)
    assert (n_ts == ts_idx.size)



