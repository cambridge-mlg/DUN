from os import path
import zipfile
try:
    import urllib
    from urllib import urlretrieve
except Exception:
    import urllib.request as urllib

import numpy as np


def load_gap_UCI(base_dir, dname, n_split=0, gap=True):

    if not path.exists(base_dir + '/UCI_for_sharing'):
        urllib.urlretrieve('https://javierantoran.github.io/assets/datasets/UCI_for_sharing.zip',
                           filename=base_dir + '/UCI_for_sharing.zip')
        with zipfile.ZipFile(base_dir + '/UCI_for_sharing.zip', 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    np.random.seed(1234)
    dir_load = base_dir + '/UCI_for_sharing/standard/' + dname + '/data/'

    if gap:
        dir_idx = base_dir + '/UCI_for_sharing/gap/' + dname + '/data/'
    else:
        dir_idx = base_dir + '/UCI_for_sharing/standard/' + dname + '/data/'

    data = np.loadtxt(dir_load + 'data.txt')
    feature_idx = np.loadtxt(dir_load + 'index_features.txt').astype(int)
    target_idx = np.loadtxt(dir_load + 'index_target.txt').astype(int)

    test_idx_list = []
    train_idx_list = []

    for i in range(20):
        try:
            test_idx_list.append(np.loadtxt(dir_idx + 'index_test_%d.txt' % i).astype(int))
            train_idx_list.append(np.loadtxt(dir_idx + 'index_train_%d.txt' % i).astype(int))
        except Exception:
            pass

    data_train = data[train_idx_list[n_split], :]
    data_test = data[test_idx_list[n_split], :]

    X_train = data_train[:, feature_idx].astype(np.float32)
    X_test = data_test[:, feature_idx].astype(np.float32)
    y_train = data_train[:, target_idx].astype(np.float32)
    y_test = data_test[:, target_idx].astype(np.float32)

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

    x_stds[x_stds < 1e-10] = 1.

    X_train = ((X_train - x_means) / x_stds)
    y_train = ((y_train - y_means) / y_stds)[:, np.newaxis]
    X_test = ((X_test - x_means) / x_stds)
    y_test = ((y_test - y_means) / y_stds)[:, np.newaxis]

    return X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds
