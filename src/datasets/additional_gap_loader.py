import zipfile
import pickle
try:
    import urllib
    from urllib import urlretrieve
except Exception:
    import urllib.request as urllib
from os import path

import numpy as np
from numpy.random import uniform, randn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.utils import mkdir


def load_axis(base_dir):

    if not path.exists(base_dir + '/gap_classification'):
        urllib.urlretrieve('https://javierantoran.github.io/assets/datasets/gap_classification.zip',
                           filename=base_dir + '/gap_classification.zip')
        with zipfile.ZipFile(base_dir + '/gap_classification.zip', 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    file1 = base_dir + '/gap_classification/axis.pkl'

    with open(file1, 'rb') as f:
        axis_tupple = pickle.load(f)
        axis_x = axis_tupple[0].astype(np.float32)
        axis_y = axis_tupple[1].astype(np.float32)[:, np.newaxis]

        x_means, x_stds = axis_x.mean(axis=0), axis_x.std(axis=0)
        y_means, y_stds = axis_y.mean(axis=0), axis_y.std(axis=0)

        X = ((axis_x - x_means) / x_stds).astype(np.float32)
        Y = ((axis_y - y_means) / y_stds).astype(np.float32)
    return X, Y


def load_origin(base_dir):

    if not path.exists(base_dir + '/gap_classification'):
        urllib.urlretrieve('https://javierantoran.github.io/assets/datasets/gap_classification.zip',
                           filename=base_dir + '/gap_classification.zip')
        with zipfile.ZipFile(base_dir + '/gap_classification.zip', 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    file2 = base_dir + '/gap_classification/origin.pkl'

    with open(file2, 'rb') as f:
        origin_tupple = pickle.load(f)
        origin_x = origin_tupple[0].astype(np.float32)
        origin_y = origin_tupple[1].astype(np.float32)[:, np.newaxis]

        x_means, x_stds = origin_x.mean(axis=0), origin_x.std(axis=0)
        y_means, y_stds = origin_y.mean(axis=0), origin_y.std(axis=0)

        X = ((origin_x - x_means) / x_stds).astype(np.float32)
        Y = ((origin_y - y_means) / y_stds).astype(np.float32)

    return X, Y


def load_agw_1d(base_dir, get_feats=False):
    if not path.exists(base_dir + '/agw_data'):
        mkdir(base_dir + '/agw_data')
        urllib.urlretrieve('https://raw.githubusercontent.com/wjmaddox/drbayes/master/experiments/synthetic_regression/ckpts/data.npy',
                           filename=base_dir + '/agw_data/data.npy')

    def features(x):
        return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

    data = np.load(base_dir + '/agw_data/data.npy')
    x, y = data[:, 0], data[:, 1]
    y = y[:, None]
    f = features(x)

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)
    f_means, f_stds = f.mean(axis=0), f.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)
    F = ((f - f_means) / f_stds).astype(np.float32)

    if get_feats:
        return F, Y

    return X[:, None], Y


def load_andrew_1d(base_dir):
    if not path.exists(base_dir + '/andrew_1d'):
        print('base_dir does not point to data directory')

    with open(base_dir + '/andrew_1d/1d_cosine_separated.pkl', 'rb') as f:
        data = pickle.load(f)
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)

    return X, Y


def load_matern_1d(base_dir):
    if not path.exists(base_dir + '/matern_data/'):
        mkdir(base_dir + '/matern_data/')

        def gen_1d_matern_data():
            from GPy.kern.src.sde_matern import Matern32
            np.random.seed(4)

            lengthscale = 0.5
            variance = 1.0
            sig_noise = 0.15

            n1_points = 200
            x1 = np.random.uniform(-2, -1, n1_points)[:, None]

            n2_points = 200
            x2 = np.random.uniform(0.5, 2.5, n2_points)[:, None]

            no_points = n1_points + n2_points
            x = np.concatenate([x1, x2], axis=0)
            x.sort(axis=0)

            k = Matern32(input_dim=1, variance=variance, lengthscale=lengthscale)
            C = k.K(x, x) + np.eye(no_points) * sig_noise ** 2

            y = np.random.multivariate_normal(np.zeros((no_points)), C)[:, None]

            x_means, x_stds = x.mean(axis=0), x.std(axis=0)
            y_means, y_stds = y.mean(axis=0), y.std(axis=0)

            X = ((x - x_means) / x_stds).astype(np.float32)
            Y = ((y - y_means) / y_stds).astype(np.float32)

            return X, Y

        x, y = gen_1d_matern_data()
        xy = np.concatenate([x, y], axis=1)
        np.save(base_dir + '/matern_data/matern_1d.npy', xy)
        return x, y
    else:
        xy = np.load(base_dir + '/matern_data/matern_1d.npy')
        x = xy[:, 0]
        x = x[:, None]
        y = xy[:, 1]
        y = y[:, None]
        return x, y


def load_my_1d(base_dir):
    if not path.exists(base_dir + '/my_1d_data/'):
        mkdir(base_dir + '/my_1d_data/')

        def gen_my_1d(hetero=False):

            np.random.seed(0)
            Npoints = 1002
            x0 = uniform(-1, 0, size=int(Npoints / 3))
            x1 = uniform(1.7, 2.5, size=int(Npoints / 3))
            x2 = uniform(4, 5, size=int(Npoints / 3))
            x = np.concatenate([x0, x1, x2])

            def function(x):
                return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

            y = function(x)

            homo_noise_std = 0.25
            homo_noise = randn(*x.shape) * homo_noise_std
            y_homo = y + homo_noise

            hetero_noise_std = np.abs(0.1 * np.abs(x) ** 1.5)
            hetero_noise = randn(*x.shape) * hetero_noise_std
            y_hetero = y + hetero_noise

            X = x[:, np.newaxis]
            y_joint = np.stack([y_homo, y_hetero], axis=1)

            X_train, X_test, y_joint_train, y_joint_test = train_test_split(X, y_joint, test_size=0.5, random_state=42)
            y_hetero_train, y_hetero_test = y_joint_train[:, 1, np.newaxis], y_joint_test[:, 1, np.newaxis]
            y_homo_train, y_homo_test = y_joint_train[:, 0, np.newaxis], y_joint_test[:, 0, np.newaxis]

            x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
            y_hetero_means, y_hetero_stds = y_hetero_train.mean(axis=0), y_hetero_train.std(axis=0)
            y_homo_means, y_homo_stds = y_homo_test.mean(axis=0), y_homo_test.std(axis=0)

            X_train = ((X_train - x_means) / x_stds).astype(np.float32)
            X_test = ((X_test - x_means) / x_stds).astype(np.float32)

            y_hetero_train = ((y_hetero_train - y_hetero_means) / y_hetero_stds).astype(np.float32)
            y_hetero_test = ((y_hetero_test - y_hetero_means) / y_hetero_stds).astype(np.float32)

            y_homo_train = ((y_homo_train - y_homo_means) / y_homo_stds).astype(np.float32)
            y_homo_test = ((y_homo_test - y_homo_means) / y_homo_stds).astype(np.float32)

            if hetero:
                return X_train, y_hetero_train, X_test, y_hetero_test
            else:
                return X_train, y_homo_train, X_test, y_homo_test

        X_train, y_homo_train, X_test, y_homo_test = gen_my_1d()
        xy = np.concatenate([X_train, y_homo_train, X_test, y_homo_test], axis=1)
        np.save(base_dir + '/my_1d_data/my_1d_data.npy', xy)
        return X_train, y_homo_train, X_test, y_homo_test

    xy = np.load(base_dir + '/my_1d_data/my_1d_data.npy')
    X_train = xy[:, 0, None].astype(np.float32)
    y_homo_train = xy[:, 1, None].astype(np.float32)
    X_test = xy[:, 2, None].astype(np.float32)
    y_homo_test = xy[:, 3, None].astype(np.float32)

    return X_train, y_homo_train, X_test, y_homo_test


def load_wiggle():

    np.random.seed(0)
    Npoints = 300
    x = randn(Npoints) * 2.5 + 5  # uniform(0, 10, size=Npoints)

    def function(x):
        return np.sin(np.pi * x) + 0.2 * np.cos(np.pi * x * 4) - 0.3 * x

    y = function(x)

    homo_noise_std = 0.25
    homo_noise = randn(*x.shape) * homo_noise_std
    y = y + homo_noise

    x = x[:, None]
    y = y[:, None]

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)

    return X, Y

