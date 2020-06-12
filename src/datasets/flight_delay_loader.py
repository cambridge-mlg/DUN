from os import path
import bz2
import random
import zipfile
try:
    import urllib
    from urllib import urlretrieve
except Exception:
    import urllib.request as urllib

import numpy as np
import pandas as pd

from src.utils import mkdir


def load_official_flight(base_dir, k800=False):
    if not path.exists(base_dir + '/flight'):
        mkdir(base_dir + '/flight')

    if not path.isfile(base_dir + '/flight/filtered_data.pickle'):
        urllib.urlretrieve('https://javierantoran.github.io/assets/datasets/filtered_flight_data.pickle.zip',
                           filename=base_dir + '/flight/filtered_flight_data.pickle.zip')

        with zipfile.ZipFile(base_dir + '/flight/filtered_flight_data.pickle.zip', 'r') as zip_ref:
            zip_ref.extractall(base_dir + '/flight/')

    file1 = base_dir + '/flight/filtered_data.pickle'
    filtered = pd.read_pickle(file1)

    inputs = filtered[['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime',
                       'AirTime', 'Distance', 'plane_age']].values

    outputs = filtered[['ArrDelay']].values

    if k800 is False:
        X_train = inputs[:-100000].astype(np.float32)
        y_train = outputs[:-100000].astype(np.float32)
        X_test = inputs[-100000:].astype(np.float32)
        y_test = outputs[-100000:].astype(np.float32)
    else:
        X_train = inputs[:700000].astype(np.float32)
        y_train = outputs[:700000].astype(np.float32)
        X_test = inputs[700000:800000].astype(np.float32)
        y_test = outputs[700000:800000].astype(np.float32)

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

    x_stds[x_stds < 1e-10] = 1.

    X_train = ((X_train - x_means) / x_stds)
    y_train = ((y_train - y_means) / y_stds)
    X_test = ((X_test - x_means) / x_stds)
    y_test = ((y_test - y_means) / y_stds)

    return X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds


load_flight = load_official_flight

