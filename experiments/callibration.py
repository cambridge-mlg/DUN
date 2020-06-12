import numpy as np
from scipy.stats import norm
from src.utils import np_get_one_hot


def callibration_bin_probs(target_cdf, n_bins, cummulative=False):
    uniform_bins = np.linspace(0, 1, n_bins + 1)
    uniform_idxs = np.digitize(target_cdf, uniform_bins, right=False)

    bin_counts = np.zeros(n_bins)
    for idx in range(n_bins + 1):
        if idx == n_bins:
            bin_counts[idx-1] += np.sum((uniform_idxs == idx+1).astype(int))
        else:
            bin_counts[idx] = np.sum((uniform_idxs == idx+1).astype(int))

    assert bin_counts.sum() == target_cdf.shape[0]
    bin_prop = bin_counts / target_cdf.shape[0]

    if cummulative:
        bin_prop = np.cumsum(bin_prop)
    return bin_prop, bin_counts, uniform_bins


def gauss_callibration(pred_means, pred_stds, targets, n_bins, cummulative=False, two_sided=False):
    if two_sided:
        norm_pred_err = (pred_means - targets) / pred_stds
        uniform_pred_err = norm.cdf(norm_pred_err)
    else:
        norm_pred_err = np.abs(pred_means - targets) / pred_stds
        uniform_pred_err = norm.cdf(norm_pred_err) * 2 - 1

    bin_prop, bin_counts, uniform_bins = callibration_bin_probs(target_cdf=uniform_pred_err,
                                                                n_bins=n_bins, cummulative=cummulative)

    bin_centers = uniform_bins[1:] - 0.5 / n_bins
    bin_width = 1 / n_bins

    if not cummulative:
        reference = np.ones(len(bin_centers)) * bin_width
    else:
        reference = np.arange(0, 1 + 1 / len(bin_centers), 1 / len(bin_centers))
    # TODO: ensure reference is correct, I think it should just be bin centers. Also return counts
    return bin_prop, bin_centers, bin_width, bin_counts, reference


def cat_callibration(probs, y_test, n_bins, top_k=None):
    all_preds = probs
    pred_class = np.argmax(all_preds, axis=1)

    pred_class_OH = np_get_one_hot(pred_class, probs.shape[1])
    targets_class_OH = np_get_one_hot(y_test.reshape(-1).astype(int), probs.shape[1])

    # indexing with top k is wrong
    if top_k is not None:
        top_k_idx = all_preds.argsort(axis=1)[:, -top_k:]

        all_preds = np.concatenate([all_preds[row_idxs, top_k_idx[row_idxs, :]]
                                    for row_idxs in range(top_k_idx.shape[0])], axis=0)
        targets_class_OH = np.concatenate([targets_class_OH[row_idxs, top_k_idx[row_idxs, :]]
                                           for row_idxs in range(top_k_idx.shape[0])], axis=0)
        pred_class_OH = np.concatenate([pred_class_OH[row_idxs, top_k_idx[row_idxs, :]]
                                        for row_idxs in range(top_k_idx.shape[0])], axis=0)

    expanded_preds = np.reshape(all_preds, -1)
    # These reshapes on the one hot vectors count every possible class as a different prediction
    pred_class_OH_expand = np.reshape(pred_class_OH, -1)
    targets_class_OH_expand = np.reshape(targets_class_OH, -1)
    correct_vec = (targets_class_OH_expand * (pred_class_OH_expand == targets_class_OH_expand)).astype(int)

    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_step = bin_limits[1] - bin_limits[0]
    bin_centers = bin_limits[:-1] + bin_step / 2

    bin_idxs = np.digitize(expanded_preds, bin_limits, right=False) - 1

    bin_counts = np.ones(n_bins)
    bin_corrects = np.zeros(n_bins)
    for nbin in range(n_bins+1):

        if nbin == n_bins:
            bin_counts[nbin-1] += np.sum((bin_idxs == nbin).astype(int))
            bin_corrects[nbin-1] += np.sum(correct_vec[bin_idxs == nbin])
        else:
            bin_counts[nbin] = np.sum((bin_idxs == nbin).astype(int))
            bin_corrects[nbin] = np.sum(correct_vec[bin_idxs == nbin])
    bin_probs = bin_corrects / bin_counts

    bin_probs[bin_counts == 0] = 0

    if top_k is not None:
        assert bin_counts.sum() == probs.shape[0] * top_k
    else:
        assert bin_counts.sum() == probs.shape[0] * probs.shape[1]

    # reference = bin_centers
    reference = np.array([expanded_preds[bin_idxs == nbin].mean() for nbin in range(n_bins)])
    reference[bin_counts == 0] = 0
    # TODO: calculate reference as average accuracy
    return bin_probs, bin_centers, bin_step, bin_counts, reference


def expected_callibration_error(bin_probs, reference, bin_counts, tail=False):
    bin_abs_error = np.abs(bin_probs - reference)
    if tail:
        tail_count = bin_counts[0] + bin_counts[-1]
        ECE = (bin_abs_error[0] * bin_counts[0] + bin_abs_error[-1] * bin_counts[-1]) / tail_count
    else:
        ECE = (bin_abs_error * bin_counts / bin_counts.sum(axis=0)).sum(axis=0)
    assert not np.isnan(ECE)
    return ECE


if __name__ == '__main__':

    print('Quick demonstartion on diabetes dataset')

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import BayesianRidge
    import matplotlib.pyplot as plt

    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

    X_train = ((X_train - x_means) / x_stds).astype(np.float32)
    X_test = ((X_test - x_means) / x_stds).astype(np.float32)

    y_train = ((y_train - y_means) / y_stds).astype(np.float32)
    y_test = ((y_test - y_means) / y_stds).astype(np.float32)

    #####################################################################################################

    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, y_train)
    y_pred, y_pred_std = clf.predict(X_test, return_std=True)

    #####################################################################################################

    n_bins = 10

    bin_prop, bin_centers, bin_width, reference = \
        gauss_callibration(y_pred, y_pred_std, y_test, n_bins=n_bins, cummulative=False, two_sided=False)

    plt.figure(dpi=100)
    ax = plt.gca()

    plt.bar(bin_centers, bin_prop, bin_width, edgecolor='k', alpha=0.3)

    ax.plot(bin_centers, bin_prop, c='r', alpha=0.7, zorder=10)
    ax.scatter(bin_centers, bin_prop, 20, c='r', alpha=0.7, label=None, zorder=10)

    ax.axhline(1 / n_bins, linestyle='--', c='k')
    ax.set_xlim([0, 1])

    # Cumulative version:

    bin_prop, bin_centers, bin_width, reference = \
        gauss_callibration(y_pred, y_pred_std, y_test, n_bins=n_bins, cummulative=True, two_sided=False)

    plt.figure(dpi=100)
    ax = plt.gca()

    plt.bar(bin_centers, bin_prop, bin_width, edgecolor='k', alpha=0.3)

    ax.plot(bin_centers + bin_width / 2, bin_prop, c='r', alpha=0.7, zorder=10)
    ax.scatter(bin_centers + bin_width / 2, bin_prop, 20, c='r', alpha=0.7, label=None, zorder=10)

    ax.plot(reference, reference, linestyle='--', c='k', label='reference')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.show()
