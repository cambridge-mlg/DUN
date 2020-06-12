import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from src.utils import np_get_one_hot, generate_ind_batch, rms

matplotlib.use('Agg')

c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, line_alpha=1, ax=None, lw=1, linestyle='-', fill_linewidths=0.2):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    plt_return = ax.plot(x, y, color=color, lw=lw, linestyle=linestyle, alpha=line_alpha)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, linewidths=fill_linewidths)
    return plt_return


def evaluate_per_depth_classsification(net, X_train, y_train, X_test, y_test, batch_size=None, cuda=True):
    n_layers = net.model.n_layers
    bm = torch.ones((n_layers, n_layers)).tril()
    bin_mat = np.concatenate([np.zeros((1, n_layers)), bm], axis=0)

    if cuda:
        y_train = torch.from_numpy(y_train).cuda().long()
        y_test = torch.from_numpy(y_test).cuda().long()
    else:
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()

    if batch_size is None:
        prob_mtx_pervec = net.vec_predict(X_train, bin_mat)
        prob_mtx_pervec_test = net.vec_predict(X_test, bin_mat)
    else:
        pervecs = []
        for batch in generate_ind_batch(len(X_train), batch_size, False):
            pervecs.append(net.vec_predict(X_train[batch], bin_mat))
        prob_mtx_pervec = torch.cat(pervecs, 1)

        pervecs = []
        for batch in generate_ind_batch(len(X_test), batch_size, False):
            pervecs.append(net.vec_predict(X_test[batch], bin_mat))
        prob_mtx_pervec_test = torch.cat(pervecs, 1)

    prob_mtx_pervec_test = prob_mtx_pervec_test.log()
    prob_mtx_pervec = prob_mtx_pervec.log()

    train_err_vec = []
    train_ce_vec = []
    test_err_vec = []
    test_ce_vec = []

    for d in range(n_layers + 1):
        minus_loglike = F.nll_loss(prob_mtx_pervec[d, :, :], y_train,
                                   reduction='mean').item()
        pred = prob_mtx_pervec[d, :, :].max(dim=1, keepdim=False)[1]  # get the index of the max probability
        err = pred.ne(y_train).sum().item() / y_train.shape[0]
        train_err_vec.append(err)
        train_ce_vec.append(minus_loglike)

    for d in range(n_layers + 1):
        minus_loglike = F.nll_loss(prob_mtx_pervec_test[d, :, :], y_test,
                                   reduction='mean').item()
        pred = prob_mtx_pervec_test[d, :, :].max(dim=1, keepdim=False)[1]  # get the index of the max probability
        err = pred.ne(y_test).sum().item() / y_test.shape[0]
        test_err_vec.append(err)
        test_ce_vec.append(minus_loglike)

    train_err_vec = np.array(train_err_vec)
    train_ce_vec = np.array(train_ce_vec)
    test_err_vec = np.array(test_err_vec)
    test_ce_vec = np.array(test_ce_vec)
    return train_err_vec, train_ce_vec, test_err_vec, test_ce_vec


def evaluate_per_depth_regression(net, X_train, y_train, X_test, y_test, batch_size=None, cuda=True):
    n_layers = net.model.n_layers
    bm = torch.ones((n_layers, n_layers)).tril()
    bin_mat = np.concatenate([np.zeros((1, n_layers)), bm], axis=0)

    fnll = net.f_neg_loglike

    if cuda:
        y_train = torch.from_numpy(y_train).cuda().long()
        y_test = torch.from_numpy(y_test).cuda().long()
    else:
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()

    if batch_size is None:
        prob_mtx_pervec = net.vec_predict(X_train, bin_mat)
        prob_mtx_pervec_test = net.vec_predict(X_test, bin_mat)
    else:
        pervecs = []
        for batch in generate_ind_batch(len(X_train), batch_size, False):
            pervecs.append(net.vec_predict(X_train[batch], bin_mat))
        prob_mtx_pervec = torch.cat(pervecs, 1)

        pervecs = []
        for batch in generate_ind_batch(len(X_test), batch_size, False):
            pervecs.append(net.vec_predict(X_test[batch], bin_mat))
        prob_mtx_pervec_test = torch.cat(pervecs, 1)

    train_err_vec = []
    train_ce_vec = []
    test_err_vec = []
    test_ce_vec = []

    for d in range(n_layers + 1):
        minus_loglike = fnll(prob_mtx_pervec[d, :, :], y_train).mean(dim=0).item()
        err = rms(prob_mtx_pervec[d, :, :], y_train)
        train_err_vec.append(err)
        train_ce_vec.append(minus_loglike)

    for d in range(n_layers + 1):
        minus_loglike = fnll(prob_mtx_pervec_test[d, :, :], y_test).mean(dim=0).item()
        err = rms(prob_mtx_pervec[d, :, :], y_test)
        test_err_vec.append(err)
        test_ce_vec.append(minus_loglike)

    train_err_vec = np.array(train_err_vec)
    train_ce_vec = np.array(train_ce_vec)
    test_err_vec = np.array(test_err_vec)
    test_ce_vec = np.array(test_ce_vec)
    return train_err_vec, train_ce_vec, test_err_vec, test_ce_vec


def plot_predictive_2d_classification(savefile, net, X_train, y_train, extent=5.5, stepdim=350, dpi=200, show=False,
                                      batch_size=None):
    torch.cuda.empty_cache()

    dim_range = np.linspace(-extent, extent, stepdim)
    dimx, dimy = np.meshgrid(dim_range, dim_range)
    dim_mtx = np.concatenate((np.expand_dims(dimx, 2), np.expand_dims(dimy, 2)), axis=2).reshape((stepdim ** 2, 2))

    # VI
    input_mtx = torch.from_numpy(dim_mtx).type(torch.FloatTensor)

    if batch_size is not None:
        total_stack = []
        aux_loader = generate_ind_batch(input_mtx.shape[0], batch_size=batch_size, random=False, roundup=True)
        for idxs in aux_loader:
            probs = net.predict(input_mtx[idxs], depth=None, get_std=False).data
            total_stack.append(probs)
        prob_mtx = torch.cat(total_stack, dim=0)  #
    else:
        prob_mtx = net.predict(input_mtx, depth=None, get_std=False)

    prob_mtx = prob_mtx[:, 1]  # Just keep class 1
    prob_mtx = prob_mtx.view(stepdim, stepdim).data.cpu().numpy()

    # Plotting
    # fig, axes = plt.subplots(1, 2, dpi=180)
    plt.figure(dpi=dpi)
    axes = [plt.gca()]
    axes[0].imshow(prob_mtx, extent=(-extent, extent, -extent, extent), cmap='coolwarm', origin='lower')
    axes[0].set_title('Variational Arq Resnet')
    for i in range(2):
        idxs = (y_train == i)
        axes[0].plot(X_train[idxs, 0], X_train[idxs, 1], '.', c=c[i], alpha=1, markersize=3.7)

    if savefile is not None:
        plt.savefig(savefile + ".png", format='png', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig=None)


def plot_depth_distributions(savefile, net, train_ce_vec=None, test_ce_vec=None, train_err_vec=None, test_err_vec=None,
                             dpi=200, show=False, legend=False):
    n_layers = net.model.n_layers

    plt.figure(dpi=dpi)
    q_probs = net.prob_model.current_posterior.data.cpu().numpy()
    plt.bar(range(n_layers + 1), q_probs, alpha=0.8, label='prob_model posterior')
    plt.bar(range(n_layers + 1), net.prob_model.prior_probs.data.cpu().numpy(), alpha=0.2, label='prior probs')
    plt.grid()
    plt.xlabel('N active blocks (depth)')
    expected_depth = np.sum(q_probs * np.arange(len(q_probs)))
    cuttoff = np.max(q_probs)*0.95
    depth_95th = np.argmax(q_probs > cuttoff)
    plt.title('Expected depth: %f, 95th depth: %d' % (expected_depth, depth_95th))
    ax = plt.gca()
    ax.set_ylabel('depth prob')
    if legend:
        ax.legend()

    if train_ce_vec is not None and test_ce_vec is not None:
        ax2 = ax.twinx()
        p2, = ax2.plot(range(n_layers + 1), -train_ce_vec, 'r-.', label='train loglike')
        ax2.plot(range(n_layers + 1), -test_ce_vec, 'r--.', label='test loglike')
        ax2.set_ylabel('log like')
        ax2.tick_params(axis='y', colors=p2.get_color())
        if legend:
            ax2.legend()

    if train_err_vec is not None and test_err_vec is not None:
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        p3, = ax3.plot(range(n_layers + 1), train_err_vec, 'g-.', label='train err')
        ax3.plot(range(n_layers + 1), test_err_vec, 'g--.', label='test err')
        ax3.set_ylabel('class error')
        ax3.tick_params(axis='y', colors=p3.get_color())
        if legend:
            ax3.legend()

    if savefile is not None:
        plt.savefig(savefile + ".png", format='png', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig=None)


def plot_layer_contributions_2d_classification(savefile, net, X_train, y_train, depths=None, extent=5.5, stepdim=150,
                                               dpi=100, show=None):
    torch.cuda.empty_cache()
    n_layers = net.model.n_layers

    dim_range = np.linspace(-extent, extent, stepdim)
    dimx, dimy = np.meshgrid(dim_range, dim_range)
    dim_mtx = np.concatenate((np.expand_dims(dimx, 2), np.expand_dims(dimy, 2)), axis=2).reshape((stepdim ** 2, 2))

    bm = torch.ones((n_layers, n_layers)).tril()
    bin_mat = np.concatenate([np.zeros((1, n_layers)), bm], axis=0)
    if depths is not None:
        bin_mat = bin_mat[depths, :]

    # VI
    input_mtx = torch.from_numpy(dim_mtx).type(torch.FloatTensor)
    prob_mtx_pervec = net.vec_predict(input_mtx, bin_mat)[:, :, 1]  # only keep probs for one of the classes

    for ii in range(bin_mat.shape[0]):
        prob_mtx = prob_mtx_pervec[ii].view(stepdim, stepdim).cpu().numpy()

        plt.figure(dpi=dpi)
        axes = [plt.gca()]
        axes[0].imshow(prob_mtx, extent=(-extent, extent, -extent, extent), cmap='coolwarm', origin='lower')
        axes[0].set_title('Function for depth: %d' % bin_mat[ii, :].sum())
        for i in range(2):
            idxs = (y_train == i)
            axes[0].plot(X_train[idxs, 0], X_train[idxs, 1], '.', c=c[i], alpha=1, markersize=3.7)

        if savefile is not None:
            plt.savefig(savefile + ('d%d' % (bin_mat[ii, :].sum())) + ".png", format='png', bbox_inches='tight')
        if show:
            plt.show()
        elif not show:
            plt.close(fig=None)
        else:
            pass


def plot_calibration_curve_probs(savefile, probs, y_test, n_bins=10, dpi=200, grid_alph=0.3,
                                 title=None, ax=None, show=False):

    all_preds = probs
    pred_class = np.argmax(all_preds, axis=1)

    expanded_preds = np.reshape(all_preds, -1)
    # These reshapes on the one hot vectors count every possible class as a different prediction
    pred_class_OH_expand = np.reshape(np_get_one_hot(pred_class, probs.shape[1]), -1)
    targets_class_OH_expand = np.reshape(np_get_one_hot(y_test.reshape(-1, 1).astype(int), probs.shape[1]), -1)
    correct_vec = (targets_class_OH_expand * (pred_class_OH_expand == targets_class_OH_expand)).astype(int)

    bin_limits = np.linspace(0, 1, n_bins + 1)

    bin_step = bin_limits[1] - bin_limits[0]
    bin_centers = bin_limits[:-1] + bin_step / 2

    bin_idxs = np.digitize(expanded_preds, bin_limits, right=True) - 1

    bin_counts = np.ones(n_bins)
    bin_corrects = np.zeros(n_bins)
    for nbin in range(n_bins):
        bin_counts[nbin] = np.sum((bin_idxs == nbin).astype(int))
        bin_corrects[nbin] = np.sum(correct_vec[bin_idxs == nbin])

    bin_probs = bin_corrects / bin_counts

    if ax is None:
        plt.figure(dpi=dpi)
        ax = plt.gca()
    bar_ret = ax.bar(bin_centers, bin_probs, 1 / n_bins, edgecolor='k', alpha=0.9)
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), '--', c='k')

    ax.set_xlabel('predicted probability')
    ax.set_ylabel('correct proportion')

    if title is not None:
        ax.set_title(title)
    ax.set_xticks(bin_limits)
    ax.yaxis.grid(alpha=grid_alph)
    ax.xaxis.grid(alpha=grid_alph)
    ax.set_ylim((0, 1))
    if probs.shape[1] == 2:
        ax.set_xlim((0.5, 1))
    else:
        ax.set_xlim((0, 1))
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile + ".png", format='png', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig=None)

    return bar_ret


def plot_calibration_curve(savefile, net, X_test, y_test, n_bins=10, dpi=200, show=False):
    preds = net.sample_predict(X_test, grad=False).sum(dim=0).data.cpu().numpy()

    all_preds = preds
    pred_class = np.argmax(all_preds, axis=1)

    expanded_preds = np.reshape(all_preds, -1)
    # These reshapes on the one hot vectors count every possible class as a different prediction
    pred_class_OH_expand = np.reshape(np_get_one_hot(pred_class, preds.shape[1]), -1)
    targets_class_OH_expand = np.reshape(np_get_one_hot(y_test.reshape(-1, 1).astype(int), preds.shape[1]), -1)
    correct_vec = (targets_class_OH_expand * (pred_class_OH_expand == targets_class_OH_expand)).astype(int)

    #############################################

    bin_limits = np.linspace(0, 1, n_bins + 1)

    bin_step = bin_limits[1] - bin_limits[0]
    bin_centers = bin_limits[:-1] + bin_step / 2

    bin_idxs = np.digitize(expanded_preds, bin_limits, right=True) - 1

    bin_counts = np.ones(n_bins)
    bin_corrects = np.zeros(n_bins)
    for nbin in range(n_bins):
        bin_counts[nbin] = np.sum((bin_idxs == nbin).astype(int))
        bin_corrects[nbin] = np.sum(correct_vec[bin_idxs == nbin])

    bin_probs = bin_corrects / bin_counts

    plt.figure(dpi=dpi)
    ax = plt.gca()
    ax.bar(bin_centers, bin_probs, 1 / n_bins, edgecolor='k')
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), '--', c='k')

    ax.set_xlabel('predicted probability')
    ax.set_ylabel('correct proportion')

    ax.set_title('Calibration Curve')
    ax.set_xticks(bin_limits)
    ax.yaxis.grid(alpha=0.5)
    ax.xaxis.grid(alpha=0.5)
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile + ".png", format='png', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig=None)
