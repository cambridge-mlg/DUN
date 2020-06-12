import os
import argparse
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import Datafeed, generate_ind_batch
from src.datasets.additional_gap_loader import load_my_1d, load_agw_1d, load_andrew_1d
from src.datasets.additional_gap_loader import load_matern_1d, load_axis, load_origin, load_wiggle
from src.probability import depth_categorical_VI
from src.DUN.train_fc import train_fc_DUN
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_fc_models import arq_uncert_fc_resnet, arq_uncert_fc_MLP
from src.baselines.SGD import SGD_regression_homo
from src.baselines.dropout import dropout_regression_homo
from src.baselines.mfvi import MFVI_regression_homo
from src.baselines.training_wrappers import regression_baseline_net, regression_baseline_net_VI
from src.baselines.train_fc import train_fc_baseline

matplotlib.use('Agg')

tic = time()

c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

parser = argparse.ArgumentParser(description='Toy dataset running script')

parser.add_argument('--n_epochs', type=int, default=4000,
                    help='number of iterations performed by the optimizer (default: 4000)')
parser.add_argument('--dataset', help='toggles which dataset to optimize for (default: my_1d)',
                    choices=['my_1d', 'matern_1d', 'agw_1d', 'andrew_1d', 'axis', 'origin', 'wiggle'], default='my_1d')
parser.add_argument('--datadir', type=str, help='where to save dataset (default: ../data/)',
                    default='../data/')
parser.add_argument('--gpu', type=int, help='which GPU to run on (default: None)', default=None)
parser.add_argument('--inference', type=str, help='model to use (default: DUN)',
                    default='DUN', choices=['DUN', 'MFVI', 'Dropout', 'SGD'])
parser.add_argument('--num_workers', type=int, help='number of parallel workers for dataloading (default: 1)', default=1)
parser.add_argument('--N_layers', type=int, help='number of hidden layers to use (default: 2)', default=2)
parser.add_argument('--width', type=int, help='number of hidden units to use (default: 50)', default=50)
parser.add_argument('--savedir', type=str, help='where to save results (default: ./saves/)',
                    default='./saves/')
parser.add_argument('--overcount', type=int, help='how many times to count data towards ELBO (default: 1)', default=1)
parser.add_argument('--lr', type=float, help='learning rate (default: 1e-3)', default=1e-3)
parser.add_argument('--wd', type=float, help='weight_decay, (default: 0)', default=0)
parser.add_argument('--network', type=str,
                    help='model type when using DUNs (other methods use ResNet) (default: ResNet)',
                    default='ResNet', choices=['ResNet', 'MLP'])

args = parser.parse_args()


# Some defaults
batch_size = 2048
momentum = 0.9
epochs = args.n_epochs
nb_its_dev = 5

cuda = (args.gpu is not None)
if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print('cuda', cuda)


# Choosing dataset

if args.dataset == 'my_1d':
    X_train, y_train, X_test, y_test = load_my_1d(args.datadir)
elif args.dataset == 'matern_1d':
    X_train, y_train = load_matern_1d(args.datadir)
elif args.dataset == 'agw_1d':
    X_train, y_train = load_agw_1d(args.datadir, get_feats=False)
elif args.dataset == 'andrew_1d':
    X_train, y_train = load_andrew_1d(args.datadir)
elif args.dataset == 'axis':
    X_train, y_train = load_axis(args.datadir)
elif args.dataset == 'origin':
    X_train, y_train = load_origin(args.datadir)
elif args.dataset == 'wiggle':
    X_train, y_train = load_wiggle()

trainset = Datafeed(torch.Tensor(X_train), torch.Tensor(y_train), transform=None)
if args.dataset == 'my_1d':
    valset = Datafeed(torch.Tensor(X_test), torch.Tensor(y_test), transform=None)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
else:
    valset = Datafeed(torch.Tensor(X_train), torch.Tensor(y_train), transform=None)
    print(X_train.shape, y_train.shape)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                          num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=args.num_workers)

# Preparing method

N_train = X_train.shape[0] * args.overcount
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

width = args.width
n_layers = args.N_layers
wd = args.wd

if args.inference == 'MFVI':
    prior_sig = 1

    model = MFVI_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                 width=width, n_layers=n_layers, prior_sig=1)

    net = regression_baseline_net_VI(model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=None,
                                     MC_samples=10, train_samples=5)

elif args.inference == 'Dropout':
    model = dropout_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers, p_drop=0.1)

    net = regression_baseline_net(model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=None,
                                  MC_samples=10, weight_decay=wd)
elif args.inference == 'SGD':
    model = SGD_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                width=width, n_layers=n_layers)

    net = regression_baseline_net(model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=None,
                                  MC_samples=0, weight_decay=wd)
elif args.inference == 'DUN':

    if args.network == 'ResNet':
        model = arq_uncert_fc_resnet(input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False)
    elif args.network == 'MLP':
        model = arq_uncert_fc_MLP(input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False)
    else:
        raise Exception('Bad network type. This should never raise as there is a previous assert.')

    prior_probs = [1 / (n_layers + 1)] * (n_layers + 1)
    prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
    net = DUN_VI(model, prob_model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None,
                 regression=True, pred_sig=None, weight_decay=wd)

name = '_'.join([args.inference, args.dataset, str(args.N_layers), str(args.width), str(args.lr), str(args.wd),
                 str(args.overcount)])

if args.network == 'MLP':
    name += '_MLP'

# Training
if args.inference in ['MFVI', 'Dropout', 'SGD']:
    marginal_loglike_estimate, train_mean_predictive_loglike, dev_mean_predictive_loglike, err_train, err_dev, \
        approx_d_posterior, true_d_posterior, true_likelihood, exact_ELBO, basedir = \
        train_fc_baseline(net, name, args.savedir, batch_size, epochs, trainloader, valloader, cuda=cuda,
                       flat_ims=False, nb_its_dev=nb_its_dev, early_stop=None,
                       track_posterior=False, track_exact_ELBO=False, seed=0, save_freq=nb_its_dev)
else:
    marginal_loglike_estimate, train_mean_predictive_loglike, dev_mean_predictive_loglike, err_train, err_dev, \
        approx_d_posterior, true_d_posterior, true_likelihood, exact_ELBO, basedir = \
        train_fc_DUN(net, name, args.savedir, batch_size, epochs, trainloader, valloader,
                  cuda, seed=0, flat_ims=False, nb_its_dev=nb_its_dev, early_stop=None,
                  track_posterior=False, track_exact_ELBO=False, tags=None,
                  load_path=None, save_freq=nb_its_dev)

# Plot and save results

media_dir = basedir + '/media'

# training progress plots

textsize = 8
marker = 1
fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].semilogy(range(0, epochs, nb_its_dev), err_dev[::nb_its_dev], 'b-')
ax[0].semilogy(err_train[:epochs], 'r--')
ax[0].set_ylim(top=1, bottom=1e-2)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('% error')
ax[0].grid(b=True, which='major', color='k', linestyle='-')
ax[0].grid(b=True, which='minor', color='k', linestyle='--')
lgd = ax[0].legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] +
             ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    item.set_fontsize(textsize)
    item.set_weight('normal')

ax[1].plot(range(0, epochs, nb_its_dev), dev_mean_predictive_loglike[::nb_its_dev], 'b-')
ax[1].plot(train_mean_predictive_loglike[:epochs], 'r--')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('mean loglike')
ax[1].grid(b=True, which='major', color='k', linestyle='-')
ax[1].grid(b=True, which='minor', color='k', linestyle='--')
lgd = ax[1].legend(['test', 'train'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] +
             ax[1].get_xticklabels() + ax[1].get_yticklabels()):
    item.set_fontsize(textsize)
    item.set_weight('normal')
plt.tight_layout()
plt.savefig(os.path.join(media_dir, 'training.png'), format='png', bbox_inches='tight')


models_dir = basedir + '/models'
net.load(os.path.join(models_dir, 'theta_best.dat'))

# 1d plot
show_range = 5
ylim = 3
add_noise = False

if args.dataset in ['my_1d', 'matern_1d', 'agw_1d', 'andrew_1d', 'wiggle']:

    x_view = np.linspace(-show_range, show_range, 8000)
    subsample = 1
    x_view = torch.Tensor(x_view).unsqueeze(1)

    if args.inference != 'DUN':
        pred_mu, pred_std = net.predict(x_view, Nsamples=50, return_model_std=True)
    else:
        pred_mu, pred_std = net.predict(x_view, get_std=True, return_model_std=True)

    pred_mu = pred_mu.data.cpu().numpy()
    try:
        pred_std = pred_std.data.cpu().numpy()
    except Exception:
        pred_std = 0

    plt.figure(dpi=200)
    plt.scatter(X_train[::subsample], y_train[::subsample], s=5, alpha=0.5, c=c[0])
    plt.plot(x_view, pred_mu, c=c[3])

    if add_noise:
        noise_std = net.f_neg_loglike.log_std.exp().data.cpu().numpy()
    else:
        noise_std = 0

    plt.fill_between(x_view[:, 0],
                     pred_mu[:, 0] + (pred_std[:, 0] ** 2 + noise_std ** 2) ** 0.5,
                     pred_mu[:, 0] - (pred_std[:, 0] ** 2 + noise_std ** 2) ** 0.5, color=c[3], alpha=0.3)

    plt.ylim([-ylim, ylim])
    plt.xlim([-show_range, show_range])
    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, 'prediction_1d.png'), format='png', bbox_inches='tight')

    if args.inference == 'DUN':
        layer_preds = net.layer_predict(x_view).data.cpu().numpy()
        for i in range(layer_preds.shape[0]):
            plt.figure(dpi=80)
            plt.scatter(X_train[::subsample], y_train[::subsample], s=5, alpha=0.5, c=c[0])
            _ = plt.plot(x_view[:, 0], layer_preds[i, :, 0].T, alpha=0.8, c='r')
            plt.title(i)
            plt.ylim([-ylim, ylim])
            plt.xlim([-show_range, show_range])
            plt.tight_layout()
            plt.savefig(os.path.join(media_dir, str(i) + '_layerwise.png'), format='png', bbox_inches='tight')

# 2d dataset plots
if args.dataset in ['axis', 'origin']:

    if args.inference != 'SGD':

        extent = 2.5
        stepdim = 200
        dpi = 200
        batch_size = 4096

        dim_range = np.linspace(-extent, extent, stepdim)
        dimx, dimy = np.meshgrid(dim_range, dim_range)
        dim_mtx = np.concatenate((np.expand_dims(dimx, 2), np.expand_dims(dimy, 2)), axis=2).reshape((stepdim ** 2, 2))

        input_mtx = torch.from_numpy(dim_mtx).type(torch.FloatTensor)

        if batch_size is not None:
            total_stack = []
            mean_stack = []
            layer_preds_stack = []
            aux_loader = generate_ind_batch(input_mtx.shape[0], batch_size=batch_size, random=False, roundup=True)
            for idxs in aux_loader:
                print(idxs)
                if args.inference != 'DUN':
                    mean, stds = net.predict(input_mtx[idxs], Nsamples=50, return_model_std=True)
                else:
                    mean, stds = net.predict(input_mtx[idxs], depth=None, get_std=True, return_model_std=True)
                mean_stack.append(mean)
                total_stack.append(stds)
            prob_mtx = torch.cat(total_stack, dim=0)  # append in dim 1 because 0 are probs
            mean = torch.cat(mean_stack, dim=0)
        else:
            if args.inference != 'DUN':
                mean, prob_mtx = net.predict(input_mtx, Nsamples=100, return_model_std=True)
            else:
                mean, prob_mtx = net.predict(input_mtx, depth=None, get_std=True, return_model_std=True)

        prob_mtx = prob_mtx.view(stepdim, stepdim).data.cpu().numpy()

        prob_mtx = np.log(prob_mtx.clip(max=3))

        plt.figure(dpi=dpi)
        axes = [plt.gca()]
        im = axes[0].imshow(prob_mtx, extent=(-extent, extent, -extent, extent), cmap='coolwarm', origin='lower')

        axes[0].plot(X_train[:, 0], X_train[:, 1], '.', c=c[6], alpha=1, markersize=1.5)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('log std')
        plt.tight_layout()
        plt.savefig(os.path.join(media_dir, '2d.png'), format='png', bbox_inches='tight')

    # Now for 1d slices

    if args.dataset == 'origin':

        ylim = 4
        show_range = 4
        stepdim = 500

        add_noise = False

        x0 = np.linspace(-show_range, show_range, stepdim)
        x1 = np.linspace(-show_range, show_range, stepdim)
        x_in = np.stack([x0, x1], axis=1)
        input_mtx = torch.from_numpy(x_in).type(torch.FloatTensor)

        if args.inference != 'DUN':
            pred_mu, pred_std = net.predict(input_mtx, Nsamples=500, return_model_std=True)
        else:
            pred_mu, pred_std = net.predict(input_mtx, depth=None, get_std=True, return_model_std=True)

        pred_mu = pred_mu.data.cpu().numpy()
        try:
            pred_std = pred_std.data.cpu().numpy()
        except Exception:
            pred_std = 0

        projected_data = (X_train[:, 0] + X_train[:, 1]) / 2

        x_view = (x0 + x1) * 0.5
        x_view = x_view[:, None]

        plt.figure(dpi=120)
        plt.scatter(projected_data, y_train, s=5, alpha=0.5, c=c[0])

        plt.plot(x_view, pred_mu, c=c[3])

        if add_noise:
            noise_std = net.f_neg_loglike.log_std.exp().data.cpu().numpy()
        else:
            noise_std = 0

        plt.fill_between(x_view[:, 0],
                         pred_mu[:, 0] + (pred_std[:, 0] ** 2 + noise_std ** 2) ** 0.5,
                         pred_mu[:, 0] - (pred_std[:, 0] ** 2 + noise_std ** 2) ** 0.5, color=c[3], alpha=0.1)

        plt.ylim([-ylim, ylim])
        plt.xlim([-show_range, show_range])
        plt.tight_layout()
        plt.savefig(os.path.join(media_dir, 'prediction_1d.png'), format='png', bbox_inches='tight')

        if args.inference == 'DUN':
            layer_preds = net.layer_predict(input_mtx).data.cpu().numpy()
            for i in range(layer_preds.shape[0]):
                plt.figure(dpi=60)
                plt.scatter(projected_data, y_train, s=5, alpha=0.5, c=c[0])
                _ = plt.plot(x_view[:, 0], layer_preds[i, :, 0].T, alpha=0.8, c='r')
                plt.title(i)
                plt.ylim([-ylim, ylim])
                plt.xlim([-show_range, show_range])
                plt.tight_layout()
                plt.savefig(os.path.join(media_dir, str(i) + '_layerwise.png'), format='png', bbox_inches='tight')

    if args.dataset == 'axis':
        ylim = 4
        show_range = 4
        stepdim = 500

        add_noise = False

        x0 = np.zeros(stepdim)
        x1 = np.linspace(-show_range, show_range, stepdim)
        x_in = np.stack([x0, x1], axis=1)
        input_mtx = torch.from_numpy(x_in).type(torch.FloatTensor)

        if args.inference != 'DUN':
            pred_mu, pred_std = net.predict(input_mtx, Nsamples=500, return_model_std=True)
        else:
            pred_mu, pred_std = net.predict(input_mtx, depth=None, get_std=True, return_model_std=True)

        pred_mu = pred_mu.data.cpu().numpy()
        try:
            pred_std = pred_std.data.cpu().numpy()
        except Exception:
            pred_std = 0

        projected_data = X_train[:, 1]

        x_view = x1
        x_view = x_view[:, None]

        plt.figure(dpi=120)
        plt.scatter(projected_data, y_train, s=5, alpha=0.5, c=c[0])

        plt.plot(x_view, pred_mu, c=c[3])

        if add_noise:
            noise_std = net.f_neg_loglike.log_std.exp().data.cpu().numpy()
        else:
            noise_std = 0

        plt.fill_between(x_view[:, 0],
                         pred_mu[:, 0] + (pred_std[:, 0] ** 2 + noise_std ** 2) ** 0.5,
                         pred_mu[:, 0] - (pred_std[:, 0] ** 2 + noise_std ** 2) ** 0.5, color=c[3], alpha=0.1)

        plt.ylim([-ylim, ylim])
        plt.xlim([-show_range, show_range])
        plt.tight_layout()
        plt.savefig(os.path.join(media_dir, 'prediction_1d.png'), format='png', bbox_inches='tight')

        if args.inference == 'DUN':
            layer_preds = net.layer_predict(input_mtx).data.cpu().numpy()
            for i in range(layer_preds.shape[0]):
                plt.figure(dpi=60)
                plt.scatter(projected_data, y_train, s=5, alpha=0.5, c=c[0])
                _ = plt.plot(x_view[:, 0], layer_preds[i, :, 0].T, alpha=0.8, c='r')
                plt.title(i)
                plt.ylim([-ylim, ylim])
                plt.xlim([-show_range, show_range])
                plt.tight_layout()
                plt.savefig(os.path.join(media_dir, str(i) + '_layerwise.png'), format='png', bbox_inches='tight')


toc = time()
print(toc - tic)
