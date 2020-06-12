import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import hpbandster.core.result as hpres

from src.probability import pMOM_loglike, diag_w_Gauss_loglike, depth_categorical_VI
from src.utils import Datafeed, mkdir
from src.datasets import load_flight, load_gap_UCI
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_fc_models import arq_uncert_fc_resnet, arq_uncert_fc_MLP
from src.baselines.training_wrappers import regression_baseline_net, regression_baseline_net_VI
from src.baselines.SGD import SGD_regression_homo
from src.baselines.mfvi import MFVI_regression_homo
from src.baselines.dropout import dropout_regression_homo

uci_names = ['boston', 'concrete', 'energy', 'power', 'wine', 'yacht', 'kin8nm', 'naval', 'protein']
uci_gap_names = ['boston_gap', 'concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                 'kin8nm_gap', 'naval_gap', 'protein_gap']

parser = argparse.ArgumentParser(description='Final training for regression models')
parser.add_argument('--dataset', type=str, help='dataset to train on',
                    choices=["flights"]+uci_names+uci_gap_names)
parser.add_argument('--split', type=str, help='ataset split to train on (default: 0)', default="0")
parser.add_argument('--method', type=str, help='inference method',
                    choices=['DUN_prior', 'DUN_none', 'MFVI', 'SGD', 'Dropout', 'DUN_wd',
                             'DUN_wd-MLP', 'DUN_none-MLP'])
parser.add_argument('--network', type=str, help='network to use for DUNs (default: ResNet)',
                    choices=['MLP', 'ResNet'], default="ResNet")
parser.add_argument('--width', type=int, help='width of the hidden units (default: 100)', default=100)
parser.add_argument('--batch_size', type=int, help='training chunk size (default: 128)', default=128)
parser.add_argument('--valprop', type=float, help='valprop that was used (default: 0.15)', default=0.15)
parser.add_argument('--num', type=str, default="0",
                    help='training run (useful for ensembles and other repeated training) (default: 0)')
parser.add_argument('--gpu', type=int, help='which GPU to run on (default: 0)', default=0)
parser.add_argument('--data_folder', type=str, help='where to find/put the data (default: ../../data/)', default='../../data/')
parser.add_argument('--hpo_results_dir', type=str, help='where to find BOHB results (default: ../hyperparams/results/)',
                    default='../hyperparams/results/')
parser.add_argument('--results_dir', type=str, default='./results/',
                    help='where to put trained models (default: ./results/)')

# cuda = torch.cuda.is_available()
cuda = True


def create_net(method, config, input_dim, output_dim, N_train, network, width, cuda):
    n_layers = config['n_layers']

    if network == "MLP":
        arq_uncert_fc_method = arq_uncert_fc_MLP
    else:
        arq_uncert_fc_method = arq_uncert_fc_resnet

    if method == "DUN_none":
        prior_probs = [1/(n_layers + 1)] * (n_layers + 1)

        model = arq_uncert_fc_method(input_dim, output_dim, width, n_layers, w_prior=None)
        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, N_train, lr=config['lr'],
                     momentum=config['momentum'], cuda=cuda,
                     schedule=None, regression=True)

    elif method == "DUN_prior":
        prior_probs = [1 / (n_layers + 1)] * (n_layers + 1)

        if config['prior'] == 'gauss':
            w_prior = diag_w_Gauss_loglike(μ=0, σ2=config['gauss_σ2'])
        elif config['prior'] == 'pMOM':
            w_prior = pMOM_loglike(r=config['pMOM_r'], τ=1, σ2=config['pMOM_σ2'])
        else:
            raise Exception('We should be using a prior')

        model = arq_uncert_fc_method(input_dim, output_dim, width, n_layers, w_prior=w_prior,
                                     BMA_prior=config['BMA_prior'])
        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                     schedule=None, regression=True)

    elif method == "DUN_wd":
        prior_probs = [1/(n_layers + 1)] * (n_layers + 1)

        model = arq_uncert_fc_method(input_dim, output_dim, width, n_layers, w_prior=None)

        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, N_train, lr=config['lr'],
                     momentum=config['momentum'], cuda=cuda,
                     schedule=None, regression=True, weight_decay=config['weight_decay'])

    elif method == "SGD":
        model = SGD_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers)
        net = regression_baseline_net(model, N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                                      schedule=None, weight_decay=config['weight_decay'])

    elif method == "MFVI":
        model = MFVI_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                     width=width, n_layers=n_layers, prior_sig=config['prior_std'])

        net = regression_baseline_net_VI(model, N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                                         schedule=None, MC_samples=20, train_samples=3)

    elif method == "Dropout":
        model = dropout_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                        width=width, n_layers=n_layers, p_drop=config['p_drop'])
        net = regression_baseline_net(model, N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                                      schedule=None, MC_samples=20, weight_decay=config['weight_decay'])

    else:
        raise Exception("Inference method not implemented")

    return net


def get_dset_split(dataset, split, data_dir, return_means_stds=False):
    if dataset == "flights":
        X_train, X_test, _, _, y_train, y_test, y_means, y_stds = load_flight(base_dir=data_dir,
                                                                              k800=(split == "800k"))

        trainset = Datafeed(X_train, y_train, transform=None)
        testset = Datafeed(X_test, y_test, transform=None)

        N_train = X_train.shape[0]
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

    elif dataset in uci_names + uci_gap_names:
        gap = False
        if dataset in uci_gap_names:
            gap = True
            dataset = dataset[:-4]

        X_train, X_test, _, _, y_train, y_test, y_means, y_stds = \
            load_gap_UCI(base_dir=data_dir, dname=dataset, n_split=int(split), gap=gap)

        trainset = Datafeed(X_train, y_train, transform=None)
        testset = Datafeed(X_test, y_test, transform=None)

        N_train = X_train.shape[0]
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

    else:
        raise Exception("Dataset not implemented yet.")

    if return_means_stds:
        return trainset, testset, N_train, input_dim, output_dim, y_means, y_stds

    return trainset, testset, N_train, input_dim, output_dim


def train_loop(net, trainloader, testloader, epochs, save_path):
    df = pd.DataFrame({"epoch": [], "train_NLL": [], "train_err": [], "test_NLL": [], "test_err": []})
    budget = epochs * 2
    for i in range(epochs):
        print(f"it {i} / {epochs}")
        nb_samples = 0
        MLL_est = 0
        train_err = 0
        train_NLL = 0
        for j, (x, y) in enumerate(trainloader):
            MLL, NLL, err = net.fit(x, y)

            train_NLL += NLL * x.shape[0]
            train_err += err * x.shape[0]
            MLL_est += MLL
            nb_samples += len(x)

            if j < 5 and i == 0:
                nb_samples_test = 0
                test_NLL = 0
                test_err = 0
                for x, y in testloader:
                    NLL, err = net.eval(x, y)

                    test_NLL += NLL * x.shape[0]
                    test_err += err * x.shape[0]
                    nb_samples_test += len(x)

                test_NLL /= nb_samples_test
                test_err /= nb_samples_test

                print(j, test_NLL, test_err)
                if np.isnan(test_NLL) or np.isinf(test_NLL):
                    return True

        train_NLL /= nb_samples
        train_err /= nb_samples

        net.update_lr()

        # eval on test set
        if i + 1 == epochs:
            net.save(f"{save_path}/net_itr_{int(i+1)}.dat")

        nb_samples = 0
        test_NLL = 0
        test_err = 0
        for x, y in testloader:
            NLL, err = net.eval(x, y)

            test_NLL += NLL * x.shape[0]
            test_err += err * x.shape[0]
            nb_samples += len(x)

        test_NLL /= nb_samples
        test_err /= nb_samples

        print(i, test_NLL, test_err)

        df = df.append({"epoch": i, "train_NLL": train_NLL, "train_err": train_err,
                        "test_NLL": test_NLL, "test_err": test_err}, ignore_index=True)
        df.to_csv(f"{save_path}/logs.csv")

    return False


def get_best_configs(results_dir):
    res_folders = Path(results_dir).glob('*/*/*/*/*')

    dfs = []
    for folder in res_folders:
        try:
            pathsplit = folder.parts
            run_id = pathsplit[-1]
            batch_size = pathsplit[-2]
            width = pathsplit[-3]
            method = pathsplit[-4]
            details = pathsplit[-5]
            pathsplit2 = details.split("_")

            if pathsplit2[1] == 'gap':
                pathsplit2.pop(1)
                pathsplit2[0] += '_gap'

            if pathsplit2[0] == 'flights':
                pathsplit2.insert(1, "split")

            dataset = pathsplit2[0]
            split = pathsplit2[2]
            valprop = pathsplit2[4]

            if "MLP" in method:
                network = "MLP"
            else:
                network = "ResNet"

            result = hpres.logged_results_to_HBS_result(folder)
            df = pd.concat(
                result.get_pandas_dataframe(loss_fn=lambda r: {"val_NLL": r.loss, "val_err": r.info['valid err'],
                                                               "best_itr": int(r.info['best iteration'])}),
                # ^ requires hpbandster/core/result.py line 514 to be:
                # all_losses.append(loss_fn(r))
                # pip install git+https://github.com/JamesAllingham/HpBandSter.git@bugfix/result/loss_fn --force 
                axis=1
            )

            df.dropna(subset=['val_NLL'], inplace=True)
            df["dataset"] = dataset
            df["split"] = split
            df["valprop"] = valprop
            df["method"] = method
            df["width"] = width
            df["batch_size"] = batch_size
            df["run_id"] = run_id
            df["network"] = network

            dfs.append(df)
        except Exception as e:
            print(e)
            print(folder)
            print("")

    df = pd.concat(dfs)

    datasets = df["dataset"].drop_duplicates()
    print("Found {0:1} unique datasets: {1}".format(len(datasets), list(datasets)))
    methods = df["method"].drop_duplicates()
    print("Found {0:1} unique methods: {1}".format(len(methods), list(methods)))
    valprops = df["valprop"].drop_duplicates()
    print("Found {0:1} unique valprops: {1}".format(len(valprops), list(valprops)))
    networks = df["network"].drop_duplicates()
    print("Found {0:1} unique networks: {1}".format(len(networks), list(networks)))

    df = (df.sort_values('val_NLL')
            .groupby(["dataset", "split", "method", "network", "valprop", "width", "batch_size"])
            .first()
            .reset_index()
         )

    return df


def main(args):
    dataset = args.dataset
    split = args.split
    method = args.method
    network = args.network
    width = args.width
    batch_size = args.batch_size
    valprop = args.valprop
    num = args.num
    data_dir = args.data_folder
    hpo_results_dir = args.hpo_results_dir
    results_dir = args.results_dir

    if dataset == "flights":
        split = "800k" if split == "1" else "2M"

    df = get_best_configs(hpo_results_dir)
    df = df[(df.dataset == dataset) & (df.split == split) & (df.method == method) &
            (df.network == network) & (df.valprop == str(valprop)) &
            (df.width == str(width)) & (df.batch_size == str(batch_size))]
    if len(df) > 0:
        config = df.to_dict('records')[0]
    else:
        raise RuntimeError("HPO results for chosen config not found.")

    epochs = int(config["best_itr"])

    save_path = f"{results_dir}/{dataset}/{split}/{valprop}/{method}/{network}/{width}/{batch_size}/{num}"
    mkdir(save_path)

    # create data
    trainset, testset, N_train, input_dim, output_dim = get_dset_split(dataset, split, data_dir)

    # create net
    if "MLP" in method:
        method = method[:-4]

    keep_trying = True
    while keep_trying:
        net = create_net(method, config, input_dim, output_dim, N_train, network, width, cuda)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0, pin_memory=cuda)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0, pin_memory=cuda)

        # train net
        keep_trying = train_loop(net, trainloader, testloader, epochs, save_path)


if __name__ == "__main__":
    args = parser.parse_args()

    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    main(args)
