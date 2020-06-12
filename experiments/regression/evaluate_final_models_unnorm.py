import argparse
import time
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from experiments.regression.retrain_best.final_train import get_dset_split, create_net
from experiments.callibration import gauss_callibration, expected_callibration_error
from experiments.regression.retrain_best.final_train import get_best_configs
from src.utils import get_rms, get_gauss_loglike
from src.baselines.training_wrappers import ensemble_predict

parser = argparse.ArgumentParser(description='Unnormalised evaluation of models for regression experiments.')
parser.add_argument('--batch_size', type=int, default=512,
                    help='the batch size to use (default: 512)')
parser.add_argument('--hpo_results_dir', type=str, help='where to find BOHB results (default: ./hyperparams/results/)',
                    default='./hyperparams/results/')
parser.add_argument('--models_dir', type=str, default='./retrain_best/results/',
                    help="where to find the trained models (default: ./retrain_best/results/)")
parser.add_argument('--data_dir', type=str, default='../data/',
                    help="where to find the data (default: ../data/)")


def main(args):
    cuda = False
    data_dir = args.data_dir
    batch_size = args.batch_size

    configs_df = get_best_configs(args.hpo_results_dir)
    print(configs_df)

    df_path = Path("./regression_results.csv")
    if df_path.exists():
        df = pd.read_csv(df_path, dtype={
            "method": str, "epochs": int, "dataset": str, "split": str, "n_samples": int,
            "valprop": float, "network": str, "num": int, "width": int, "batch_size": int,
            "ll": float, "err": float, "ece": float, "tail_ece": float,
            "batch_time": float
        })
    else:
        dtypes = np.dtype([
            ("method", str), ("epochs", int), ("dataset", str), ("split", str), ("n_samples", int),
            ("valprop", float), ("network", str), ("num", int), ("width", int), ("batch_size", int),
            ("ll", float), ("err", float), ("ece", float), ("tail_ece", float),
            ("batch_time", float)
        ])
        data = np.empty(0, dtype=dtypes)
        df = pd.DataFrame(data)

    models_folder = Path(args.models_dir)

    folders = list(models_folder.glob('*/*/*/*/*/*/*/*/*.dat'))
    print(len(folders))
    for model_path in folders:
        parts = model_path.parts

        epochs = int(str(parts[-1]).split("_")[-1][:-4])
        number = parts[-2]
        dataset = parts[-9]
        if number != "0" and dataset != "flights":
            continue

        batch_size = parts[-3]
        width = parts[-4]
        network = parts[-5]
        method = parts[-6]
        valprop = parts[-7]
        split = parts[-8]

        ensemble_paths = list(model_path.parents[1].glob("*/" + parts[-1]))
        max_num = len(ensemble_paths)
        ensemble_numbers = [x for x in [2, 3, 4, 5, 7, 10, 15, 20] if x <= max_num]

        try:
            config = configs_df[(configs_df.dataset == dataset) & (configs_df.split == split) &
                                (configs_df.method == method) & (configs_df.valprop == valprop) &
                                (configs_df.network == network) & (configs_df.width == width) &
                                (configs_df.batch_size == batch_size)].to_dict('records')[0]
        except Exception as e:
            print(e)
            continue

        if "MLP" in method:
            method = method[:-4]

        # make predictions
        n_samples = [1]
        if method in ("MFVI", "Dropout"):
            n_samples = [5, 10, 15, 20, 25, 30]  # TODO: 1

        max_num = 3 if dataset != "flights" else 1
        for num in range(5):
            for n_sample in n_samples:
                kwargs = {"Nsamples": n_sample} if "DUN" not in method else {"get_std": True}
                kwargs["return_model_std"] = False

                row_to_add = {"method": method, "epochs": epochs, "dataset": dataset, "split": split,
                              "n_samples": n_sample, "valprop": float(valprop), "network": network, "width": int(width),
                              "batch_size": int(batch_size), "num": num if dataset != "flights" else number}

                if not len(df.loc[(df[list(row_to_add)] == pd.Series(row_to_add)).all(axis=1)]) > 0:
                    # get correct dataset
                    _, testset, N_train, input_dim, output_dim, y_means, y_stds =\
                        get_dset_split(dataset, split, data_dir, return_means_stds=True)
                    testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size),
                                                             shuffle=False, num_workers=0, pin_memory=cuda)
                    # build appropriate model
                    net = create_net(method, config, input_dim, output_dim, N_train, network, int(width), cuda)
                    # load weights
                    net.load(model_path, to_cpu=True)

                    means, pred_stds, ys, times = [], [], [], []
                    for x, y in testloader:
                        tic = time.time()
                        mean, pred_std = net.predict(x, **kwargs)
                        toc = time.time()

                        means.append(mean)
                        pred_stds.append(pred_std)
                        ys.append(y)
                        times.append(toc - tic)

                    means = torch.cat(means, dim=0)
                    pred_stds = torch.cat(pred_stds, dim=0)
                    ys = torch.cat(ys, dim=0)

                    batch_time = np.mean(times)

                    try:
                        rms = get_rms(means, ys, y_means, y_stds)
                    except Exception as e:
                        print(e)
                        exit(0)

                    ll = get_gauss_loglike(means, pred_stds, ys, y_means, y_stds)
                    bin_probs, _, _, bin_counts, reference =\
                        gauss_callibration(means, pred_stds, ys, 10, cummulative=False, two_sided=False)
                    ece = expected_callibration_error(bin_probs, reference, bin_counts)
                    tail_ece = expected_callibration_error(bin_probs, reference, bin_counts, tail=True)

                    row_to_add.update({"ll": ll, "err": rms.item(), "ece": ece,
                                       "tail_ece": tail_ece, "batch_time": batch_time})
                    df = df.append(row_to_add, ignore_index=True)
                    df.to_csv(df_path, index=False)

            if method == "SGD":
                for i in ensemble_numbers:
                    row_to_add = {"method": "ensemble", "epochs": epochs, "dataset": dataset, "split": split,
                                  "n_samples": i, "valprop": float(valprop), "network": network, "num": num,
                                  "width": int(width), "batch_size": int(batch_size)}
                    if len(df.loc[(df[list(row_to_add)] == pd.Series(row_to_add)).all(axis=1)]) > 0:
                        continue

                    _, testset, N_train, input_dim, output_dim, y_means, y_stds =\
                        get_dset_split(dataset, split, data_dir, return_means_stds=True)
                    testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size),
                                                             shuffle=False, num_workers=0, pin_memory=cuda)
                    # build appropriate model
                    net = create_net(method, config, input_dim, output_dim, N_train, network, int(width), cuda)

                    savefiles = ensemble_paths[:i]

                    means, pred_stds, ys, times = [], [], [], []
                    for x, y in testloader:
                        tic = time.time()
                        mean, pred_std = ensemble_predict(net, savefiles, x, return_model_std=False, to_cpu=True)
                        toc = time.time()

                        means.append(mean)
                        pred_stds.append(pred_std)
                        ys.append(y)
                        times.append(toc - tic)

                    means = torch.cat(means, dim=0)
                    pred_stds = torch.cat(pred_stds, dim=0)
                    ys = torch.cat(ys, dim=0)

                    batch_time = np.mean(times)

                    rms = get_rms(means, ys, y_means, y_stds)
                    ll = get_gauss_loglike(means, pred_stds, ys, y_means, y_stds)
                    bin_probs, _, _, bin_counts, reference =\
                        gauss_callibration(means, pred_stds, ys, 10, cummulative=False, two_sided=False)
                    ece = expected_callibration_error(bin_probs, reference, bin_counts)
                    tail_ece = expected_callibration_error(bin_probs, reference, bin_counts, tail=True)

                    row_to_add.update({"ll": ll, "err": rms.item(), "ece": ece,
                                       "tail_ece": tail_ece, "batch_time": batch_time})
                    df = df.append(row_to_add, ignore_index=True)
                    df.to_csv(df_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
