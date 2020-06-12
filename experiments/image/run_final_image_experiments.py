import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.image.test_methods import baseline_test_stats, ensemble_test_stats, DUN_test_stats
from experiments.image.test_methods import baseline_OOD_AUC_ROC, ensemble_OOD_AUC_ROC, DUN_OOD_AUC_ROC
from experiments.image.test_methods import baseline_batch_time, ensemble_batch_time, DUN_batch_time
from experiments.image.test_methods import baseline_class_rej, ensemble_class_rej, DUN_class_rej
from src.probability import depth_categorical_VI
from src.datasets.image_loaders import get_image_loader
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_img_resnets import resnet50

parser = argparse.ArgumentParser(description='DeScRiPtIoN')
parser.add_argument('--batch_size', type=int, default=256,
                    help='the batch size to use (default: 256)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use (if None, all GPUs will be used) (default: None)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of dataloader workers to use (default: 4)')
parser.add_argument('--method', type=str, choices=['SGD', 'ensemble', 'dropout', 'DUN'],
                    help='method to run')
parser.add_argument('--dataset', type=str, choices=['MNIST', 'Fashion', 'CIFAR10', 'CIFAR100', 'SVHN'],
                    help="dataset to use")
parser.add_argument('--model', type=str, choices=['resnet50'], default='resnet50',
                    help="model to use (default: resnet50)")
parser.add_argument('--models_dir', type=str, default='./results/',
                    help="where to find the models (default: ./results/)")
parser.add_argument('--data_dir', type=str, default='../data/',
                    help="where to find the data (default: ../data/)")

args = parser.parse_args()

method = args.method
dataset = args.dataset
model = args.model
data_dir = args.data_dir
workers = args.workers


# check whether we already have a pandas dataframe with results, if so load
df_path = Path(f"./results/image_results_{dataset}_{model}_{method}.csv")
if df_path.exists():
    df = pd.read_csv(df_path)
    if "warmup" not in df:
        df["warmup"] = 0
    if "use_no_train_post" not in df:
        df["use_no_train_post"] = False
else:
    dtypes = np.dtype([
        ("method", str), ("dataset", str), ("model", str), ("wd", bool), ("p_drop", float), ("start", int),
        ("stop", int), ("number", int), ("n_samples", int), ("warmup", int),
        ("ll", float), ("err", float), ("ece", float), ("brier", float), ("rotation", int), ("corruption", int),
        ("auc_roc", float), ("err_props", list), ("target_dataset", str),
        ("batch_time", float), ("batch_size", int),
        ("best_or_last", str), ("use_no_train_post", bool)
    ])
    data = np.empty(0, dtype=dtypes)
    df = pd.DataFrame(data)

# add to pandas dataframe by looking at models folder
# for each row to add in pandas dataframe, run a bunch of experiments

models_folder = Path(args.models_dir).expanduser()

target_datasets = {
    "MNIST": ["Fashion"],
    "Fashion": ["MNIST", "KMNIST"],
    "CIFAR10": ["SVHN"],
    "CIFAR100": ["SVHN"],
    "SVHN": ["CIFAR10"]
}

n_samples_ranges = {
    "SGD": [1],
    "ensemble": [2, 3, 5, 7, 10, 15, 20],
    "dropout": [1, 2, 3, 5, 7, 10, 15, 20],
    "DUN": [1]
}

corruptions = {
    "CIFAR10": [(0, cor) for cor in range(1, 6)],
    "CIFAR100": [(0, cor) for cor in range(1, 6)],
    "MNIST": [],
    "Fashion": [],
    "SVHN": []
}

rotations = {
    "CIFAR10": [],
    "CIFAR100": [],
    "MNIST": [(rot, 0) for rot in range(15, 181, 15)],
    "Fashion": [],
    "SVHN": []
}

num_classeses = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "Fashion": 10,
    "SVHN": 10
}

input_chanelses = {
    "CIFAR10": 3,
    "CIFAR100": 3,
    "MNIST": 1,
    "Fashion": 1,
    "SVHN": 3
}

if method == "SGD" or method == "dropout":
    test_stats_method = baseline_test_stats
    OOD_AUC_ROC_method = baseline_OOD_AUC_ROC
    batch_time_method = baseline_batch_time
    class_rejection_method = baseline_class_rej
elif method == "ensemble":
    test_stats_method = ensemble_test_stats
    OOD_AUC_ROC_method = ensemble_OOD_AUC_ROC
    batch_time_method = ensemble_batch_time
    class_rejection_method = ensemble_class_rej
else:
    test_stats_method = DUN_test_stats
    OOD_AUC_ROC_method = DUN_OOD_AUC_ROC
    batch_time_method = DUN_batch_time
    class_rejection_method = DUN_class_rej

if model == "resnet50":
    model_class = resnet50
else:
    raise Warning(f"Model class not implemented: {model}")

folders = list(models_folder.glob('*'))
print(len(folders))
method_filt = method if method != "ensemble" else "SGD"
folders = [folder for folder in folders if folder.match("_".join([dataset, model, method_filt, "*"]))]
print(len(folders))

row_to_add_proto = {"dataset": dataset, "method": method, "model": model}

if method == "ensemble":
    # create dataframe of ensembles only
    ensembles_dtypes = np.dtype([
        ("wd", bool), ("number", int)
    ])
    ensembles_data = np.empty(0, dtype=ensembles_dtypes)
    ensembles_df = pd.DataFrame(ensembles_data)

    for folder in folders:
        folder_split = str(folder).split("/")[-1].split("_")
        if folder_split[2] != "SGD" or folder_split[1] != model or folder_split[0] != dataset:
            continue

        wd = folder_split[3] == "wd"
        num = int(folder_split[4])

        ensembles_df = ensembles_df.append({"wd": wd, "number": num}, ignore_index=True)

    ensembles_df = ensembles_df.sort_values('number')

no_train_posteriors = {}

# run experiments
print(len(folders))
for folder in folders:
    folder_split = str(folder).split("/")[-1].split("_")
    num = int(folder_split[-1])

    if method == "ensemble" and num != 0:
        continue

    wd = folder_split[-2] == "wd"
    p_drop = 0 if method != "dropout" else float(folder_split[-3][1:])
    start, stop = (0, 0) if method != "DUN" else map(int, folder_split[-3].split("-"))
    if method == "DUN" and len(folder_split) == 7:
        row_to_add_proto["warmup"] = int(folder_split[-4][-1])
    else:
        row_to_add_proto["warmup"] = 0

    row_to_add_proto["number"] = num
    # ^ This will be wrong for ensembles
    row_to_add_proto["wd"] = wd
    row_to_add_proto["p_drop"] = float(p_drop)

    use_d_post_options = [False] if method != "DUN" else [True, False]
    for use_no_train_post in use_d_post_options:
        row_to_add_proto["use_no_train_post"] = use_no_train_post

        for best_or_last in ["best", "last"]:
            row_to_add_proto["best_or_last"] = best_or_last
            if method != "DUN":
                model_obj = model_class(
                    arch_uncert=False, num_classes=num_classeses[dataset], zero_init_residual=True,
                    initial_conv='1x3' if dataset != "Imagenet" else '3x3', concat_pool=False,
                    input_chanels=input_chanelses[dataset], p_drop=p_drop,
                )
            else:
                n_layers = stop - start
                row_to_add_proto["start"] = start
                row_to_add_proto["stop"] = stop

                prior_probs = [1 / (n_layers)] * (n_layers)
                prob_model = depth_categorical_VI(prior_probs, cuda=True)

                model_base = model_class(arch_uncert=True, start_depth=start, end_depth=stop,
                                         num_classes=num_classeses[dataset], zero_init_residual=True,
                                         initial_conv='1x3' if dataset != "Imagenet" else '3x3', concat_pool=False,
                                         input_chanels=input_chanelses[dataset], p_drop=0)

                N_train = 0

                model_obj = DUN_VI(model_base, prob_model, N_train, lr=0.1, momentum=0.9, weight_decay=1e-4, cuda=True,
                                   schedule=None, regression=False, pred_sig=None)

            for n_samples in n_samples_ranges[method]:

                if method != "ensemble":
                    num_repeats = 1
                else:
                    filt_ensembles_df = ensembles_df[(ensembles_df.wd == wd)].reset_index()
                    n_ensemble_models = len(filt_ensembles_df)
                    num_repeats = math.floor(n_ensemble_models/n_samples)

                for num_repeat in range(num_repeats):
                    if method == "ensemble":
                        row_to_add_proto["number"] = num_repeat

                    model_path = ("checkpoint.pth.tar" if best_or_last == "last" else "model_best.pth.tar")
                    if method != "ensemble":
                        savefile = folder / model_path
                    else:
                        model_indices = range(num_repeat*n_samples, (num_repeat + 1)*n_samples)
                        savefile = [folder.parent / "_".join(folder_split[:-1] +
                                                             [str(filt_ensembles_df["number"][int(model_idx)])])
                                    / model_path for model_idx in model_indices]

                    row_to_add_proto["n_samples"] = n_samples

                    if method == "DUN" and savefile not in no_train_posteriors.keys() and use_no_train_post:
                        _, train_loader, _, _, _, Ntrain = \
                            get_image_loader(dataset, batch_size=args.batch_size, cuda=True, workers=workers,
                                             data_dir=data_dir, distributed=False)

                        model_obj.load(savefile)
                        model_obj.N_train = Ntrain
                        notrain_post, _ = model_obj.get_exact_d_posterior(train_loader, train_bn=False,
                                                                          logposterior=False)
                        no_train_posteriors[savefile] = notrain_post.data

                    kwargs = {}
                    if method == "DUN" and use_no_train_post:
                        kwargs = {"d_posterior": no_train_posteriors[savefile]}

                    # all measurements of err, ll, ece, brier
                    for rotation, corruption in [(0, 0)] + corruptions[dataset] + rotations[dataset]:
                        row_to_add = row_to_add_proto.copy()
                        row_to_add.update({"rotation": rotation, "corruption": corruption})

                        if len(df.loc[(df[list(row_to_add)] == pd.Series(row_to_add)).all(axis=1)]) > 0:
                            continue

                        rotation = None if rotation == 0 else rotation
                        corruption = None if corruption == 0 else corruption
                        err, ll, brier, ece = test_stats_method(model_obj, savefile, dataset, data_dir,
                                                                corruption=corruption, rotation=rotation,
                                                                batch_size=args.batch_size, cuda=True,
                                                                gpu=args.gpu, MC_samples=n_samples,
                                                                workers=workers, **kwargs)

                        row_to_add.update({"err": err, "ll": ll.item(), "brier": brier.item(), "ece": ece})
                        df = df.append(row_to_add, ignore_index=True)
                        df.to_csv(df_path, index=False)

                    # aur roc measurements
                    for target_dataset in target_datasets[dataset]:
                        row_to_add = row_to_add_proto.copy()
                        row_to_add.update({"target_dataset": target_dataset})
                        if len(df.loc[(df[list(row_to_add)] == pd.Series(row_to_add)).all(axis=1)]) > 0:
                            continue

                        _, _, roc_auc = OOD_AUC_ROC_method(model_obj, savefile, dataset, target_dataset, data_dir,
                                                           batch_size=args.batch_size, cuda=True, gpu=args.gpu,
                                                           MC_samples=n_samples, workers=workers, **kwargs)

                        row_to_add.update({"auc_roc": roc_auc})
                        df = df.append(row_to_add, ignore_index=True)
                        df.to_csv(df_path, index=False)

                    # rejection measurements
                    for target_dataset in target_datasets[dataset]:
                        row_to_add = row_to_add_proto.copy()
                        row_to_add.update({"target_dataset": target_dataset})
                        if len(df.loc[(df[list(row_to_add)] == pd.Series(row_to_add)).all(axis=1)]) > 1: 
                            # the row to add will match the above experiment
                            continue

                        err_props = class_rejection_method(model_obj, savefile, dataset, target_dataset, data_dir,
                                                           batch_size=args.batch_size, cuda=True, gpu=args.gpu,
                                                           MC_samples=n_samples, workers=workers, **kwargs)

                        row_to_add.update({"err_props": err_props})
                        df = df.append(row_to_add, ignore_index=True)
                        df.to_csv(df_path, index=False)

                    # timing measurements
                    row_to_add = row_to_add_proto.copy()
                    row_to_add.update({"batch_size": int(args.batch_size)})
                    if len(df.loc[(df[list(row_to_add)] == pd.Series(row_to_add)).all(axis=1)]) > 0 or (dataset != "MNIST" and dataset != "CIFAR10"):
                        continue

                    batch_time = batch_time_method(model_obj, savefile, dataset, data_dir, batch_size=args.batch_size,
                                                   cuda=True, gpu=args.gpu, MC_samples=n_samples, workers=workers)
                    row_to_add.update({"batch_time": batch_time})
                    df = df.append(row_to_add, ignore_index=True)
                    df.to_csv(df_path, index=False)
