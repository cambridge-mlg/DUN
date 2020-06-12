import time

import torch
import torch.nn.functional as F
import numpy as np

from src.baselines.img_utils import load_img_resnet, img_resnet_predict
from src.baselines.img_utils import evaluate_predictive_entropy, ensemble_evaluate_predictive_entropy
from src.baselines.img_utils import get_preds_targets, ensemble_get_preds_targets, ensemble_time_preds
from src.utils import torch_onehot
from src.datasets.image_loaders import get_image_loader

from experiments.callibration import cat_callibration, expected_callibration_error
from experiments.image.OOD_utils import evaluate_predictive_entropy_DUN, get_preds_targets_DUN
from experiments.image.OOD_utils import cross_load_dataset, get_roc_params
from experiments.image.OOD_utils import load_corrupted_dataset, rotate_load_dataset


def class_brier(y, log_probs=None, probs=None):
    assert log_probs is None or probs is None
    if log_probs is not None:
        probs = log_probs.exp()
    elif probs is not None:
        pass
    else:
        raise Exception('either log_probs or probs must not be None')
    assert probs.max().item() <= (1. + 1e-4) and probs.min().item() >= 0.
    if len(y.shape) > 1:
        y = y.squeeze(1)
    y_oh = torch_onehot(y, probs.shape[1])
    brier = (probs - y_oh).pow(2).sum(dim=1).mean(dim=0)
    return brier


def class_err(y, model_out):
    pred = model_out.max(dim=1, keepdim=False)[1]  # get the index of the max probability
    err = pred.ne(y.data).sum().item() / y.shape[0]
    return err


def class_ll(y, log_probs=None, probs=None, eps=1e-40):
    assert log_probs is None or probs is None
    if log_probs is not None:
        pass
    elif probs is not None:
        log_probs = torch.log(probs.clamp(min=eps))
    else:
        raise Exception('either log_probs or probs must not be None')
    nll = F.nll_loss(log_probs, y, reduction='mean')
    return -nll


def class_ECE(y, log_probs=None, probs=None, nbins=10, top_k=1):
    assert log_probs is None or probs is None
    if log_probs is not None:
        probs = log_probs.exp()
    elif probs is not None:
        pass
    else:
        raise Exception('either log_probs or probs must not be None')
    assert probs.max().item() <= (1. + 1e-4) and probs.min().item() >= 0.
    probs = probs.clamp(max=(1-1e-8))
    bin_probs, _, _, bin_counts, reference = cat_callibration(probs.cpu().numpy(), y.cpu().numpy(),
                                                              nbins, top_k=top_k)
    ECE = expected_callibration_error(bin_probs, reference, bin_counts)
    return ECE


# OOD AUC-ROC

def baseline_OOD_AUC_ROC(model, savefile, source_dset, target_dset, data_dir, batch_size=256, cuda=True, gpu=None,
                         MC_samples=1, workers=4):
    model = load_img_resnet(model, savefile, gpu=gpu)

    source_loader, target_loader = cross_load_dataset(source_dset, target_dset, data_dir=data_dir,
                                                      batch_size=batch_size, cuda=cuda, workers=workers)
    source_entropy = evaluate_predictive_entropy(model, source_loader, cuda=cuda, MC_samples=MC_samples).cpu().numpy()
    target_entropy = evaluate_predictive_entropy(model, target_loader, cuda=cuda, MC_samples=MC_samples).cpu().numpy()
    fpr, tpr, roc_auc = get_roc_params(source_entropy, target_entropy)
    return fpr, tpr, roc_auc


def ensemble_OOD_AUC_ROC(model, savefile_list, source_dset, target_dset, data_dir, batch_size=256, cuda=True,
                         gpu=None, MC_samples=0, workers=4):
    source_loader, target_loader = cross_load_dataset(source_dset, target_dset, data_dir=data_dir,
                                                      batch_size=batch_size, cuda=cuda, workers=workers)
    source_entropy = ensemble_evaluate_predictive_entropy(model, savefile_list, source_loader, cuda=cuda,
                                                          gpu=gpu).cpu().numpy()
    target_entropy = ensemble_evaluate_predictive_entropy(model, savefile_list, target_loader, cuda=cuda,
                                                          gpu=gpu).cpu().numpy()
    fpr, tpr, roc_auc = get_roc_params(source_entropy, target_entropy)
    return fpr, tpr, roc_auc


def DUN_OOD_AUC_ROC(net, savefile, source_dset, target_dset, data_dir, batch_size=256, cuda=True,
                    gpu=None, MC_samples=0, workers=4, d_posterior=None):
    source_loader, target_loader = cross_load_dataset(source_dset, target_dset, data_dir=data_dir,
                                                      batch_size=batch_size, cuda=cuda, workers=workers)
    net.load(savefile)
    if d_posterior is not None:
        net.prob_model.current_posterior = d_posterior

    source_entropy = evaluate_predictive_entropy_DUN(net, source_loader)
    target_entropy = evaluate_predictive_entropy_DUN(net, target_loader)

    fpr, tpr, roc_auc = get_roc_params(source_entropy, target_entropy)
    return fpr, tpr, roc_auc


# test-err loglike, ECE, brier

def baseline_test_stats(model, savefile, dset, data_dir, corruption=None, rotation=None, batch_size=256, cuda=True,
                        gpu=None, MC_samples=1, workers=4, iterate=False, no_ece=False):
    model = load_img_resnet(model, savefile, gpu=gpu)

    assert not (corruption is not None and rotation is not None)
    if corruption is None and rotation is None:
        _, _, val_loader, _, _, _ = \
            get_image_loader(dset, batch_size, cuda=cuda, workers=workers, distributed=False, data_dir=data_dir)
    elif corruption is not None:
        val_loader = load_corrupted_dataset(dset, severity=corruption, data_dir=data_dir, batch_size=batch_size,
                                            cuda=cuda, workers=workers)
    elif rotation is not None:
        val_loader = rotate_load_dataset(dset, rotation, data_dir=data_dir,
                                         batch_size=batch_size, cuda=cuda, workers=workers)

    logprob_vec, target_vec = get_preds_targets(model, val_loader, cuda, MC_samples, return_vector=iterate)

    if iterate:
        brier_vec = []
        err_vec = []
        ll_vec = []
        ece_vec = []

        for n_samples in range(1, logprob_vec.shape[1]+1):
            comb_logprobs = torch.logsumexp(logprob_vec[:, :n_samples, :], dim=1, keepdim=False) - np.log(n_samples)

            brier_vec.append(class_brier(y=target_vec, log_probs=comb_logprobs, probs=None))
            err_vec.append(class_err(y=target_vec, model_out=comb_logprobs))
            ll_vec.append(class_ll(y=target_vec, log_probs=comb_logprobs, probs=None, eps=1e-40))
            ece_vec.append(float('nan') if no_ece else class_ECE(y=target_vec, log_probs=comb_logprobs,
                                                                 probs=None, nbins=10))
        return err_vec, ll_vec, brier_vec, ece_vec

    brier = class_brier(y=target_vec, log_probs=logprob_vec, probs=None)
    err = class_err(y=target_vec, model_out=logprob_vec)
    ll = class_ll(y=target_vec, log_probs=logprob_vec, probs=None, eps=1e-40)
    ece = class_ECE(y=target_vec, log_probs=logprob_vec, probs=None, nbins=10)
    return err, ll, brier, ece


def ensemble_test_stats(model, savefile_list, dset, data_dir, corruption=None, rotation=None, batch_size=256, cuda=True,
                        gpu=None, MC_samples=0, workers=4,  iterate=False):
    assert not (corruption is not None and rotation is not None)
    if corruption is None and rotation is None:
        _, _, val_loader, _, _, _ = \
            get_image_loader(dset, batch_size, cuda=cuda, workers=workers, distributed=False, data_dir=data_dir)
    elif corruption is not None:
        val_loader = load_corrupted_dataset(dset, severity=corruption, data_dir=data_dir, batch_size=batch_size,
                                            cuda=cuda, workers=workers)
    elif rotation is not None:
        val_loader = rotate_load_dataset(dset, rotation, data_dir=data_dir,
                                         batch_size=batch_size, cuda=cuda, workers=workers)

    logprob_vec, target_vec = ensemble_get_preds_targets(model, savefile_list, val_loader, cuda=cuda, gpu=gpu,
                                                         return_vector=iterate)
    if iterate:
        brier_vec = []
        err_vec = []
        ll_vec = []
        ece_vec = []

        for n_samples in range(1, logprob_vec.shape[1] + 1):
            comb_logprobs = torch.logsumexp(logprob_vec[:, :n_samples, :], dim=1, keepdim=False) - np.log(n_samples)

            brier_vec.append(class_brier(y=target_vec, log_probs=comb_logprobs, probs=None))
            err_vec.append(class_err(y=target_vec, model_out=comb_logprobs))
            ll_vec.append(class_ll(y=target_vec, log_probs=comb_logprobs, probs=None, eps=1e-40))
            ece_vec.append(class_ECE(y=target_vec, log_probs=comb_logprobs, probs=None, nbins=10))
        return err_vec, ll_vec, brier_vec, ece_vec

    brier = class_brier(y=target_vec, log_probs=logprob_vec, probs=None)
    err = class_err(y=target_vec, model_out=logprob_vec)
    ll = class_ll(y=target_vec, log_probs=logprob_vec, probs=None, eps=1e-40)
    ece = class_ECE(y=target_vec, log_probs=logprob_vec, probs=None, nbins=10)
    return err, ll, brier, ece


def DUN_test_stats(net, savefile, dset, data_dir, corruption=None, rotation=None, batch_size=256, cuda=True,
                   gpu=None, MC_samples=0, workers=4, d_posterior=None):
    assert not (corruption is not None and rotation is not None)
    if corruption is None and rotation is None:
        _, _, val_loader, _, _, _ = \
            get_image_loader(dset, batch_size, cuda=cuda, workers=workers, distributed=False, data_dir=data_dir)
    elif corruption is not None:
        val_loader = load_corrupted_dataset(dset, severity=corruption, data_dir=data_dir, batch_size=batch_size,
                                            cuda=cuda, workers=workers)
    elif rotation is not None:
        val_loader = rotate_load_dataset(dset, rotation, data_dir=data_dir,
                                         batch_size=batch_size, cuda=cuda, workers=workers)

    net.load(savefile)

    if d_posterior is not None:
        net.prob_model.current_posterior = d_posterior

    prob_vec, target_vec = get_preds_targets_DUN(net, val_loader)

    brier = class_brier(y=target_vec, probs=prob_vec, log_probs=None)
    err = class_err(y=target_vec, model_out=prob_vec)
    ll = class_ll(y=target_vec, probs=prob_vec, log_probs=None, eps=1e-40)
    ece = class_ECE(y=target_vec, probs=prob_vec, log_probs=None, nbins=10)

    return err, ll, brier, ece


def baseline_class_rej(model, savefile, source_dset, target_dset, data_dir, batch_size=256, cuda=True,
                       gpu=None, MC_samples=1, rejection_step=0.005, workers=4):
    model = load_img_resnet(model, savefile, gpu=gpu)

    source_loader, target_loader = cross_load_dataset(source_dset, target_dset, data_dir=data_dir,
                                                      batch_size=batch_size, cuda=cuda, workers=workers)

    source_entropy = evaluate_predictive_entropy(model, source_loader, cuda=cuda, MC_samples=MC_samples).cpu().numpy()
    target_entropy = evaluate_predictive_entropy(model, target_loader, cuda=cuda, MC_samples=MC_samples).cpu().numpy()

    logprob_vec, target_vec = get_preds_targets(model, source_loader, cuda, MC_samples)
    pred = logprob_vec.max(dim=1, keepdim=False)[1]  # get the index of the max probability
    err_vec_in = pred.ne(target_vec.data).cpu().numpy()
    err_vec_out = np.ones(target_entropy.shape[0])

    full_err_vec = np.concatenate([err_vec_in, err_vec_out], axis=0)
    full_entropy_vec = np.concatenate([source_entropy, target_entropy], axis=0)
    sort_entropy_idxs = np.argsort(full_entropy_vec, axis=0)
    Npoints = sort_entropy_idxs.shape[0]

    err_props = []

    for rej_prop in np.arange(0, 1, rejection_step):
        N_reject = np.round(Npoints * rej_prop).astype(int)
        if N_reject > 0:
            accepted_idx = sort_entropy_idxs[:-N_reject]
        else:
            accepted_idx = sort_entropy_idxs

        err_props.append(full_err_vec[accepted_idx].sum() / accepted_idx.shape[0])

        assert err_props[-1].max() <= 1 and err_props[-1].min() >= 0

    return np.array(err_props)


def DUN_class_rej(net, savefile, source_dset, target_dset, data_dir, batch_size=256, cuda=True, gpu=None,
                  MC_samples=1, rejection_step=0.005, workers=4, d_posterior=None):
    source_loader, target_loader = cross_load_dataset(source_dset, target_dset, data_dir=data_dir,
                                                      batch_size=batch_size, cuda=cuda, workers=workers)

    net.load(savefile)
    if d_posterior is not None:
        net.prob_model.current_posterior = d_posterior

    source_entropy = evaluate_predictive_entropy_DUN(net, source_loader)
    target_entropy = evaluate_predictive_entropy_DUN(net, target_loader)

    logprob_vec, target_vec = get_preds_targets_DUN(net, source_loader)
    pred = logprob_vec.max(dim=1, keepdim=False)[1]  # get the index of the max probability
    err_vec_in = pred.ne(target_vec.data).cpu().numpy()
    err_vec_out = np.ones(target_entropy.shape[0])

    full_err_vec = np.concatenate([err_vec_in, err_vec_out], axis=0)
    full_entropy_vec = np.concatenate([source_entropy, target_entropy], axis=0)
    sort_entropy_idxs = np.argsort(full_entropy_vec, axis=0)
    Npoints = sort_entropy_idxs.shape[0]

    err_props = []

    for rej_prop in np.arange(0, 1, rejection_step):
        N_reject = np.round(Npoints * rej_prop).astype(int)
        if N_reject > 0:
            accepted_idx = sort_entropy_idxs[:-N_reject]
        else:
            accepted_idx = sort_entropy_idxs

        err_props.append(full_err_vec[accepted_idx].sum() / accepted_idx.shape[0])

        assert err_props[-1].max() <= 1 and err_props[-1].min() >= 0

    return np.array(err_props)


def ensemble_class_rej(model, savefile_list, source_dset, target_dset, data_dir, batch_size=256,
                       cuda=True, gpu=None, MC_samples=1, rejection_step=0.005, workers=4):
    source_loader, target_loader = cross_load_dataset(source_dset, target_dset, data_dir=data_dir,
                                                      batch_size=batch_size, cuda=cuda, workers=workers)

    source_entropy = ensemble_evaluate_predictive_entropy(model, savefile_list, source_loader, cuda=cuda,
                                                          gpu=gpu).cpu().numpy()
    target_entropy = ensemble_evaluate_predictive_entropy(model, savefile_list, target_loader, cuda=cuda,
                                                          gpu=gpu).cpu().numpy()

    logprob_vec, target_vec = ensemble_get_preds_targets(model, savefile_list, source_loader, cuda=cuda, gpu=gpu)
    pred = logprob_vec.max(dim=1, keepdim=False)[1]  # get the index of the max probability
    err_vec_in = pred.ne(target_vec.data).cpu().numpy()
    err_vec_out = np.ones(target_entropy.shape[0])

    full_err_vec = np.concatenate([err_vec_in, err_vec_out], axis=0)
    full_entropy_vec = np.concatenate([source_entropy, target_entropy], axis=0)
    sort_entropy_idxs = np.argsort(full_entropy_vec, axis=0)
    Npoints = sort_entropy_idxs.shape[0]

    err_props = []

    for rej_prop in np.arange(0, 1, rejection_step):
        N_reject = np.round(Npoints * rej_prop).astype(int)
        if N_reject > 0:
            accepted_idx = sort_entropy_idxs[:-N_reject]
        else:
            accepted_idx = sort_entropy_idxs

        err_props.append(full_err_vec[accepted_idx].sum() / accepted_idx.shape[0])

        assert err_props[-1].max() <= 1 and err_props[-1].min() >= 0

    return np.array(err_props)


# batch time
def baseline_batch_time(model, savefile, dset, data_dir, batch_size=256, cuda=True,
                        gpu=None, MC_samples=0, workers=4, big_data=False):
    model = load_img_resnet(model, savefile, gpu=gpu)

    _, _, val_loader, _, _, _ = \
        get_image_loader(dset, batch_size, cuda=cuda, workers=workers, distributed=False, data_dir=data_dir)

    times = []
    for i, (images, target) in enumerate(val_loader):
        if big_data and i > 5:
            break

        if cuda:
            images = images.cuda(None, non_blocking=True)
            target = target.cuda(None, non_blocking=True)
        data_time = time.time()

        _ = img_resnet_predict(model, images, MC_samples=MC_samples)

        batch_time = time.time() - data_time
        if not (big_data and i == 0):
            times.append(batch_time)

    return np.mean(times)


def ensemble_batch_time(model, savefile_list, dset, data_dir, batch_size=256, cuda=True,
                        gpu=None, MC_samples=0, workers=4, big_data=False):
    _, _, val_loader, _, _, _ = \
        get_image_loader(dset, batch_size, cuda=cuda, workers=workers, distributed=False, data_dir=data_dir)

    times = []
    for i, (images, target) in enumerate(val_loader):
        if big_data and i > 5:
            break

        data_time = time.time()
        if cuda:
            images = images.cuda(None, non_blocking=True)
            target = target.cuda(None, non_blocking=True)

        _ = ensemble_time_preds(model, savefile_list, images, gpu)

        batch_time = time.time() - data_time
        if not (big_data and i == 0):
            times.append(batch_time)

    return np.mean(times)


def DUN_batch_time(net, savefile, dset, data_dir, batch_size=256, cuda=True,
                   gpu=None, MC_samples=0, workers=4, big_data=False):
    _, _, val_loader, _, _, _ = \
        get_image_loader(dset, batch_size, cuda=cuda, workers=workers, distributed=False, data_dir=data_dir)

    net.load(savefile)

    times = []
    for i, (images, _) in enumerate(val_loader):
        if big_data and i > 5:
            break

        data_time = time.time()

        _ = net.fast_predict(images).data

        batch_time = time.time() - data_time
        if not (big_data and i == 0):
            times.append(batch_time)

    return np.mean(times)
