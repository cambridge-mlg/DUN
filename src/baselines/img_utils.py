from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_img_resnet(model, savefile, gpu=None):
    cuda_enabled = torch.cuda.is_available()
    if cuda_enabled:
        if gpu is None:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model)
            model = model.cuda()
            checkpoint = torch.load(savefile)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(savefile, map_location=loc)

            state_dict = checkpoint['state_dict']

            try:
                model.load_state_dict(state_dict)
            except Exception:
                print('Model saved on multiple GPUs, converting to single GPU')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
    else:
        checkpoint = torch.load(savefile, lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    model = model.eval()
    return model


def img_resnet_predict(model, x, MC_samples=1, return_vector=False):
    model = model.eval()
    with torch.no_grad():
        if MC_samples == 1:
            output = model(x).data
            log_probs = F.log_softmax(output, dim=1)
        else:
            pred_samples = []
            for _ in range(MC_samples):
                output = model(x).data
                pred_samples.append(F.log_softmax(output, dim=1))
            pred_samples = torch.stack(pred_samples, dim=1)
            if return_vector:
                log_probs = pred_samples
            else:
                log_probs = torch.logsumexp(pred_samples, dim=1, keepdim=False) - np.log(pred_samples.shape[1])

    return log_probs


def entropy_from_logprobs(log_probs):
    return - (log_probs.exp() * log_probs).sum(dim=1)


def evaluate_predictive_entropy(model, loader, cuda, MC_samples):
    entropy_vec = []
    for images, _ in loader:
        if cuda:
            images = images.cuda(None, non_blocking=True)

        log_probs = img_resnet_predict(model, images, MC_samples=MC_samples)
        entropy_vec.append(entropy_from_logprobs(log_probs))

    entropy_vec = torch.cat(entropy_vec, dim=0)
    return entropy_vec


def get_preds_targets(model, loader, cuda, MC_samples, return_vector=False):
    logprob_vec = []
    target_vec = []
    for images, target in loader:
        if cuda:
            images = images.cuda(None, non_blocking=True)
            target = target.cuda(None, non_blocking=True)

        log_probs = img_resnet_predict(model, images, MC_samples=MC_samples, return_vector=return_vector)
        logprob_vec.append(log_probs.data.cpu())
        target_vec.append(target.data.cpu())

    logprob_vec = torch.cat(logprob_vec, dim=0)
    target_vec = torch.cat(target_vec, dim=0)
    return logprob_vec.data.cpu(), target_vec.data.cpu()


def ensemble_evaluate_predictive_entropy(model, model_saves, loader, cuda, gpu=None):
    model_logprob_vec = []
    for save in model_saves:
        print(save)
        model = load_img_resnet(model, save, gpu=gpu)

        logprob_vec = []
        for images, _ in loader:
            if cuda:
                images = images.cuda(None, non_blocking=True)

            log_probs = img_resnet_predict(model, images, MC_samples=1).data
            logprob_vec.append(log_probs)
        logprob_vec = torch.cat(logprob_vec, dim=0)
        model_logprob_vec.append(logprob_vec)

    model_logprob_vec = torch.stack(model_logprob_vec, dim=0)
    expected_logprobs = model_logprob_vec.logsumexp(dim=0) - np.log(model_logprob_vec.shape[0])
    return entropy_from_logprobs(expected_logprobs)


def ensemble_get_preds_targets(model, model_saves, loader, cuda, gpu=None, return_vector=False):
    model_logprob_vec = []
    for save in model_saves:
        print(save)
        model = load_img_resnet(model, save, gpu=gpu)
        logprob_vec = []
        target_vec = []
        for images, target in loader:
            if cuda:
                images = images.cuda(None, non_blocking=True)
                target = target.cuda(None, non_blocking=True)

            log_probs = img_resnet_predict(model, images, MC_samples=1)
            logprob_vec.append(log_probs.data.cpu())
            target_vec.append(target.data.cpu())

        logprob_vec = torch.cat(logprob_vec, dim=0)
        model_logprob_vec.append(logprob_vec)
        target_vec = torch.cat(target_vec, dim=0)

    model_logprob_vec = torch.stack(model_logprob_vec, dim=1)
    if return_vector:
        return model_logprob_vec.data.cpu(), target_vec.data.cpu()
    else:
        expected_logprobs = model_logprob_vec.logsumexp(dim=1) - np.log(model_logprob_vec.shape[1])
    return expected_logprobs.data.cpu(), target_vec.data.cpu()


def ensemble_time_preds(model, model_saves, x,  gpu=None):

    model_logprob_vec = []

    for save in model_saves:
        model = load_img_resnet(model, save, gpu=gpu)

        log_probs = img_resnet_predict(model, x, MC_samples=1)
        model_logprob_vec.append(log_probs)

    model_logprob_vec = torch.stack(model_logprob_vec, dim=0)
    expected_logprobs = model_logprob_vec.logsumexp(dim=0) - np.log(model_logprob_vec.shape[0])

    return expected_logprobs.data.cpu()
