import argparse
import os
import shutil
import time
import glob

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed

from src.datasets.image_loaders import get_image_loader
from src.utils import mkdir, save_object, cprint, load_object
from src.probability import depth_categorical_VI
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_img_resnets import resnet18, resnet34, resnet50, resnet101


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=["CIFAR10", "CIFAR100", "SVHN", "MNIST", "Fashion"],
                    help='dataset to train (default: MNIST)')
parser.add_argument('--data_dir', type=str, default='../data/',
                    help='directory where datasets are saved (default: ../data/)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=None, type=int,
                    help='number of total epochs to run (if None, use dataset default)')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--savedir', default='./results/', type=str,
                    help='path where to save checkpoints (default: ./results/)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use. (default: 0)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size to use. (default: 256)')
parser.add_argument('--model', type=str, default='resnet50',
                    choices=["resnet18", "resnet32", "resnet50", "resnet101"],
                    help='model to train (default: resnet50)')
parser.add_argument('--start_depth', default=1, type=int,
                    help='first layer to be uncertain about (default: 1)')
parser.add_argument('--end_depth', default=13, type=int,
                    help='last layer to be uncertain about + 1 (default: 13)')
parser.add_argument('--q_nograd_its', default=0, type=int,
                    help='number of warmup epochs (where q is not learnt) (default: 0)')

best_err1 = 1
lr = 0.1
momentum = 0.9


def main(args):
    dataset = args.dataset
    workers = args.workers
    epochs = args.epochs
    weight_decay = args.weight_decay
    resume = args.resume
    savedir = args.savedir
    gpu = args.gpu
    q_nograd_its = args.q_nograd_its
    batch_size = args.batch_size
    data_dir = args.data_dir
    start_depth = args.start_depth
    end_depth = args.end_depth
    model = args.model

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    savedir += "/" + "_".join([dataset, model, "DUN", f"warm{q_nograd_its}", f"{start_depth}-{end_depth}"]) 
    savedir += "_wd" if weight_decay != 0 else "_nowd"
    num = len(glob.glob(savedir + "*"))
    savedir += f"_{num}"

    epoch_dict = {
        'Imagenet': 90,
        'SmallImagenet': 90,
        'CIFAR10': 300,
        'CIFAR100': 300,
        'SVHN': 90,
        'Fashion': 90,
        'MNIST': 90
    }

    milestone_dict = {
        'Imagenet': [30, 60],  # This is pytorch default
        'SmallImagenet': [30, 60],
        'CIFAR10': [150, 225],
        'CIFAR100': [150, 225],
        'SVHN': [50, 70],
        'Fashion': [40, 70],
        'MNIST': [40, 70]
    }

    if epochs is None:
        epochs = epoch_dict[dataset]
    milestones = milestone_dict[dataset]

    initial_conv = '3x3' if dataset in ['Imagenet', 'SmallImagenet'] else '1x3'
    input_chanels = 1 if dataset in ['MNIST', 'Fashion'] else 3
    if dataset in ['Imagenet', 'SmallImagenet']:
        num_classes = 1000
    elif dataset in ['CIFAR100']:
        num_classes = 100
    else:
        num_classes = 10

    if model == 'resnet18':
        model_class = resnet18
    elif model == 'resnet18':
        model_class = resnet34
    elif model == 'resnet50':
        model_class = resnet50
    elif model == 'resnet101':
        model_class = resnet101
    else:
        raise Exception('requested model not implemented')

    cuda = torch.cuda.is_available()
    print('cuda', cuda)
    assert cuda

    n_layers = end_depth - start_depth

    prior_probs = [1 / (n_layers)] * (n_layers)
    prob_model = depth_categorical_VI(prior_probs, cuda=cuda)

    model = model_class(arch_uncert=True, start_depth=start_depth, end_depth=end_depth, num_classes=num_classes,
                        zero_init_residual=True, initial_conv=initial_conv, concat_pool=False,
                        input_chanels=input_chanels, p_drop=0)

    N_train = 0

    net = DUN_VI(model, prob_model, N_train, lr=lr, momentum=momentum, weight_decay=weight_decay, cuda=cuda,
                 schedule=milestones, regression=False, pred_sig=None)

    train_loop(net, dname=dataset, data_dir=data_dir, epochs=epochs, workers=workers, resume=resume, savedir=savedir,
               q_nograd_its=q_nograd_its, batch_size=batch_size)


def train_loop(net, dname, data_dir, epochs=90, workers=4, resume='', savedir='./',
               save_all_epochs=False, q_nograd_its=0, batch_size=256):
    mkdir(savedir)
    global best_err1

    # Load data here:
    _, train_loader, val_loader, _, _, Ntrain = \
        get_image_loader(dname, batch_size, cuda=True, workers=workers, distributed=False, data_dir=data_dir)

    net.N_train = Ntrain

    start_epoch = 0

    marginal_loglike = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    dev_loss = np.zeros(epochs)

    err_train = np.zeros(epochs)
    err_dev = np.zeros(epochs)

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            start_epoch, best_err1 = net.load(resume)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

        candidate_progress_file = resume.split('/')
        candidate_progress_file = '/'.join(candidate_progress_file[:-1]) + '/stats_array.pkl'

        if os.path.isfile(candidate_progress_file):
            print("=> found progress file at '{}'".format(candidate_progress_file))
            try:
                marginal_loglike, err_train, train_loss, err_dev, dev_loss = \
                    load_object(candidate_progress_file)
                print("=> Loaded progress file at '{}'".format(candidate_progress_file))
            except Exception:
                print("=> Unable to load progress file at '{}'".format(candidate_progress_file))
        else:
            print("=> NOT found progress file at '{}'".format(candidate_progress_file))

    if q_nograd_its > 0:
        net.prob_model.q_logits.requires_grad = False

    for epoch in range(start_epoch, epochs):
        if q_nograd_its > 0 and epoch == q_nograd_its:
            net.prob_model.q_logits.requires_grad = True

        tic = time.time()
        nb_samples = 0
        for x, y in train_loader:
            marg_loglike_estimate, minus_loglike, err = net.fit(x, y)

            marginal_loglike[epoch] += marg_loglike_estimate * x.shape[0]
            err_train[epoch] += err * x.shape[0]
            train_loss[epoch] += minus_loglike * x.shape[0]
            nb_samples += len(x)

        marginal_loglike[epoch] /= nb_samples
        train_loss[epoch] /= nb_samples
        err_train[epoch] /= nb_samples

        toc = time.time()

        # ---- print
        print('\n depth approx posterior', net.prob_model.current_posterior.data.cpu().numpy())
        print("it %d/%d, ELBO/evidence %.4f, pred minus loglike = %f, err = %f" %
              (epoch, epochs, marginal_loglike[epoch], train_loss[epoch], err_train[epoch]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))

        net.update_lr()

        # ---- dev
        tic = time.time()
        nb_samples = 0
        for x, y in val_loader:
            minus_loglike, err = net.eval(x, y)

            dev_loss[epoch] += minus_loglike * x.shape[0]
            err_dev[epoch] += err * x.shape[0]
            nb_samples += len(x)

        dev_loss[epoch] /= nb_samples
        err_dev[epoch] /= nb_samples

        toc = time.time()

        cprint('g', '     pred minus loglike = %f, err = %f\n' % (dev_loss[epoch], err_dev[epoch]), end="")
        cprint('g', '    time: %f seconds\n' % (toc - tic))

        filename = 'checkpoint.pth.tar'
        if save_all_epochs:
            filename = str(epoch) + '_' + filename
        net.save(os.path.join(savedir, filename), best_err1)
        if err_dev[epoch] < best_err1:
            best_err1 = err_dev[epoch]
            cprint('b', 'best top1 dev err: %f' % err_dev[epoch])
            shutil.copyfile(os.path.join(savedir, filename), os.path.join(savedir, 'model_best.pth.tar'))

        all_results = [marginal_loglike, err_train, train_loss, err_dev, dev_loss]
        save_object(all_results, os.path.join(savedir, 'stats_array.pkl'))


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
