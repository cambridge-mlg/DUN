import os
import sys
import pickle

import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from PIL import Image


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)


def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


def np_get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)
        if not v.is_cuda and cuda:
            v = v.cuda()
        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)
        out.append(v)
    return out


def rms(x, y):
    return F.mse_loss(x, y, reduction='mean').sqrt()


def get_rms(mu, y, y_means, y_stds):
    x_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    assert x_un.shape[1] == 1
    assert y_un.shape[1] == 1
    return rms(x_un, y_un)


def get_gauss_loglike(mu, sigma, y, y_means, y_stds):
    mu_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    sigma_un = sigma * y_stds
    assert mu_un.shape[1] == 1
    assert y_un.shape[1] == 1
    assert sigma_un.shape[1] == 1
    dist = Normal(mu_un, sigma_un)
    return dist.log_prob(y_un).mean(axis=0).item()  # mean over datapoints


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return (nb_samples + (-nb_samples % batch_size)) / batch_size  # roundup division

    return nb_samples / batch_size


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))

    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size: (i + 1) * batch_size]


class BaseNet(object):
    def __init__(self):
        cprint('c', '\nNet:')
        self.scheduler = None

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def make_scheduler(self, gamma=0.1, milestones=None):
        self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

    def update_lr(self):
        self.epoch += 1
        if self.scheduler is not None:
            self.scheduler.step()

    def save(self, filename, best_err=None):
        cprint('c', 'Writting %s\n' % filename)
        if best_err is None:
            try:
                torch.save({
                    'epoch': self.epoch,
                    'lr': self.lr,
                    'model': self.model,
                    'prob_model': self.prob_model,
                    'f_neg_loglike': self.f_neg_loglike,
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler}, filename)
            except Exception:
                torch.save({
                    'epoch': self.epoch,
                    'lr': self.lr,
                    'model': self.model,
                    'f_neg_loglike': self.f_neg_loglike,
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler}, filename)

        else:
            torch.save({
                'epoch': self.epoch,
                'best_err': best_err,
                'lr': self.lr,
                'model': self.model,
                'prob_model': self.prob_model,
                'f_neg_loglike': self.f_neg_loglike,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler}, filename)

        # TODO: fix this try except hack here and below

    def load(self, filename, parallel=False, to_cpu=False):
        if to_cpu:
            state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        if parallel:
            assert isinstance(self.model, nn.DataParallel)
            self.model = self.model.cuda()

        self.model = state_dict['model']
        try:
            self.prob_model = state_dict['prob_model']
        except Exception:
            pass
        self.f_neg_loglike = state_dict['f_neg_loglike']
        self.optimizer = state_dict['optimizer']
        self.scheduler = state_dict['scheduler']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        try:
            best_err = state_dict['best_err']
            return self.epoch, best_err
        except Exception:
            return self.epoch


class DatafeedImage(data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


class Datafeed(data.Dataset):

    def __init__(self, x_train, y_train=None, transform=None):
        self.data = x_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[index]
        else:
            return img

    def __len__(self):
        return len(self.data)


def torch_onehot(y, Nclass):
    if y.is_cuda:
        y = y.type(torch.cuda.LongTensor)
    else:
        y = y.type(torch.LongTensor)
    y_onehot = torch.zeros((y.shape[0], Nclass)).type(y.type())
    # In your for loop
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            return pickle.load(f, encoding="latin1")


def pt2np_image(images):
    """Convert pytorch images to ones that can be plotted with pyplot imshow"""
    images = images.data.cpu().numpy()
    images = np.swapaxes(images, 1, 3)
    images = np.swapaxes(images, 1, 2)
    return images
