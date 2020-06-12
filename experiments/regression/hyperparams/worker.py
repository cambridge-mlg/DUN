import torch
import numpy as np
from sklearn.model_selection import train_test_split
from hpbandster.core.worker import Worker
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS

from src.probability import pMOM_loglike, diag_w_Gauss_loglike, depth_categorical_VI
from src.utils import Datafeed
from src.datasets import load_flight, gen_spirals, load_gap_UCI
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_fc_models import arq_uncert_fc_resnet, arq_uncert_fc_MLP

from src.baselines.training_wrappers import regression_baseline_net, regression_baseline_net_VI
from src.baselines.SGD import SGD_regression_homo
from src.baselines.mfvi import MFVI_regression_homo
from src.baselines.dropout import dropout_regression_homo


class DUN_none_Worker(Worker):
    def __init__(self, *args, network, width, batch_size, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = width
        self.batch_size = batch_size

        # setup default hyper-parameter search ranges
        self.lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1, default_value=1e-2, log=True)
        self.momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.0, upper=0.99, default_value=0.5, log=False)
        self.n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=40, default_value=5)

        self.network = network

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # setup dataloaders
        if torch.cuda.is_available():
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=True, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=True, num_workers=0)
        else:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=False, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=False, num_workers=0)
        # setup model
        n_layers = config['n_layers']

        prior_probs = [1/(n_layers + 1)] * (n_layers + 1)

        cuda = torch.cuda.is_available()

        if self.network == 'MLP':
            model = arq_uncert_fc_MLP(self.input_dim, self.output_dim, self.width, n_layers, w_prior=None)
        elif self.network == 'ResNet':
            model = arq_uncert_fc_resnet(self.input_dim, self.output_dim, self.width, n_layers, w_prior=None)

        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, self.N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                     schedule=None, regression=self.regression)

        return train_loop(net, trainloader, valloader, budget, self.early_stop)

    def get_configspace(self):
        """
        It builds the configuration space with the needed hyperparameters.
        """
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([self.lr, self.momentum])
        cs.add_hyperparameters([self.n_layers])

        return cs


class DUN_wd_Worker(Worker):
    # DUN_none + weight decay
    def __init__(self, *args, network, width, batch_size, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = width
        self.batch_size = batch_size

        # setup default hyper-parameter search ranges
        self.lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1, default_value=1e-2, log=True)
        self.momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.0, upper=0.99, default_value=0.5, log=False)
        self.n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=40, default_value=5)
        self.weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-1, default_value=5e-4,
                                                           log=True)
        self.network = network

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # setup dataloaders
        if torch.cuda.is_available():
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=True, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=True, num_workers=0)
        else:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=False, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=False, num_workers=0)
        # setup model
        n_layers = config['n_layers']

        prior_probs = [1/(n_layers + 1)] * (n_layers + 1)

        cuda = torch.cuda.is_available()

        if self.network == 'MLP':
            model = arq_uncert_fc_MLP(self.input_dim, self.output_dim, self.width, n_layers, w_prior=None)
        elif self.network == 'ResNet':
            model = arq_uncert_fc_resnet(self.input_dim, self.output_dim, self.width, n_layers, w_prior=None)

        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, self.N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                     schedule=None, regression=self.regression, weight_decay=config['weight_decay'])

        return train_loop(net, trainloader, valloader, budget, self.early_stop)

    def get_configspace(self):
        """
        It builds the configuration space with the needed hyperparameters.
        """
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([self.lr, self.momentum])
        cs.add_hyperparameters([self.n_layers])
        cs.add_hyperparameters([self.weight_decay])

        return cs


class DUN_prior_Worker(Worker):
    def __init__(self, *args, network, width, batch_size, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = width
        self.batch_size = batch_size

        # setup default hyper-parameter search ranges
        self.lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1, default_value=1e-2, log=True)
        self.momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.0, upper=0.99, default_value=0.5, log=False)

        self.n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=40, default_value=5)

        self.prior = CSH.CategoricalHyperparameter('prior', ['gauss', 'pMOM'])
        self.BMA_prior = CSH.CategoricalHyperparameter('BMA_prior', [True, False])
        self.gauss_σ2 = CSH.UniformFloatHyperparameter('gauss_σ2', lower=1e-2, upper=10, default_value=1, log=True)
        self.pMOM_σ2 = CSH.UniformFloatHyperparameter('pMOM_σ2', lower=1e-2, upper=10, default_value=1, log=True)
        self.pMOM_r = CSH.UniformIntegerHyperparameter('pMOM_r', lower=1, upper=3)

        self.network = network

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # setup dataloaders
        if torch.cuda.is_available():
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=True, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=True, num_workers=0)
        else:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=False, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=False, num_workers=0)
        # setup model
        n_layers = config['n_layers']

        prior_probs = [1 / (n_layers + 1)] * (n_layers + 1)

        cuda = torch.cuda.is_available()

        if config['prior'] == 'gauss':
            w_prior = diag_w_Gauss_loglike(μ=0, σ2=config['gauss_σ2'])
        elif config['prior'] == 'pMOM':
            w_prior = pMOM_loglike(r=config['pMOM_r'], τ=1, σ2=config['pMOM_σ2'])
        else:
            raise Exception('We should be using a prior')

        if self.network == 'MLP':
            model = arq_uncert_fc_MLP(self.input_dim, self.output_dim, self.width, n_layers, w_prior=w_prior,
                                      BMA_prior=config['BMA_prior'])
        elif self.network == 'ResNet':
            model = arq_uncert_fc_resnet(self.input_dim, self.output_dim, self.width, n_layers, w_prior=w_prior,
                                         BMA_prior=config['BMA_prior'])

        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, self.N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                     schedule=None, regression=self.regression)

        return train_loop(net, trainloader, valloader, budget, self.early_stop)

    def get_configspace(self):
        """
        It builds the configuration space with the needed hyperparameters.
        """
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([self.lr, self.momentum])

        cs.add_hyperparameters([self.n_layers])

        cs.add_hyperparameters(
            [self.prior, self.gauss_σ2, self.pMOM_σ2, self.pMOM_r])

        cs.add_hyperparameters([self.BMA_prior])

        cond = CS.EqualsCondition(self.gauss_σ2, self.prior, 'gauss')
        cs.add_condition(cond)
        cond = CS.EqualsCondition(self.pMOM_σ2, self.prior, 'pMOM')
        cs.add_condition(cond)
        cond = CS.EqualsCondition(self.pMOM_r, self.prior, 'pMOM')
        cs.add_condition(cond)

        return cs


class SGDWorker(Worker):
    def __init__(self, *args, network, width, batch_size, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = width
        self.batch_size = batch_size

        # setup default hyper-parameter search ranges
        self.lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1, default_value=1e-2, log=True)
        self.momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.0, upper=0.99, default_value=0.5,
                                                       log=False)
        self.n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=40, default_value=2)
        self.weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-1, default_value=5e-4,
                                                           log=True)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # setup dataloaders
        if torch.cuda.is_available():
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=True, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=True, num_workers=0)
        else:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=False, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=False, num_workers=0)
        # setup model
        n_layers = config['n_layers']

        cuda = torch.cuda.is_available()

        model = SGD_regression_homo(input_dim=self.input_dim, output_dim=self.output_dim,
                                    width=self.width, n_layers=n_layers)
        net = regression_baseline_net(model, self.N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                                      schedule=None, weight_decay=config['weight_decay'])

        return train_loop(net, trainloader, valloader, budget, self.early_stop)

    def get_configspace(self):
        """
        It builds the configuration space with the needed hyperparameters.
        """
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([self.lr, self.momentum])
        cs.add_hyperparameters([self.n_layers])
        cs.add_hyperparameters([self.weight_decay])
        return cs


class MFVIWorker(Worker):
    def __init__(self, *args, network, width, batch_size, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = width
        self.batch_size = batch_size

        # setup default hyper-parameter search ranges
        self.lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1, default_value=1e-2, log=True)
        self.momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.0, upper=0.99, default_value=0.5,
                                                       log=False)
        self.n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=40, default_value=2)
        self.prior_std = CSH.UniformFloatHyperparameter('prior_std', lower=1e-2, upper=10, default_value=1, log=True)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # setup dataloaders
        if torch.cuda.is_available():
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=True, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=True, num_workers=0)
        else:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=False, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=False, num_workers=0)
        # setup model
        n_layers = config['n_layers']

        cuda = torch.cuda.is_available()

        model = MFVI_regression_homo(input_dim=self.input_dim, output_dim=self.output_dim,
                                     width=self.width, n_layers=n_layers, prior_sig=config['prior_std'])

        net = regression_baseline_net_VI(model, self.N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                                         schedule=None, MC_samples=20, train_samples=3)

        return train_loop(net, trainloader, valloader, budget, self.early_stop)

    def get_configspace(self):
        """
        It builds the configuration space with the needed hyperparameters.
        """
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([self.lr, self.momentum])
        cs.add_hyperparameters([self.n_layers])
        cs.add_hyperparameters([self.prior_std])
        return cs


class DropoutWorker(Worker):
    def __init__(self, *args, network, width, batch_size, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = width
        self.batch_size = batch_size

        # setup default hyper-parameter search ranges
        self.lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1, default_value=1e-2, log=True)
        self.momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.0, upper=0.99, default_value=0.5,
                                                       log=False)
        self.n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=40, default_value=2)

        self.weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-1, default_value=5e-4,
                                                           log=True)
        self.p_drop = CSH.UniformFloatHyperparameter('p_drop', lower=0.005, upper=0.5, default_value=0.2, log=True)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # setup dataloaders
        if torch.cuda.is_available():
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=True, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=True, num_workers=0)
        else:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                      pin_memory=False, num_workers=0)
            valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False,
                                                    pin_memory=False, num_workers=0)
        # setup model
        n_layers = config['n_layers']

        cuda = torch.cuda.is_available()

        model = dropout_regression_homo(input_dim=self.input_dim, output_dim=self.output_dim,
                                        width=self.width, n_layers=n_layers, p_drop=config['p_drop'])
        net = regression_baseline_net(model, self.N_train, lr=config['lr'], momentum=config['momentum'], cuda=cuda,
                                      schedule=None, MC_samples=20, weight_decay=config['weight_decay'])

        return train_loop(net, trainloader, valloader, budget, self.early_stop)

    def get_configspace(self):
        """
        It builds the configuration space with the needed hyperparameters.
        """
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([self.lr, self.momentum])
        cs.add_hyperparameters([self.n_layers])
        cs.add_hyperparameters([self.weight_decay, self.p_drop])
        return cs


def assign_model_class(model_name):
    if model_name == 'DUN_none':
        base_class = DUN_none_Worker
    elif model_name == 'DUN_wd':
        base_class = DUN_wd_Worker
    elif model_name == 'DUN_prior':
        base_class = DUN_prior_Worker
    elif model_name == 'Dropout':
        base_class = DropoutWorker
    elif model_name == 'MFVI':
        base_class = MFVIWorker
    elif model_name == 'SGD':
        base_class = SGDWorker
    else:
        raise Exception('model name not recognised')
    return base_class


def create_SpiralsWorker(model, network, width, batch_size):

    base_class = assign_model_class(model)

    class SpiralsWorker(base_class):
        def __init__(self, *args, early_stop=None, **kwargs):
            super().__init__(*args, network=network, width=width, batch_size=batch_size, **kwargs)

            # setup dataset
            X, y = gen_spirals(n_samples=2000, shuffle=True, noise=0.2, random_state=1234,
                               n_arms=2, start_angle=0, stop_angle=720)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1234)

            x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)

            X_train = ((X_train - x_means) / x_stds).astype(np.float32)
            X_test = ((X_test - x_means) / x_stds).astype(np.float32)

            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)

            self.trainset = Datafeed(X_train, y_train, transform=None)
            self.valset = Datafeed(X_test, y_test, transform=None)

            self.N_train = X_train.shape[0]
            self.input_dim = 2
            self.output_dim = 2
            self.early_stop = early_stop

            self.regression = False

    return SpiralsWorker


def create_FlightWorker(model, network, width, batch_size):

    base_class = assign_model_class(model)

    class FlightWorker(base_class):
        def __init__(self, *args, base_dir='nb_dir/data/', prop_val=0.05, k800=False, early_stop=None, **kwargs):
            super().__init__(*args, network=network, width=width, batch_size=batch_size, **kwargs)

            # setup dataset
            X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds = load_flight(base_dir, k800=k800)

            X_train = (X_train * x_stds) + x_means
            y_train = (y_train * y_stds) + y_means

            # print(X_train.shape)

            Ntrain = int(X_train.shape[0] * (1-prop_val))
            X_val = X_train[Ntrain:]
            y_val = y_train[Ntrain:]
            X_train = X_train[:Ntrain]
            y_train = y_train[:Ntrain]

            # print(X_train.shape)

            x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
            y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

            x_stds[x_stds < 1e-10] = 1.

            X_train = ((X_train - x_means) / x_stds)
            y_train = ((y_train - y_means) / y_stds)

            X_val = ((X_val - x_means) / x_stds)
            y_val = ((y_val - y_means) / y_stds)

            self.trainset = Datafeed(X_train, y_train, transform=None)
            self.valset = Datafeed(X_val, y_val, transform=None)

            self.N_train = X_train.shape[0]
            self.input_dim = X_train.shape[1]
            self.output_dim = y_train.shape[1]
            self.early_stop = early_stop

            self.regression = True

    return FlightWorker


def create_UCIWorker(model, network, width, batch_size):

    base_class = assign_model_class(model)

    class UCI_worker(base_class):
        def __init__(self, dname, *args, base_dir='nb_dir/data/', prop_val=0.15, n_split=0, early_stop=None, **kwargs):
            super().__init__(*args, network=network, width=width, batch_size=batch_size, **kwargs)

            gap = False
            if dname in ['boston', 'concrete', 'energy', 'power', 'wine', 'yacht', 'kin8nm', 'naval', 'protein']:
                pass
            elif dname in ['boston_gap', 'concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                           'kin8nm_gap', 'naval_gap', 'protein_gap']:
                gap = True
                dname = dname[:-4]

            X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds = \
                load_gap_UCI(base_dir=base_dir, dname=dname, n_split=n_split, gap=gap)

            X_train = (X_train * x_stds) + x_means
            y_train = (y_train * y_stds) + y_means

            # print(X_train.shape)
            Ntrain = int(X_train.shape[0] * (1-prop_val))
            X_val = X_train[Ntrain:]
            y_val = y_train[Ntrain:]
            X_train = X_train[:Ntrain]
            y_train = y_train[:Ntrain]

            # print(X_train.shape)
            x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
            y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

            x_stds[x_stds < 1e-10] = 1.

            X_train = ((X_train - x_means) / x_stds)
            y_train = ((y_train - y_means) / y_stds)

            X_val = ((X_val - x_means) / x_stds)
            y_val = ((y_val - y_means) / y_stds)

            self.trainset = Datafeed(X_train, y_train, transform=None)
            self.valset = Datafeed(X_val, y_val, transform=None)

            self.N_train = X_train.shape[0]
            self.input_dim = X_train.shape[1]
            self.output_dim = y_train.shape[1]
            self.early_stop = early_stop

            self.regression = True

    return UCI_worker


def train_loop(net, trainloader, valloader, budget, early_stop=None):
    # train for some budget
    train_NLLs = []
    train_errs = []
    MLL_ests = []
    valid_NLLs = []
    valid_errs = []
    prev_best_epoch = 0
    prev_best_NLL = np.inf
    for i in range(int(budget)):
        print('it %d / %d' % (i, budget))
        nb_samples = 0
        MLL_est = 0
        train_err = 0
        train_NLL = 0
        for x, y in trainloader:
            MLL, NLL, err = net.fit(x, y)

            train_NLL += NLL * x.shape[0]
            train_err += err * x.shape[0]
            MLL_est += MLL
            nb_samples += len(x)

        train_NLL /= nb_samples
        train_err /= nb_samples
        MLL_est /= nb_samples

        train_NLLs.append(train_NLL)
        train_errs.append(train_err)
        MLL_ests.append(MLL)

        net.update_lr()

        # eval on validation set
        nb_samples = 0
        valid_NLL = 0
        valid_err = 0
        for x, y in valloader:
            NLL, err = net.eval(x, y)

            valid_NLL += NLL * x.shape[0]
            valid_err += err * x.shape[0]
            nb_samples += len(x)

        valid_NLL /= nb_samples
        valid_err /= nb_samples
        valid_NLLs.append(valid_NLL)
        if valid_NLLs[-1] == np.nan:
            valid_NLLs[-1] = np.inf
        valid_errs.append(valid_err)

        if i > 0 and np.isnan(valid_NLL):  # Dont finish runs that NaN
            print('STOPPING DUE TO NAN')
            break
        # update vars every iteration to enable early stopping
        if valid_NLL < prev_best_NLL:
            prev_best_NLL = valid_NLL
            prev_best_epoch = i
        if early_stop is not None and (i - prev_best_epoch) > early_stop:
            print('EARLY STOPPING due to no improvement for %d epochs' % early_stop)
            break

    best_itr = np.argmin(valid_NLLs)
    print('best_itr: %d' % best_itr)
    return ({
        'loss': valid_NLLs[best_itr],  # remember: HpBandSter always minimizes!
        'info': {
            'best iteration': float(best_itr),
            'train err': train_errs[best_itr],
            'valid err': valid_errs[best_itr],
            'train NLL': train_NLLs[best_itr],
            'valid NLL': valid_NLLs[best_itr],
            'MLL est': MLL_ests[best_itr],
            'number of parameters': net.get_nb_parameters(),
        }
    })
