import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from src.utils import BaseNet, cprint, to_variable
from src.utils import rms
from src.probability import homo_Gauss_mloglike, depth_categorical


class DUN(BaseNet):
    def __init__(self, model, prob_model, N_train, lr=1e-2, momentum=0.5, weight_decay=0,
                 cuda=True, schedule=None, regression=False, pred_sig=None):
        super(DUN, self).__init__()

        cprint('y', 'DUN learnt with marginal likelihood categorical output')
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = model
        self.prob_model = prob_model
        self.cuda = cuda
        self.regression = regression
        self.pred_sig = pred_sig
        if self.regression:
            self.f_neg_loglike = homo_Gauss_mloglike(self.model.output_dim, self.pred_sig)
            self.f_neg_loglike_test = self.f_neg_loglike
        else:
            self.f_neg_loglike = nn.CrossEntropyLoss(reduction='none')  # This one takes logits
            self.f_neg_loglike_test = nn.NLLLoss(reduction='none')  # This one takes log probs

        self.N_train = N_train
        self.create_net()
        self.create_opt()
        self.schedule = schedule
        if self.schedule is not None and len(self.schedule) > 0:
            self.make_scheduler(gamma=0.1, milestones=self.schedule)
        self.epoch = 0

    def create_net(self):
        if self.cuda:
            self.model.cuda()
            self.f_neg_loglike.cuda()

            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        param_list = list(self.model.parameters()) + list(self.prob_model.parameters())
        if self.regression and self.pred_sig is None:
            param_list += list(self.f_neg_loglike.parameters())
        self.optimizer = torch.optim.SGD(param_list, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def fit(self, x, y):
        """Optimise stchastically estimated marginal joint of parameters and weights"""
        self.model.train()
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        if not self.regression:
            y = y.long()
        self.optimizer.zero_grad()

        act_vec = self.model.forward(x)
        prior_loglikes = self.model.get_w_prior_loglike(k=None)

        joint_loglike_per_depth = self.prob_model.get_w_joint_loglike(prior_loglikes, act_vec,
                                                                      y, self.f_neg_loglike, self.N_train)  # size(depth)
        log_marginal_over_depth = self.prob_model.get_marg_loglike(joint_loglike_per_depth)
        loss = -log_marginal_over_depth / self.N_train
        loss.backward()
        self.optimizer.step()

        # Note this posterior is 1 it behind the parameter settings as it is estimated with acts before optim step
        log_depth_posteriors = self.prob_model.get_depth_log_posterior(joint_loglike_per_depth, log_marginal_over_depth)
        self.prob_model.current_posterior = log_depth_posteriors.exp()

        if self.regression:
            means, model_stds = depth_categorical.marginalise_d_predict(
                act_vec.data, self.prob_model.current_posterior,
                depth=None, softmax=(not self.regression), get_std=True)
            mean_pred_negloglike = self.f_neg_loglike(means, y, model_std=model_stds).mean(dim=0).data
            err = rms(means, y).item()
        else:
            probs = depth_categorical.marginalise_d_predict(act_vec.data, self.prob_model.current_posterior,
                                                            depth=None, softmax=(not self.regression))
            mean_pred_negloglike = self.f_neg_loglike_test(torch.log(probs), y).mean(dim=0).data
            pred = probs.max(dim=1, keepdim=False)[1]  # get the index of the max probability
            err = pred.ne(y.data).sum().item() / y.shape[0]

        return log_marginal_over_depth.data.item(), mean_pred_negloglike.item(), err

    def eval(self, x, y):
        # TODO: make computationally stable with logsoftmax and nll loss -> would require making a log prediction method
        self.model.eval()
        with torch.no_grad():
            x, y = to_variable(var=(x, y), cuda=self.cuda)
            if not self.regression:
                y = y.long()

            act_vec = self.model.forward(x)

            if self.regression:
                means, model_stds = depth_categorical.marginalise_d_predict(act_vec.data,
                                                                            self.prob_model.current_posterior,
                                                                            depth=None, softmax=(not self.regression),
                                                                            get_std=True)
                mean_pred_negloglike = self.f_neg_loglike(means, y, model_std=model_stds).mean(dim=0).data
                err = rms(means, y).item()
            else:
                probs = depth_categorical.marginalise_d_predict(act_vec.data, self.prob_model.current_posterior,
                                                                depth=None, softmax=(not self.regression))
                mean_pred_negloglike = self.f_neg_loglike_test(torch.log(probs), y).mean(dim=0).data
                pred = probs.max(dim=1, keepdim=False)[1]  # get the index of the max probability
                err = pred.ne(y.data).sum().item() / y.shape[0]

            return mean_pred_negloglike.item(), err

    def layer_predict(self, x):
        self.model.eval()
        x, = to_variable(var=(x, ), cuda=self.cuda)
        out = self.model.forward(x).data
        if not self.regression:
            out = F.softmax(out, dim=2)
        return out

    def vec_predict(self, x, bin_mat):
        """Get predictions for specific binary vector configurations"""
        self.model.eval()
        x, = to_variable(var=(x, ), cuda=self.cuda)
        out = x.data.new(bin_mat.shape[0], x.shape[0], self.model.output_dim)
        for s in range(bin_mat.shape[0]):
            out[s] = self.model.vec_forward(x, bin_mat[s,:]).data
        if not self.regression:
            probs = F.softmax(out, dim=2)
        return probs.data

    def predict(self, x, depth=None, get_std=False, return_model_std=False):
        self.model.eval()
        with torch.no_grad():
            x, = to_variable(var=(x,), cuda=self.cuda)
            # if depth is None:
            #     depth = self.model.n_layers
            act_vec = self.model.forward(x, depth=depth).data

            if get_std:
                pred_mu, model_std = depth_categorical.marginalise_d_predict(act_vec, self.prob_model.current_posterior,
                                                                             depth=depth,
                                                                             softmax=(not self.regression),
                                                                             get_std=get_std)

                if return_model_std:
                    return pred_mu.data, model_std.data
                else:
                    pred_std = (model_std ** 2 + self.f_neg_loglike.log_std.exp() ** 2).pow(0.5)
                    return pred_mu.data, pred_std.data
            else:
                probs = depth_categorical.marginalise_d_predict(act_vec, self.prob_model.current_posterior, depth=depth,
                                                                softmax=(not self.regression), get_std=get_std)
                return probs.data

    def fast_predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x, = to_variable(var=(x,), cuda=self.cuda)

            act_vec = self.model.fast_forward_impl(x, self.prob_model.current_posterior, min_prob=1e-2).data

            probs = depth_categorical.marginalise_d_predict(act_vec, self.prob_model.current_posterior, depth=None,
                                                            softmax=True, get_std=False)
            return probs.data

    def get_exact_d_posterior(self, trainloader, train_bn=False, logposterior=False):
        """Get exact posterior over depth and log marginal over weights with full forward pass"""
        if train_bn:
            self.model.train()
        else:
            self.model.eval()
        with torch.no_grad():
            prior_loglikes = self.model.get_w_prior_loglike(k=None)

            N_train = len(trainloader.dataset)
            assert N_train == self.N_train
            cum_joint_loglike_per_depth = []

            for x, y in trainloader:
                x, y = to_variable(var=(x, y), cuda=self.cuda)
                if not self.regression:
                    y = y.long()
                act_vec = self.model.forward(x)

                joint_loglike_per_depth = self.prob_model.get_w_joint_loglike(prior_loglikes, act_vec,
                                                                              y, self.f_neg_loglike,
                                                                              N_train)  # size(depth)
                cum_joint_loglike_per_depth.append((x.shape[0] / N_train) * joint_loglike_per_depth.data.unsqueeze(0))

            cum_joint_loglike_per_depth = torch.cat(cum_joint_loglike_per_depth, dim=0).sum(dim=0)
            log_marginal_over_depth = self.prob_model.get_marg_loglike(cum_joint_loglike_per_depth)
            log_depth_posteriors = self.prob_model.get_depth_log_posterior(cum_joint_loglike_per_depth,
                                                                           log_marginal_over_depth)
            if logposterior:
                exact_posterior = log_depth_posteriors
            else:
                exact_posterior = log_depth_posteriors.exp()
            return exact_posterior, log_marginal_over_depth.data.item()


class DUN_VI(DUN):
    def __init__(self, model, prob_model, N_train, lr=1e-2, momentum=0.5, weight_decay=0, cuda=True,
                 schedule=None, regression=False, pred_sig=None):
        super(DUN_VI, self).__init__(model, prob_model, N_train, lr, momentum, weight_decay, cuda,
                                     schedule, regression, pred_sig)

    def fit(self, x, y):
        """Optimise stchastically estimated marginal joint of parameters and weights"""
        self.set_mode_train(train=True)
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        if not self.regression:
            y = y.long()
        self.optimizer.zero_grad()

        act_vec = self.model.forward(x)

        prior_loglikes = self.model.get_w_prior_loglike(k=None)

        ELBO = self.prob_model.estimate_ELBO(prior_loglikes, act_vec, y, self.f_neg_loglike, self.N_train, Beta=1)

        loss = -ELBO / self.N_train
        loss.backward()
        self.optimizer.step()
        self.prob_model.current_posterior = self.prob_model.get_q_probs()

        if self.regression:
            means, model_stds = depth_categorical.marginalise_d_predict(act_vec.data, self.prob_model.current_posterior,
                                                                        depth=None, softmax=(not self.regression),
                                                                        get_std=True)
            mean_pred_negloglike = self.f_neg_loglike(means, y, model_std=model_stds).mean(dim=0).data
            err = rms(means, y).item()
        else:
            probs = depth_categorical.marginalise_d_predict(act_vec.data, self.prob_model.current_posterior,
                                                            depth=None, softmax=(not self.regression))
            mean_pred_negloglike = self.f_neg_loglike_test(torch.log(probs), y).mean(dim=0).data
            pred = probs.max(dim=1, keepdim=False)[1]  # get the index of the max probability
            err = pred.ne(y.data).sum().item() / y.shape[0]

        # print(ELBO.shape, mean_pred_loglike.shape, err.shape)
        return ELBO.data.item(), mean_pred_negloglike.item(), err

    def get_exact_ELBO(self, trainloader, train_bn=False):
        """Get exact ELBO with full forward pass"""
        if train_bn:
            self.model.train()
        else:
            self.model.eval()
        with torch.no_grad():

            prior_loglikes = self.model.get_w_prior_loglike(k=None)

            N_train = len(trainloader.dataset)
            assert N_train == self.N_train
            cum_ELBO = []

            for x, y in trainloader:
                x, y = to_variable(var=(x, y), cuda=self.cuda)
                if not self.regression:
                    y = y.long()
                act_vec = self.model.forward(x)

                ELBO = self.prob_model.estimate_ELBO(prior_loglikes, act_vec, y, self.f_neg_loglike, N_train, Beta=1)
                cum_ELBO.append((x.shape[0] / N_train) * ELBO.data.unsqueeze(0))

            cum_ELBO = torch.cat(cum_ELBO, dim=0).sum(dim=0)
            return cum_ELBO.data.item()
