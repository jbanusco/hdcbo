import torch
from hdcob.config import *
import numpy as np


class ListMetric(object):
    """Store the value of a metric at each iteration"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.x_value = []
        self.y_value = []
        self.rate = []

    def update(self, x_value, y_value):
        self.x_value = np.append(self.x_value, x_value)
        self.y_value = np.append(self.y_value, y_value)

    def check_rate(self):
        try:
            # self.rate = np.diff(self.y_value, n=1, axis=-1)[-1]
            self.rate = np.sum(np.diff(self.y_value, n=1, axis=-1)[-10:])
        except:
            self.rate = np.inf

    def check_mean(self, n=1):
        # n: num of samples
        if n < len(self.y_value):
            return np.mean(self.y_value[-n:])
        else:
            return np.mean(self.y_value[:])


def mse(predicted, target):
    """ Mean squared error; expected [N_obs, N_feats] """
    # sum over data dimensions (n_feats); average over observations (N_obs)
    if type(predicted) == list or type(predicted) == tuple:
        predicted = tensor(predicted).squeeze()

    if len(predicted.shape) == 1: predicted = predicted.unsqueeze(1)
    if len(target.shape) == 1: target = target.unsqueeze(1)

    if predicted.size()[-1] != target.size()[-1]: predicted = predicted.T

    return ((target - predicted) ** 2).sum(1).mean(0)  # torch.Size([1])


def mae(predicted, target):
    """
    Mean absolute error; expected [N_obs, N_feats]
    """
    # sum over data dimensions (n_feats); average over observations (N_obs)
    if type(predicted) == list or type(predicted) == tuple:
        predicted = tensor(predicted).squeeze()

    if len(predicted.shape) == 1: predicted = predicted.unsqueeze(1)
    if len(target.shape) == 1: target = target.unsqueeze(1)

    if predicted.size()[-1] != target.size()[-1]: predicted = predicted.T

    return torch.abs(target - predicted).sum(1).mean(0)  # torch.Size([1]


def mse_dim(predicted, target):
    """ Mean squared error for each dimension """
    return ((target - predicted) ** 2).mean(0)


def mae_dim(predicted, target):
    """ Mean absolute error for each dimension """
    return torch.abs(target - predicted).mean(0)

