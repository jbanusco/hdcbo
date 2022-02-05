import torch
from hdcob.config import *
import numpy as np
import pandas as pd


class ListMetric(object):
    """Store the value of a metric at each iteration"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.x_value = []
        self.y_value = []
        self.rate = []

    def update(self, y_value, x_value=None):
        if x_value == None:
            try:
                x_value =+ self.x_value[-1]
            except IndexError:
                x_value = tensor([0])
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

    def has_converged(self, N=100, stop_slope=-0.001):
        L = len(self.y_value)
        M = int(N / 2)
        if L < (N * 2)+2:
            return False
        else:
            # slope of the loss function, moving averaged on a window of N epochs
            # rate = [np.polyfit(np.arange(N), self.y_value[i:i + N], deg=1)[0] for i in range(L - N)]
            # rate = pd.DataFrame(data=self.y_value).rolling(window=N).mean()[N - 1:].to_numpy()
            rate = pd.DataFrame(data=np.diff(self.y_value, n=1, axis=-1)).rolling(window=N).mean()[(L - (N+1)):].to_numpy()
            # return rate.mean() < -0.0001
            return False

            # oscillations = (np.array(rate) > 0).sum()
            # if oscillations > 1:
            #     pass
            # # print("{} oscillations (> 1)".format(str((np.array(rate) > 0).sum())))
            # if np.mean(rate[-M:-1]) > stop_slope:
            #     pass
            # # print("Sl. {}. Slope criteria reached (sl > {})".format(np.mean(rate[-100:-1]), stop_slope))
            # return oscillations > 1 or np.mean(rate[-M:-1]) > stop_slope


def mahalanobis_distance(x, y=None, cov=None):
    """
    Computes the mahalanobis distance between a vector x and y. If y is not specified, the mean of x is used
    The columns m are expected to be the features and the rows n the observations.
    :param x: dataframe of size (n x m)
    :param y: optional: dataframe of size (n x m)
    :param cov: optional: covaraince of the distirbution. If not given it will be computed using x (and y if given)
    :return: mahalanobis distance
    """
    distances = np.zeros((np.shape(x)[0], 1))

    if type(x) == np.ndarray:
        # Patch for the t-SNE calling
        # Convert it to dataframe
        sh = np.shape(x)
        ind_col = np.arange(0, sh[0])
        if len(sh) > 1:
            ind_idx = np.arange(0, sh[1])
            x = pd.DataFrame(data=x, index=list(ind_idx), columns=list(ind_col))
        else:
            x = pd.DataFrame(data=x[:, np.newaxis].T, index=[0], columns=list(ind_col))

    if y is None:
        # Remove NaN values [just in case]
        nan_ind = np.unique(np.where(x.isnull())[0])
        new_x = x.drop(nan_ind, axis=0).copy()
        new_y = new_x.mean(axis=0)  # Don't know why this is what actually gives the mean of the columns [features...]
        if cov is None:
            covariance = np.cov(new_x, rowvar=False)  # Shape: m x m
        else:
            covariance = cov
    else:
        # Distance between two vectors of the same distribution
        if type(y) == np.ndarray:
            # Patch for the t-SNE calling
            # Convert it to dataframe
            sh = np.shape(y)
            ind_col = np.arange(0, sh[0])
            if len(sh) > 1:
                ind_idx = np.arange(0, sh[1])
                y = pd.DataFrame(data=y, index=list(ind_idx), columns=list(ind_col))
            else:
                y = pd.DataFrame(data=y[:, np.newaxis].T, index=[0], columns=list(ind_col))

        nan_ind_x = np.unique(np.where(x.isnull())[0])
        nan_ind_y = np.unique(np.where(y.isnull())[0])
        nan_ind = np.unique(np.concatenate((nan_ind_x, nan_ind_y)))
        new_x = x.drop(nan_ind, axis=0).copy()
        new_y = y.drop(nan_ind, axis=0).copy()

        if cov is None:
            covariance = np.cov(np.vstack([new_x, new_y]), rowvar=False)  # Shape: m x m
        else:
            covariance = cov

    difference = new_x - new_y  # Shape: n x m
    # covariance = np.cov(new_x, rowvar=False)  # Shape: m x m

    if not np.all(np.linalg.eigvals(covariance) > 0):
        print("ERROR! The covariance matrix is not positive semi-definite")
        # exit(0)

    # -- Test using the inverse of np.linalg.inv
    # cov_inv = np.linalg.inv(covariance)
    # difference_one = difference.loc[0].values[:, np.newaxis].T
    # np.einsum("ij,kj->ik", cov_inv, difference_one)  # Squared
    # dis = difference_one.dot(cov_inv.dot(difference_one.T))
    # np.einsum("ki,ij,kj->", difference_one, cov_inv, difference_one)  # Good! for one example

    # Compute the mahalanobis distance for every point
    # dis = np.einsum("ki,ij,kj->k", difference, cov_inv, difference)  # Good! first element equal as the previous one

    # -- Using the linear solver to get the inverse
    tmp_x = np.linalg.solve(covariance, difference.T)
    dis = np.einsum("ki,ik->k", difference, tmp_x)  # Good the results it the same as using the cov_inv in a stable case

    if any(dis < 0):
        print("ERROR! Something went wrong and the inner product of the distance is negative")
        # exit(0)

    mahalanobis_dist = np.sqrt(dis)

    # Save the distances
    distances[new_x.index] = mahalanobis_dist[:, np.newaxis]
    distances[nan_ind] = np.nan

    return distances


def mse(predicted, target):
    """ Mean squared error; expected [N_obs, N_feats] """
    # sum over data dimensions (n_feats); average over observations (N_obs)
    if type(predicted) == list or type(predicted) == tuple:
        predicted = tensor(predicted).squeeze()

    if len(predicted.shape) == 0: predicted = predicted.unsqueeze(0)
    if len(target.shape) == 0: target = target.unsqueeze(0)
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

    if len(predicted.shape) == 0: predicted = predicted.unsqueeze(0)
    if len(target.shape) == 0: target = target.unsqueeze(0)
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

