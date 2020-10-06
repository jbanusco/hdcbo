from typing import List
from hdcob.config import *

EPS_NOISE = torch.exp(tensor([-20]))


def kernel_rbf(x1: tensor,
               x2: tensor,
               sigma: tensor,
               lamda: tensor,
               noise: tensor = EPS_NOISE,
               add_noise: bool = True):
    """
    Computes the RBF kernel - Takes advantage of broadcasting to make computation faster [loop is in C]
    :param x1: Data observations [Ns, D]
    :param x2: Data observations 2 [Ns, D]
    :param sigma: Variance/span/amplitude of the function
    :param lamda: Lengthscale, determines how strong is the correlation between points "wiggling of the function"
    :param noise: by default very small value to avoid 0's in the diagonal [eps]
    :return: K: [Ns, Ns] covariance matrix
    """

    diff = (x1.unsqueeze(dim=1) - x2.unsqueeze(dim=0))
    diff = (diff**2).sum(dim=2)  # Get differences between subjects summed along the features dim.

    # print(torch.matrix_rank(diff, symmetric=True))  # Compute rank of the matrix
    U = torch.exp(sigma) * torch.exp(-diff / (torch.exp(lamda)))

    # Add eps in the diagonal
    if add_noise:
        num_samples_x1 = x1.shape[0]
        K = tensor(U + torch.eye(num_samples_x1).to(DEVICE) * (torch.exp(noise) + EPS_NOISE))
    else:
        K = tensor(U)

    return K


def kernel_linear(x1: tensor,
                  x2: tensor,
                  sigma: tensor,
                  center: tensor,
                  noise: tensor = EPS_NOISE,
                  add_noise: bool = True):

    num_samples_x1 = x1.shape[0]
    num_samples_x2 = x2.shape[0]
    num_features = x1.shape[1]

    x1_tmp = torch.zeros((num_samples_x1, num_features))
    x2_tmp = torch.zeros((num_samples_x2, num_features))
    for ix in range(num_features):
        x1_tmp[:, ix] = x1[:, ix] - center[ix]
        x2_tmp[:, ix] = x2[:, ix] - center[ix]
    summ_diffs = x1_tmp.mm(x2_tmp.transpose(1, 0))

    U = torch.exp(sigma) * summ_diffs

    if add_noise:
        K = U + torch.eye(num_samples_x1).to(DEVICE) * (torch.exp(noise) + EPS_NOISE)
    else:
        K = U

    return K


def kernel_mult_lamdas(x1: tensor,
                       x2: tensor,
                       sigma: tensor,
                       lamda: List,
                       noise: tensor = EPS_NOISE,
                       add_noise=True):

    num_samples_x1 = x1.shape[0]
    num_samples_x2 = x2.shape[0]
    num_features = x1.shape[1]

    sum_diffs = tensor(torch.zeros((num_samples_x1, num_samples_x2))).to(DEVICE)

    # lamdas = tensor([2 * torch.exp(lamda[ix]) ** 2 for ix in range(num_features)])
    # a = (x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2
    # b = a / lamdas
    # sum_diffs = b.sum(dim=-1).to(DEVICE)

    for ix in range(num_features):
        diff = (x1[:, ix].unsqueeze(dim=1) - x2[:, ix].unsqueeze(dim=0)) ** 2
        res = -diff / (torch.exp(lamda[ix]))
        sum_diffs += res

    U = torch.exp(sigma) * torch.exp(sum_diffs)

    # Add eps in the diagonal
    if add_noise:
        K = U + torch.eye(num_samples_x1).to(DEVICE) * (torch.exp(noise) + EPS_NOISE)
    else:
        K = U

    return K


def kernel_mult_noise(x1: tensor,
                      x2: tensor,
                      sigma: tensor,
                      lamda: tensor,
                      noise: List,
                      add_noise: bool=True):

    num_samples_x1 = x1.shape[0]
    num_samples_x2 = x2.shape[0]
    num_features = x1.shape[1]

    K = tensor(torch.zeros((num_samples_x1, num_samples_x2))).to(DEVICE)
    for ix in range(num_features):
        Kf = kernel_rbf(x1[:, ix].unsqueeze(1), x2[:, ix].unsqueeze(1), sigma, lamda, noise=noise[ix],
                        add_noise=add_noise)
        K = K + Kf

    return K


def kernel_rbf_mult_params(x1: tensor,
                           x2: tensor,
                           sigma: tensor,
                           lamda: List,
                           noise: List = None,
                           add_noise: bool = True):
    """
    Computes the RBF kernel - Takes advantage of broadcasting to make computation faster [loop is in C]
    :param x1: Data observations [Ns, D]
    :param x2: Data observations 2 [Ns, D]
    :param sigma: Variance/span/amplitude of the function
    :param lamda: Lengthscale, determines how strong is the correlation between points "wiggling of the function"
    :param eps: avoid 0's in the diagonal
    :return: K: [Ns, Ns] covariance matrix
    """

    num_samples_x1 = x1.shape[0]
    num_samples_x2 = x2.shape[0]
    num_features = x1.shape[1]

    K = tensor(torch.zeros((num_samples_x1, num_samples_x2)))

    for ix in range(num_features):
        Kf = kernel_rbf(x1[:, ix].unsqueeze(1), x2[:, ix].unsqueeze(1), sigma, lamda[ix],
                        noise=noise[ix], add_noise=add_noise)
        K = K + Kf

    return K
