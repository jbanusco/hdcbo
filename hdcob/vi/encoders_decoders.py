from torch import nn
from torch.distributions import Normal, kl_divergence
from hdcob.config import *

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- ENCODERS -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class EncoderGaussian(nn.Module):
    """ Encoder without hidden layer """

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 bias: bool = False):
        super(EncoderGaussian, self).__init__()
        self.fc11 = nn.Linear(input_dim, out_dim, bias=bias)  # For the mean
        self.fc12 = nn.Linear(input_dim, out_dim, bias=bias)  # For the logvar

    def forward(self,
                x: tensor):
        return self.fc11(x), self.fc12(x)


class EncoderGaussian_ParamV(nn.Module):
    """ Encoder without hidden layer and variance as a parameter  """
    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 bias: bool = False,
                 init_value: float = -5.):
        super(EncoderGaussian_ParamV, self).__init__()
        self.fc11 = nn.Linear(input_dim, out_dim, bias=bias)
        self.param = nn.Parameter(tensor(1, out_dim).fill_(init_value), requires_grad=True)

    def forward(self,
                x: tensor):
        return self.fc11(x), self.param


class EncoderGaussianHidden(nn.Module):
    """ Encoder with a hidden layer """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 bias=False):
        super(EncoderGaussianHidden, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc21 = torch.nn.Linear(hidden_dim, out_dim, bias=bias)
        self.fc22 = torch.nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self,
                x: tensor):
        # h = torch.sigmoid(self.fc1(x))
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)


class EncoderGaussianHidden_ParamV(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 bias: bool = False,
                 init_value: float = -5.):
        super(EncoderGaussianHidden_ParamV, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc21 = torch.nn.Linear(hidden_dim, out_dim, bias=bias)
        self.param = nn.Parameter(tensor(1, out_dim).fill_(init_value), requires_grad=True)

    def forward(self,
                x: tensor):
        # h = torch.sigmoid(self.fc1(x))
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.param


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- DECODERS ----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class DecoderGaussian(EncoderGaussian):
    """ Same structure as encoder without hidden layer """
    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 bias: bool = False):
        super(DecoderGaussian, self).__init__(input_dim, out_dim, bias)


class DecoderGaussian_ParamV(EncoderGaussian_ParamV):
    """ Same structure as encoder without hidden layer and parameter for the variance """
    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 bias: bool = False,
                 init_value: float = -5.):

        super(DecoderGaussian_ParamV, self).__init__(input_dim, out_dim, bias, init_value)


class DecoderGaussian_ParamV_Prior(EncoderGaussian_ParamV):
    """  Add a prior to the variance """

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 bias: bool = False,
                 init_value: float = -5.,
                 prior_noise: dict =None):
        super(DecoderGaussian_ParamV_Prior, self).__init__(input_dim, out_dim, bias, init_value)
        self.param_logvar = nn.Parameter(tensor(1, out_dim).fill_(-3), requires_grad=True)

        if prior_noise is None:
            raise ValueError(f"Need to specify the prior values")

        assert type(prior_noise) is dict

        self.prior_noise = prior_noise

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl(self):
        mean_p = self.prior_noise['mean']
        logvar_p = self.prior_noise['logvar']
        Q = Normal(self.param, torch.exp(self.param_logvar * 0.5))
        P = Normal(mean_p, torch.exp(logvar_p * 0.5))
        kl = kl_divergence(Q, P)

        return kl

    def forward(self,
                x: tensor):
        param = self.reparameterize(self.param, self.param_logvar)  # Sample
        return self.fc11(x), param


class DecoderGaussianHidden(EncoderGaussianHidden):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 bias=False):
        """ Same structure as encoder with hidden layer """
        super(DecoderGaussianHidden, self).__init__(input_dim, hidden_dim, out_dim, bias)


class DecoderGaussian_HiddenParamVar(EncoderGaussianHidden_ParamV):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 bias: bool = False,
                 init_value: float = -5.):
        """ Same structure as encoder with hidden layer and param for the variance """
        super(DecoderGaussian_HiddenParamVar, self).__init__(input_dim, hidden_dim, out_dim, bias, init_value)



class DecoderIterative(nn.Module):
    """  """
    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 bias=False,
                 init_value: float = -5.):
        super(DecoderIterative, self).__init__()
        self._out_dim = out_dim
        self.linears = nn.ModuleList([nn.Linear(input_dim+(i-1), 1,
                                                bias=bias) for i in range(1, out_dim+1)])
        self.param = nn.Parameter(tensor(1, out_dim).fill_(init_value), requires_grad=True)

    def forward(self,
                x: tensor):

        x_out = torch.zeros((x.shape[0], self._out_dim))
        for i, l in enumerate(self.linears):
            x_out[:, i] = (l(torch.cat((x, x_out[:, :i]), dim=1))).squeeze()

        return x_out, self.param


