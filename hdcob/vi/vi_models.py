from hdcob.vi.encoders_decoders import *
from hdcob.config import *
from torch.distributions import Normal, kl_divergence
from torch import nn


def kl_gaussian(mu, logvar):
    """ KL between two gaussians """
    Q = Normal(mu, torch.exp(logvar * 0.5))
    P = Normal(0, 1)
    kl = kl_divergence(Q, P)
    kl = kl.sum(1).mean(0)
    return kl


def gaussian_log_likelihood(target, mean, log_sigma):
    """ Log-likelihood of a univariate gaussian """
    ll = -0.5 * (LOG_2_PI + log_sigma + ((target - mean) ** 2) / torch.exp(log_sigma))
    ll = ll.sum(1).mean(0)

    return ll


class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 1,
                 hidden_dim: int = 0,
                 bias: bool = False,
                 init_noise: float = -5,
                 prior_noise: dict = None,
                 param: bool = True):
        """
        We expect data standardized [0 mean, unit variance]
        :param input_dim: Input dimensions
        :param latent_dim: Latent dimensions
        :param hidden_dim: Hidden dimensions
        :param bias: Learn bias or not
        :param param: Use a parameter for the variance of the decoder or not
        """
        super(VAE, self).__init__()

        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim

        # Encoder [Two returns one for the mean and another for the variance, in log-space]
        if self._hidden_dim == 0:
            self._encoder = EncoderGaussian(self._input_dim, self._latent_dim, bias=bias)
        else:
            self._encoder = EncoderGaussianHidden(self._input_dim, self._hidden_dim, self._latent_dim, bias=bias)

        # Decoder [One of the rec/observed mean and another for the variance, in log-space]
        if self._hidden_dim == 0:
            if param:
                if prior_noise is None:
                    self._decoder = DecoderGaussian_ParamV(self._latent_dim,
                                                           self._input_dim, bias=bias,
                                                           init_value=init_noise)
                else:
                    self._decoder = DecoderGaussian_ParamV_Prior(self._latent_dim,
                                                                 self._input_dim, bias=bias,
                                                                 init_value=init_noise,
                                                                 prior_noise=prior_noise)

            else:
                self._decoder = DecoderGaussian(self._latent_dim, self._input_dim, bias=bias)
        else:
            if param:
                self._decoder = DecoderGaussian_HiddenParamVar(self._latent_dim, self._hidden_dim, self._input_dim,
                                                               bias=bias, init_value=init_noise)
            else:
                self._decoder = DecoderGaussianHidden(self._latent_dim, self._hidden_dim, self._input_dim, bias=bias)

        # Variables to store distribution parameters
        self._lat_mu = tensor()
        self._lat_logvar = tensor()

        self._x_mu = tensor()
        self._x_logvar = tensor()

    def save_training_info(self):
        """ Generate dictionary with the training information """
        training_info = {
            "latent_mu": self._lat_mu,
            "latent_logvar": self._lat_logvar,
            "x_mu": self._x_mu,
            "x_logvar": self._x_logvar
        }
        return training_info

    def load_training_info(self,
                           training_info: dict):
        """ Load training information """
        self._lat_mu = training_info["latent_mu"]
        self._lat_logvar = training_info["latent_logvar"]
        self._x_mu = training_info["x_mu"]
        self._x_logvar = training_info["x_logvar"]

    def loss(self,
             predicted: tensor,
             target: tensor):

        """ Compute the loss """
        # Latent distribution
        try:
            mean_latent = predicted[0]
            logvar_latent = predicted[1]

            # Reconstructed distribution
            mean_rec = predicted[2]
            noise_rec = predicted[3]

            # Compute likelihood
            ll = gaussian_log_likelihood(target, mean_rec, noise_rec)

            kl = kl_gaussian(mean_latent, logvar_latent)
            if torch.isinf(kl):
                log.debug("KL is infinite")

            if hasattr(self._decoder, 'prior_noise'):
                kl = kl + self._decoder.kl().mean()

            loss = -ll + kl

            loss_dict = dict(total=loss,
                             ll=-ll,
                             kl=kl
                             )

            return loss_dict
        except Warning:
            print("Here")

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                x: tensor):
        """
        Forward pass of the model. Encodes information present in X into the latent space Z. Imputes data from it,
        and predicts Y using both
        :param x: [Ns, in_dim]
        """
        mu, logvar = self._encoder(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
            self._lat_mu = mu.detach()
            self._lat_logvar = logvar.detach()
        else:
            # Sample from the prior distribution
            if self._lat_mu.size(0) > 0:
                avg_std_per_subject = torch.exp(0.5 * self._lat_logvar.mean(dim=0))
                avg_subject = self._lat_mu.mean(dim=0)
                z = avg_subject + avg_std_per_subject * torch.randn((x.size(0), self._latent_dim))
                z = avg_subject * torch.ones(((x.size(0), self._latent_dim)))
            else:
                z = torch.zeros(((x.size(0), self._latent_dim)))

        x_mu, x_logvar = self._decoder(z)
        if self.training:
            self._x_mu = x_mu.detach()
            self._x_logvar = x_logvar.detach()

        return mu, logvar, x_mu, x_logvar


class CVAE(VAE):
    """ Conditional Variational Autoencoder """

    def __init__(self,
                 input_dim: int,
                 cond_dim: int,
                 latent_dim: int = 1,
                 hidden_dim: int = 0,
                 bias: bool = False,
                 init_noise: float = -5,
                 prior_noise: dict = None,
                 param: bool = True):
        """
        We expect data standardized [0 mean, unit variance]
        :param input_dim: Input dimensions
        :param latent_dim: Latent dimensions
        """
        super(CVAE, self).__init__(input_dim=input_dim+cond_dim,
                                   latent_dim=latent_dim,
                                   hidden_dim=hidden_dim,
                                   bias=bias,
                                   param=param)

        self._input_dim = input_dim
        self._cond_dim = cond_dim

        if self._hidden_dim == 0:
            if param:
                if prior_noise is None:
                    self._decoder = DecoderGaussian_ParamV(self._latent_dim+self._cond_dim,
                                                           self._input_dim, bias=bias,
                                                           init_value=init_noise)
                else:
                    self._decoder = DecoderGaussian_ParamV_Prior(self._latent_dim+self._cond_dim,
                                                                 self._input_dim, bias=bias,
                                                                 init_value=init_noise,
                                                                 prior_noise=prior_noise)
            else:
                self._decoder = DecoderGaussian(self._latent_dim+self._cond_dim, self._input_dim, bias=bias)
        else:
            if param:
                self._decoder = DecoderGaussian_HiddenParamVar(self._latent_dim+self._cond_dim, self._hidden_dim,
                                                               self._input_dim, init_value=init_noise, bias=bias)
            else:
                self._decoder = DecoderGaussianHidden(self._latent_dim+self._cond_dim, self._hidden_dim,
                                                      self._input_dim, bias=bias)

    def forward(self,
                x: tensor,
                cond: tensor):
        """
        Forward pass of the model. Encodes information present in X into the latent space Z. Imputes data from it,
        and predicts Y using both
        :param x: [Ns, in_dim]
        """

        input_data = torch.cat((x, cond), dim=1)
        mu, logvar = self._encoder(input_data)
        if self.training:
            z = self.reparameterize(mu, logvar)
            self._lat_mu = mu.detach()
            self._lat_logvar = logvar.detach()
        else:
            # Sample from the prior distribution
            if self._lat_mu.size(0) > 0:
                avg_std_per_subject = torch.exp(0.5 * self._lat_logvar.mean(dim=0))
                avg_subject = self._lat_mu.mean(dim=0)
                z = avg_subject + avg_std_per_subject * torch.randn((cond.size(0), self._latent_dim))
                z = avg_subject * torch.ones(((cond.size(0), self._latent_dim)))
            else:
                # Otherwise when calculating the graph (for tensorbaord) it crashes
                z = torch.zeros(((cond.size(0), self._latent_dim)))

        # Condition the latent space
        z = torch.cat((z, cond), dim=1)

        x_mu, x_logvar = self._decoder(z)
        if self.training:
            self._x_mu = x_mu.detach()
            self._x_logvar = x_logvar.detach()
        return mu, logvar, x_mu, x_logvar


class CVI(VAE):
    """ Conditional Variational Inference """

    def __init__(self,
                 input_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 latent_dim: int = 1,
                 hidden_dim: int = 0,
                 bias: bool = False,
                 init_noise: float = -5,
                 prior_noise: dict = None,
                 param: bool = True):
        """
        We expect data standardized [0 mean, unit variance]
        :param input_dim: Input dimensions
        :param latent_dim: Latent dimensions
        """
        super(CVI, self).__init__(input_dim=input_dim+cond_dim,
                                  latent_dim=latent_dim,
                                  hidden_dim=hidden_dim,
                                  bias=bias,
                                  param=param)

        self._input_dim = input_dim
        self._cond_dim = cond_dim
        self._out_dim = out_dim

        if self._hidden_dim == 0:
            if param:
                if prior_noise is None:
                    self._decoder = DecoderGaussian_ParamV(self._latent_dim+self._cond_dim,
                                                           self._out_dim, bias=bias,
                                                           init_value=init_noise)
                else:
                    self._decoder = DecoderGaussian_ParamV_Prior(self._latent_dim+self._cond_dim,
                                                                 self._out_dim, bias=bias,
                                                                 init_value=init_noise,
                                                                 prior_noise=prior_noise)
            else:
                self._decoder = DecoderGaussian(self._latent_dim+self._cond_dim, self._out_dim, bias=bias)
        else:
            if param:
                self._decoder = DecoderGaussian_HiddenParamVar(self._latent_dim+self._cond_dim, self._hidden_dim,
                                                               self._out_dim, init_value=init_noise, bias=bias)
            else:
                self._decoder = DecoderGaussianHidden(self._latent_dim+self._cond_dim, self._hidden_dim,
                                                      self._out_dim, bias=bias)

    def forward(self,
                x: tensor,
                cond: tensor):
        """
        Forward pass of the model. Encodes information present in X into the latent space Z. Imputes data from it,
        and predicts Y using both
        :param x: [Ns, in_dim]
        """

        input_data = torch.cat((x, cond), dim=1)
        mu, logvar = self._encoder(input_data)

        if self.training:
            z = self.reparameterize(mu, logvar)
            self._lat_mu = mu.detach()
            self._lat_logvar = logvar.detach()
        else:
            z = mu

        # Condition the latent space
        z = torch.cat((z, cond), dim=1)

        x_mu, x_logvar = self._decoder(z)
        if self.training:
            self._x_mu = x_mu.detach()
            self._x_logvar = x_logvar.detach()

        return mu, logvar, x_mu, x_logvar
