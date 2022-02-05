from hdcob.vi.encoders_decoders import *
from hdcob.config import *
from hdcob.gp.kernels import kernel_mult_lamdas
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
    # cov_data = tensor(np.cov(target.data.T))
    # cov_data = mean.T.mm(mean) / mean.shape[0] + torch.diag(torch.exp(log_sigma.mean(dim=0)))
    # distr = torch.distributions.MultivariateNormal(torch.zeros_like(mean), cov_data)
    # distr = torch.distributions.MultivariateNormal(mean, log_sigma)

    distr = torch.distributions.MultivariateNormal(mean.view(-1), log_sigma)
    ll = distr.log_prob(target.view(-1)) / target.shape[0]

    # # ll = distr.log_prob(target).mean(0)
    # ll = -0.5 * (LOG_2_PI + log_sigma + ((target - mean) ** 2) / torch.exp(log_sigma))
    # ll = ll.sum(1).mean(0)

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
    """ Variational Autoencoder """

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

        # self._decoder = DecoderIterative(self._latent_dim + self._cond_dim,
        #                                  self._input_dim, bias=bias,
        #                                  init_value=init_noise)

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

        # self.L = nn.Linear(self._latent_dim + self._cond_dim, self._input_dim * 2, bias=bias)
        # self.L = nn.Parameter(torch.eye(self._input_dim), requires_grad=True)

        # We need a lower triangular matrix
        num_params = int((self._input_dim * (self._input_dim - 1) / 2) + self._input_dim)
        x = torch.rand(num_params)  # Uniform sampling [0, 1] -> positive
        self.L = nn.Parameter(x, requires_grad=True)

    @staticmethod
    def kronecker(A, B):
        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))

    def forward(self,
                x: tensor,
                cond: tensor,
                num_samples=0):
        """
        Forward pass of the model. Encodes information present in X into the latent space Z. Imputes data from it,
        and predicts Y using both
        :param x: [Ns, in_dim]
        """

        if not self.training:
            x = torch.zeros_like(x)

        input_data = torch.cat((x, cond), dim=1)
        mu, logvar = self._encoder(input_data)
        if self.training:
            z = self.reparameterize(mu, logvar)
            self._lat_mu = mu.detach()
            self._lat_logvar = logvar.detach()
        else:
            # Sample from the prior distribution
            # if self._lat_mu.size(0) > 0:
            #     avg_std_per_subject = torch.exp(0.5 * self._lat_logvar.mean(dim=0))
            #     avg_subject = self._lat_mu.mean(dim=0)
            #     z = avg_subject + avg_std_per_subject * torch.randn((cond.size(0), self._latent_dim))
            #     # z = torch.randn((cond.size(0), self._latent_dim))
            #     # z = avg_subject * torch.ones(((cond.size(0), self._latent_dim)))
            #     # z = avg_subject * torch.zeros(((cond.size(0), self._latent_dim)))
            # else:
            #     # Otherwise when calculating the graph (for tensorbaord) it crashes
            #     z = torch.zeros(((cond.size(0), self._latent_dim)))
            z = mu

        # Condition the latent space
        # if num_samples == 0:
        z = torch.cat((z, cond), dim=1)

        x_mu, x_logvar = self._decoder(z)
        # L = self.L(z)

        M = torch.zeros((self._input_dim, self._input_dim))
        tril_indices = torch.tril_indices(row=self._input_dim, col=self._input_dim, offset=0)
        M[tril_indices[0], tril_indices[1]] = self.L

        # Ensure the diagonal entries are positive
        # M[torch.eye(self._input_dim, dtype=bool)] = torch.abs(torch.diag(M))
        M[torch.eye(self._input_dim, dtype=bool)] = torch.exp(torch.diag(M))
        x_cov = self.kronecker(torch.eye(x.shape[0]), M.mm(M.T))  # + torch.diag(torch.exp(x_logvar.view(-1)))

        # x_mu = x_mu.view(-1)

        # x_cov = self.L.mm(self.L.T) + torch.eye(self._input_dim) * torch.exp(x_logvar)
        # x_logvar = self.kernel(x_mu, x_mu, self.sigma, self.lamda, noise=self.noise, add_noise=True)

        if self.training:
            self._x_mu = x_mu.detach()
            self._x_logvar = x_logvar.detach()
            self._x_cov = x_cov.detach()
        # return mu, logvar, x_mu, x_logvar
        return mu, logvar, x_mu, x_cov
        # else:
        #     x_mu_def = torch.zeros((x.shape[0], self._input_dim, num_samples))
        #     x_logvar_def = torch.zeros((x.shape[0], self._input_dim, num_samples))
        #     Z = tensor(np.repeat(np.linspace(-2.5, 2.5, num_samples)[:, np.newaxis], x.shape[0], 1).T)
        #     for ix in range(0, num_samples):
        #         z = torch.cat((Z[:, ix].unsqueeze(1), cond), dim=1)
        #         x_mu, x_logvar = self._decoder(z)
        #         x_mu_def[:, :, ix] = x_mu
        #         x_logvar_def[:, :, ix] = x_logvar
        #     return mu, logvar, x_mu_def, x_logvar_def


class ICVAE(nn.Module):
    """ Variational Autoencoder """

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
        super(ICVAE, self).__init__()

        self._input_dim = input_dim
        self._cond_dim = cond_dim
        self._latent_dim = latent_dim

        self.cvaes = nn.ModuleList(
            [CVAE(1,
                  self._cond_dim + (i - 1),
                  hidden_dim=hidden_dim,
                  latent_dim=self._latent_dim,
                  bias=bias,
                  init_noise=init_noise,
                  prior_noise=prior_noise,
                  param=param)
             for i in range(1, self._input_dim+1)]
        )

    def forward(self,
                x: tensor,
                cond: tensor):
        """
        Forward pass of the model. Encodes information present in X into the latent space Z. Imputes data from it,
        and predicts Y using both
        :param x: [Ns, in_dim]
        """

        # if x.shape[-1] == 0: x = x.unsqueeze(1)
            # x = torch.zeros((cond.shape[0], 1))

        if not self.training:
            x = torch.zeros_like(x)

        mean_lat = torch.zeros((x.shape[0], self._input_dim))
        lat_logvar = torch.zeros((x.shape[0], self._input_dim))
        # lat_logvar = torch.zeros((1, self._input_dim))

        x_logvars = torch.zeros((1, self._input_dim))
        x_out = torch.zeros((x.shape[0], self._input_dim))
        for i, l in enumerate(self.cvaes):
            cond_pass = torch.cat((cond, x_out[:, :i]), dim=1)
            out_data = l.forward(x[:, i].unsqueeze(1), cond_pass)

            mean_lat[:, i] = out_data[0].squeeze()
            lat_logvar[:, i] = out_data[1].squeeze()

            x_out[:, i] = (out_data[-2]).squeeze()
            x_logvars[0, i] = out_data[-1]

        return mean_lat, lat_logvar, x_out, x_logvars

    def loss(self, predicted, target):
        return self.cvaes[0].loss(predicted, target)
        # loss = tensor([0])
        # for ix in range(0, self._input_dim):
        #     loss += self.cvaes[ix].loss(predicted, target[:, ix])

    def save_training_info(self):
        """ Generate list of dictionaries with the training information """

        training_info = [self.cvaes[ix].save_training_info() for ix in range(self._input_dim)]
        return training_info

    def load_training_info(self,
                           training_info: list):
        """ Since for the prediction we need information regarding the training data we will need to load it.
        Not only the parameters of the model """

        for ix in range(self._input_dim):
            self.cvaes[ix].load_training_info(training_info[ix])

