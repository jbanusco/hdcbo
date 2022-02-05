from torch import nn
from hdcob.config import *
from hdcob.gp.gaussian_process import RegressionGP
from hdcob.vi.vi_models import CVAE, VAE, ICVAE
from hdcob.utilities.metrics import mse
import copy


class VIRCA(nn.Module):
    """ Variational imputation and regression framework based on conditional autoencoding """
    def __init__(self,
                 input_dim: int,
                 miss_dim: int,
                 cond_dim: int = 0,
                 latent_dim: int = 1,
                 hidden_dim: int = 0,
                 target_dim: int = 1,
                 init_sigma_gp: float = 0,
                 init_lamda_gp: float = 0,
                 init_mean_gp: float = 0,  # 0
                 init_noise_gp: float = -2,  # GP noise
                 init_noise_vi: float = -2,  # VI noise
                 bias: bool = False,
                 kernel: str = "RBF_ML",
                 prior_noise: dict = None,
                 use_param: bool = False,
                 reg: str = "GP",
                 device: str="cpu"):
        """
        We expect data standardized [0 mean, unit variance]
        :param input_dim: Fixed input data in the GP regression (observed data)\
        :param target_dim: Data to predict in the GP regression
        :param miss_dim: Missing input data in the GP regression (imputed by autoencoder)
        :param cond_dim: Data to condition the autoencoder
        :param latent_dim: Latent dimensions of the autoencoder
        :param hidden_dim: Hidden dimensions of the autoencoder
        :param init_sigma_gp: Initial sigma of the GP
        :param init_lamda_gp: Initial value of the lamda for the GP
        :param init_mean_gp: Initial value of the GP mean [should be always 0]
        :param init_noise_gp: Initial value of the GP noise
        :param init_noise_vi: Initial value of the GP in the VI in the case that we use a parameter for the decoder variance
        :param bias: learn bias or not
        :param kernel: Kernel of the GP (RBF_ML recommended)
        :param prior_noise: Prior in the VI decoder variance, in case that a parameter is used
        :param use_param: Use a parameter or not for the decoder variance
        """
        super(VIRCA, self).__init__()
        self.device = device
        # VI
        self._miss_dim = miss_dim
        self._cond_dim = cond_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim

        # GP
        self._input_dim = input_dim
        self._n_features = input_dim + miss_dim
        self._target_dim = target_dim

        # Conditioning or not
        if self._cond_dim > 0:
            self.condition = True
        else:
            self.condition = False
        self.condition = True

        # Initialize VAE
        if self.condition:
            # self._VI = CVAE(self._miss_dim, self._cond_dim, hidden_dim=hidden_dim,
            #                 latent_dim=self._latent_dim, bias=bias,
            #                 init_noise=init_noise_vi, prior_noise=prior_noise, param=use_param)
            self._VI = CVAE(self._miss_dim, self._cond_dim+self._input_dim, hidden_dim=hidden_dim,
                            latent_dim=self._latent_dim, bias=bias,
                            init_noise=init_noise_vi, prior_noise=prior_noise, param=use_param, device=self.device)
            # self._VI = ICVAE(self._miss_dim, self._cond_dim + self._input_dim, hidden_dim=hidden_dim,
            #                  latent_dim=self._latent_dim, bias=bias,
            #                  init_noise=init_noise_vi, prior_noise=prior_noise, param=use_param)
        else:
            self._VI = VAE(self._miss_dim, hidden_dim=hidden_dim, latent_dim=self._latent_dim,
                           bias=bias, init_noise=init_noise_vi, prior_noise=prior_noise,
                           param=use_param)

        # Initialize GP
        if reg == "GP":
            # self._Reg = RegressionGP(init_sigma=init_sigma_gp, init_lamda=init_lamda_gp, init_mean=init_mean_gp,
            #                          init_noise=init_noise_gp, input_dim=self._n_features, output_dim=self._target_dim,
            #                          kernel=kernel)

            self._Reg = RegressionGP(init_sigma=init_sigma_gp, init_lamda=init_lamda_gp, init_mean=init_mean_gp,
                                     init_noise=init_noise_gp, input_dim=self._n_features, output_dim=self._target_dim,
                                     kernel=kernel, device=self.device)
        else:
            self._Reg = nn.Linear(self._miss_dim + self._input_dim, self._target_dim, bias=bias)

        # Test decoder
        self._decoder = copy.deepcopy(self._VI._decoder)

    def save_training_info(self):
        """ Save training info """
        if type(self._Reg) == RegressionGP:
            gp_info = self._Reg.save_training_info()
        else:
            gp_info = []
        vi_info = self._VI.save_training_info()
        info = {'gp_info': gp_info,
                'vi_info': vi_info}
        return info

    def load_training_info(self,
                           training_info):
        """ Load training info """
        if type(self._Reg) == RegressionGP:
            self._Reg.load_training_info(training_info['gp_info'])
        self._VI.load_training_info(training_info['vi_info'])

    def add_prior_gp(self,
                     param_name,
                     prior):
        self._GP.add_prior(f"{param_name}", prior)

    def loss(self, rec_target, pred_target, outputs):
        """ Loss function """

        def_value = tensor([0]).squeeze()
        num_samples = pred_target.shape[0]
        num_pred = pred_target.shape[1]
        num_miss = rec_target.shape[1]

        loss_vi = self._VI.loss(outputs[:4], rec_target)
        if type(self._Reg) == RegressionGP:
            losses_gp = self._Reg.loss((outputs[4], outputs[5]), pred_target)
            # factor_gp = 1 / (num_samples * num_pred)
            # factor_gp = 1 / num_samples
            factor_gp = 1 / num_pred
            # factor_gp = 1
        else:
            total = mse(outputs[4], pred_target)
            losses_gp = dict(total=total,
                             ll_gp=total,
                             kl_gp=0)
            factor_gp = 1

        # We need to weight the number of features predicted and imputed, in order that the optimization is stable
        factor_vi = (1 / num_miss)
        # factor_vi = 1
        # factor_vi = 1 / num_samples
        total_vi = loss_vi.get('ll', def_value) * factor_vi + loss_vi.get('kl', def_value)
        total_gp = losses_gp.get('ll_gp', def_value) * factor_gp + losses_gp.get('kl_gp', def_value)

        total = total_vi + total_gp
        gp_ll = losses_gp['ll_gp']
        lamda_kl = losses_gp['kl_gp']

        loss_dict = dict(total=total,
                         total_gp=total_gp,
                         ll_gp=gp_ll * factor_gp,
                         kl_gp=lamda_kl,
                         total_vi=total_vi,
                         kl_vi=loss_vi.get('kl', def_value),
                         ll_vi=loss_vi.get('ll', def_value) * factor_vi)

        return loss_dict

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                x: tensor,
                y: tensor,
                x_imp: tensor,
                x_cond: tensor = None):
        """
        Forward pass of the model. Encodes information present in X into the latent space Z. Imputes data from it,
        and predicts Y using both
        :param x: [Ns, in_dim] - Fixed input data in the GP - X_obs
        :param y: [Ns, out_dim] - Target data to predict in the GP regression
        :param x_imp: [Ns, miss_dim] - Ground truth missing data [used in the autoencoder] - Imputed
        :param x_cond: [Ns, cond_dim] - Data to condition the autoencoder
        """
        # Input for the autoencoder
        if self.condition:
            # inputs = [x_imp, x_cond]
            inputs = [x_imp, torch.cat((x_cond, x), dim=1)]
        else:
            inputs = [x_imp]
        mu, logvar, x_hat_mu, x_hat_logvar = self._VI(*inputs)

        # Input data for the GP (Observed data + imputed missing data)
        # x_hat_mu_2, _ = self._decoder(torch.cat((mu, x, x_cond), dim=1))
        # x_hat_mu = self.reparameterize(x_hat_mu, x_hat_logvar)
        # self._decoder = copy.deepcopy(self._VI._decoder)
        # self._decoder.requires_grad_(False)
        # self._decoder.load_state_dict(self._VI._decoder.state_dict())

        # Reparametrize
        # cov_data = x_hat_mu.T.mm(x_hat_mu) / x_hat_mu.shape[0] + torch.diag(torch.exp(x_hat_logvar.mean(dim=0)))
        # sqrt_cov = torch.cholesky(cov_data)
        # x_hat_mu = torch.randn_like(x_hat_mu).mm(sqrt_cov) + x_hat_mu
        # distr = torch.distributions.MultivariateNormal(torch.zeros_like(x_hat_mu), cov_data)
        # x_hat_mu = distr.sample()

        if self._VI.training:
            # self._decoder.load_state_dict(self._VI._decoder.state_dict())
            # x_hat_mu2 = self._decoder(torch.cat((mu, x.data, x_cond), dim=1))[0]
            X = torch.cat((x, x_hat_mu), dim=1)
            # X = torch.cat((x, x_hat_mu2), dim=1)
            # X = torch.cat((x, x_hat_mu2), dim=1)
        else:
            # x_hat_mu2 = self._decoder(torch.cat((mu, x.data, x_cond), dim=1))[0]
            X = torch.cat((x, x_imp), dim=1)
        if torch.any(torch.isnan(x_hat_mu)):
            error_msg = "NaN detected in the mean of the imputed features"
            log.debug(f"{error_msg}")
            raise RuntimeError(f"{error_msg}")

        if torch.any(torch.isinf(x_hat_mu)):
            error_msg = "NaN detected in the mean of the imputed features"
            log.debug(f"{error_msg}")
            raise RuntimeError(f"{error_msg}")

        if type(self._Reg) == RegressionGP:
            y_mu, y_cov_matrix = self._Reg(X.to(self.device), y.to(self.device))
        else:
            y_mu = self._Reg(X)
            y_cov_matrix = []

        return mu, logvar, x_hat_mu, x_hat_logvar, y_mu, y_cov_matrix

    def predict(self,
                x: tensor,
                x_cond: tensor = None):
        """
        Use the trained parameters of the model to predict observations from unseen data. Relying on conditional
        gaussian forms and expected zero-mean.
        mu_pred = mean_test + cov_test_train * cov_train^-1 * (y_train - mean_train)
        mu_pred = cov_test_train * cov_train^-1 * y_train [When means assumed to be 0]
        :param x: [Ns_test, in_dim]
        :return: x_cond: [Ns, cond_dim] - Conditional data for the VI
        """

        # num_samples = 100
        # y_train = tensor(torch.zeros(self._Reg._GP[0].y_train.shape[0], self._Reg._output_dim))
        # x_train = self._Reg._GP[0].x_train.data
        # for ix in range(0, self._Reg._output_dim):
        #     y_train[:, ix] = self._Reg._GP[ix].y_train
        #
        # data_train = torch.cat((x_train, y_train), dim=1)
        # mean = data_train.mean(dim=0)
        # cov = tensor(np.cov(data_train.T))
        # distr = torch.distributions.MultivariateNormal(mean, cov + torch.eye(cov.shape[0]) * 0.00001)

        num_samples = 0
        # x_hat = torch.zeros((x.shape[0], self._miss_dim, num_samples))
        with torch.no_grad():
            # self._VI.training = False
            x_dummy = torch.zeros((x.shape[0], self._miss_dim)).to(self.device)
            # In the VI model we will get the mean latent space (if not in training)
            # x_dummy = tensor(torch.randn_like(x_dummy))
            if self.condition:
                if x_cond is not None:
                    x_cond_pass = torch.cat((x_cond, x), dim=1)
                else:
                    x_cond_pass = x
                # self._VI.training = True
                mu, logvar, x_hat_mu, x_hat_logvar = self._VI(x_dummy, x_cond_pass, num_samples=num_samples)
            else:
                mu, logvar, x_hat_mu, x_hat_logvar = self._VI(x_dummy)

        # probabilities = torch.zeros((x.shape[0], num_samples))
        # for ix in range(0, num_samples):
        #         x_test = torch.cat((x, x_hat_mu[:, :, ix]), dim=1)
        #         if type(self._Reg) == RegressionGP:
        #             y_pred, cov_cond = self._Reg.predict(x_test)
        #         else:
        #             y_pred = self._Reg(x_test)
        #             cov_cond = []
        #
        #         # x_hat[:, :, ix] = x_hat_mu
        #         probabilities[:, ix] = distr.log_prob(torch.cat((x_test, y_pred), dim=1))

        # Get locations with higher probability
        # ind_max = probabilities.max(dim=1)[1]
        # # ind_max = probabilities.min(dim=1)[1]
        # x_hat_def = torch.zeros_like(x_hat_mu[:, :, 0].squeeze())
        # for ix in range(x.shape[0]):
        #     x_hat_def[ix, :] = x_hat_mu[ix, :, ind_max[ix]]
        # x_test = torch.cat((x, x_hat_def), dim=1)
        # y_pred, cov_cond = self._Reg.predict(x_test)

        # x_hat_mu2 = self._decoder(torch.cat((mu, x.data, x_cond), dim=1))[0]
        # x_hat_mu2 = self._decoder(torch.cat((mu, x.data), dim=1))[0]

        # x_test = torch.cat((x, x_hat_mu2), dim=1)
        x_test = torch.cat((x, x_hat_mu), dim=1)
        y_pred, cov_cond = self._Reg.predict(x_test)

        return y_pred, cov_cond, x_hat_mu

        # return y_pred, cov_cond, x_hat_def
