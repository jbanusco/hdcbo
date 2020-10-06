from torch import nn
from hdcob.config import *
from hdcob.gp.gaussian_process import RegressionGP
from hdcob.vi.vi_models import CVAE, VAE, CVI


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
                 use_param: bool = False):
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

        # Initialize VAE
        if self.condition:
            # self._VI = CVAE(self._input_dim, self._cond_dim,
            #                 hidden_dim=hidden_dim, latent_dim=self._latent_dim, bias=bias,
            #                 init_noise=init_noise_vi, prior_noise=prior_noise, param=use_param)
            self._VI = CVI(self._input_dim, self._cond_dim, self._miss_dim,
                           hidden_dim=hidden_dim, latent_dim=self._latent_dim, bias=bias,
                           init_noise=init_noise_vi, prior_noise=prior_noise, param=use_param)
        else:
            self._VI = VAE(self._miss_dim, hidden_dim=hidden_dim, latent_dim=self._latent_dim,
                           bias=bias, init_noise=init_noise_vi, prior_noise=prior_noise,
                           param=use_param)

        # Initialize GP
        self._GP = RegressionGP(init_sigma=init_sigma_gp, init_lamda=init_lamda_gp, init_mean=init_mean_gp,
                                init_noise=init_noise_gp, input_dim=self._n_features, output_dim=self._target_dim,
                                kernel=kernel)

    def save_training_info(self):
        """ Save training info """
        gp_info = self._GP.save_training_info()
        vi_info = self._VI.save_training_info()
        info = {'gp_info': gp_info,
                'vi_info': vi_info}
        return info

    def load_training_info(self,
                           training_info):
        """ Load training info """
        self._GP.load_training_info(training_info['gp_info'])
        self._VI.load_training_info(training_info['vi_info'])

    def add_prior_gp(self,
                     param_name,
                     prior):
        self._GP.add_prior(f"{param_name}", prior)

    def loss(self, rec_target, pred_target, outputs):
        """ Loss function """

        def_value = tensor([0]).squeeze()
        num_samples = pred_target.shape[0]

        loss_vi = self._VI.loss(outputs[:4], rec_target)
        losses_gp = self._GP.loss((outputs[4].squeeze(), outputs[5]), pred_target)

        # We need to weight the number of features predicted and imputed, in order that the optimization is stable
        num_pred = pred_target.shape[1]
        num_miss = rec_target.shape[1]
        factor_gp = 1 / (num_samples * num_pred)
        factor_vi = 1 / num_miss

        total = loss_vi['total'] * factor_vi + losses_gp['total'] * factor_gp
        gp_ll = losses_gp['ll_gp']
        loss_gp = losses_gp['total']
        lamda_kl = losses_gp['kl_gp']

        loss_dict = dict(total=total,
                         total_gp=loss_gp * factor_gp,
                         ll_gp=gp_ll * factor_gp,
                         kl_gp=lamda_kl * factor_gp,
                         total_vi=loss_vi.get('total', def_value) * factor_vi,
                         kl_vi=loss_vi.get('kl', def_value) * factor_vi,
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
        # NOTE: If you want to use the autoencoder version without changing anything from the code
        just set as x the x_imp
        """
        # Input for the autoencoder
        if self.condition:
            inputs = [x, x_cond]
        else:
            inputs = [x]
        mu, logvar, x_hat_mu, x_hat_logvar = self._VI(*inputs)

        # Input data for the GP (Observed data + imputed missing data)
        X = torch.cat((x, x_hat_mu), dim=1)

        if torch.any(torch.isnan(x_hat_mu)):
            error_msg = "NaN detected in the mean of the imputed features"
            log.debug(f"{error_msg}")
            raise RuntimeError(f"{error_msg}")

        if torch.any(torch.isinf(x_hat_mu)):
            error_msg = "NaN detected in the mean of the imputed features"
            log.debug(f"{error_msg}")
            raise RuntimeError(f"{error_msg}")
        y_mu, y_cov_matrix = self._GP(X, y)

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
        :param x_cond: [Ns_test, cond_dim] Conditional data for the VI
        """

        with torch.no_grad():
            # self._VI.training = False
            # x_dummy = tensor(torch.zeros(x.shape[0], self._miss_dim))
            # In the VI model we will get the mean latent space (if not in training)
            x_dummy = x
            if self.condition:
                # x_cond = torch.cat((x_cond, x), dim=1)
                mu, logvar, x_hat_mu, x_hat_logvar = self._VI(x_dummy, x_cond)
            else:
                mu, logvar, x_hat_mu, x_hat_logvar = self._VI(x_dummy)

        x_test = torch.cat((x, x_hat_mu), dim=1)
        y_pred, cov_cond = self._GP.predict(x_test)

        return y_pred, cov_cond, x_hat_mu
