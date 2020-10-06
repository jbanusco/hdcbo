import pandas as pd
from torch import nn
from torch.distributions import kl_divergence, Normal
from hdcob.gp.kernels import kernel_rbf, kernel_linear, kernel_mult_lamdas, kernel_mult_noise, kernel_rbf_mult_params
from hdcob.config import *


def likelihood_gp(mean_pred, covariance_pred, target):
    """
    Compute the loss, assuming gaussian noise
    """

    target = target.squeeze()
    mean = mean_pred.squeeze()

    # Difference between mean and target
    diff = mean - target
    if len(diff.shape) == 1:
        diff = diff.unsqueeze(1)

    # Compute square root using Cholesky
    sqrt_cov = torch.cholesky(covariance_pred)
    logdet_cov = 2 * torch.log(torch.diag(sqrt_cov)).sum()  # We need a triangular matrix, not valid the svd result.

    # Compute the inverse using a system of equations
    tmp_a, _ = torch.solve(diff, sqrt_cov)  # [Ns_train, 1]
    squared_term = tmp_a.T.mm(tmp_a)

    num_samples = target.shape[0]
    ll = -0.5 * (num_samples * LOG_2_PI + logdet_cov + squared_term)
    # ll = -0.5 * (logdet_cov + squared_term)

    # torch.distributions.multivariate_normal.MultivariateNormal(loc=mean_pred.unsqueeze(0),
    #                                                            covariance_matrix=covariance_pred).log_prob(target)

    return ll


def kl_prior(prior, mean_q, logvar_q):
    # Prior distribution
    mean_p = prior['mean']
    logvar_p = prior['logvar']

    # Define distributions
    Q = Normal(mean_q, torch.exp(logvar_q * 0.5))
    P = Normal(mean_p, torch.exp(logvar_p * 0.5))

    # KL divergence
    kl = kl_divergence(Q, P)

    return kl


class GeneralGP(nn.Module):
    """ Template class for GPs """
    def __init__(self):
        super(GeneralGP, self).__init__()

        # Training information
        self.cov_train = tensor()
        self.sqrt_cov = tensor()  # Square root
        self.y_train = tensor()  # Training observations for the GP regression
        self.x_train = tensor()  # Training locations for the GP regression

        # Transform information -- Used to scale the data, we need to save it also w. the training info. In
        # order to apply it to future new/test data
        self._mean = pd.DataFrame
        self._stdv = pd.DataFrame

    def forward(self,
                x: tensor,
                y: tensor):
        pass

    def predict(self,
                x: tensor):
        pass

    def sample(self,
               x: tensor,
               num_samples: int = 1):
        pass

    def save_training_info(self):
        """ Generate dictionary with the training information """
        training_info = {
            "cov_train": self.cov_train,
            "sqrt_cov": self.sqrt_cov,
            "x_train": self.x_train,
            "y_train": self.y_train
        }
        return training_info

    def load_training_info(self,
                           training_info: dict):
        """ Since for the prediction we need information regarding the training data we will need to load it.
        Not only the parameters of the model """
        self.cov_train = training_info["cov_train"]
        self.sqrt_cov = training_info["sqrt_cov"]
        self.x_train = training_info["x_train"]
        self.y_train = training_info["y_train"]

    @staticmethod
    def loss(predicted, target):
        pass


class GP(GeneralGP):
    """ Simple GP implementation with RBF kernel """

    def __init__(self,
                 init_sigma: float = 0.1,
                 init_lamda: float = 0.1,
                 init_noise: float = 0.1,
                 init_mean: float = 0.,
                 kernel="RBF"):
        """
        :param init_sigma:  Initial amplitude of the RBF kernel
        :param init_lamda:   Initial lengthscale of the RBF kernel
        :param init_noise: Initial value of the gaussian noise
        :param init_mean: Initial mean of the training observations
        :param kernel: Kernel of the GP
        """
        super(GP, self).__init__()

        self._dim = 1

        # Noise
        self.noise = nn.Parameter(tensor([init_noise]), requires_grad=True)

        # Kernel
        if kernel == "RBF":
            self.kernel = kernel_rbf
        elif kernel == "Lin":
            self.kernel = kernel_linear
        else:
            raise RuntimeError(f"Error kernel {kernel}")

        # Kernel params
        self.sigma = nn.Parameter(tensor([init_sigma]), requires_grad=True)
        self.lamda = nn.Parameter(tensor([init_lamda]), requires_grad=True)

        # Mean
        self.mu = nn.Parameter(tensor([init_mean]), requires_grad=False)

    def forward(self,
                x: tensor,
                y: tensor):
        """
        Using the current hyperparameters generate samples of the distribution to compute the likelihood of the
        parameters
        :param x: Training locations used to predict observations
        :param y: Training observations
        :return:
        """

        # Compute covariance matrix
        cov_train = self.kernel(x, x, self.sigma, self.lamda, noise=self.noise, add_noise=True)
        sqrt_cov = torch.cholesky(cov_train)

        if self.training:
            # Store the info.
            self.cov_train = cov_train.detach()
            self.sqrt_cov = sqrt_cov.detach()
            self.x_train = x  # Store training locations
            self.y_train = y.squeeze()

        return self.mu, cov_train

    def predict(self,
                x: tensor):
        """
        Use conditional properties of gaussian to predict posterior mean and covariance of the testing points.
        mean_cond = mean_test + cov_test_train*( cov_train^-1)*(y_train - mean_train)
        cov_cond = cov_test - cov_test_train * (cov_train^-1) * cov_test_train.T
        :param x: posterior locations / test locations
        :return:
        """

        # Compute cov. matrix between training and test points / [Ns_test, Ns_train]
        cov_test_train = self.kernel(x, self.x_train, self.sigma, self.lamda, noise=self.noise,
                                     add_noise=False)

        # Need to solve inverse AX=B, to find X -[Ns_train, 1] - Do it for both sides to get the full expression.
        diff = self.y_train - self.mu
        tmp_a, _ = torch.solve(diff.unsqueeze(1), self.sqrt_cov)  # [Ns_train, 1]
        tmp_b, _ = torch.solve(cov_test_train.T, self.sqrt_cov)  # [Ns_train, Ns_test]

        mean_cond = (self.mu + torch.mm(tmp_b.T, tmp_a)).squeeze()  # [Ns_test, 1]

        cov_test = self.kernel(x, x, self.sigma, self.lamda, noise=self.noise, add_noise=True)  # [Ns_test, Ns_test]

        cov_cond = cov_test - torch.mm(tmp_b.T, tmp_b)  # [Ns_test, Ns_test]

        return mean_cond, cov_cond

    def sample(self,
               x: tensor,
               num_samples: int = 1):
        """
        Draw samples from the GP posterior
        :param
        """

        mean_cond, cov_cond = self.predict(x)
        sqrt_sample_cov = torch.cholesky(cov_cond)

        # Draw samples from the posterior at our test points
        num_points = x.shape[0]
        z = torch.randn(int(num_points), num_samples)  # Sample from normal distribution
        f_post = mean_cond.unsqueeze(1) + sqrt_sample_cov.mm(z)  # Predict training observations

        return f_post

    @staticmethod
    def loss(predicted, target):
        """ Compute the loss of the GP """
        ll = likelihood_gp(predicted[0], predicted[1], target)
        return {'total': -ll, 'll': -ll}


class GP_Prior(GP):
    def __init__(self,
                 init_sigma: float = 0.1,  # Small
                 init_lamda: float = 0.1,  # Small
                 init_noise: float = 0.1,  # GP noise
                 init_mean: float = 0,  # 0
                 kernel="RBF"):
        """
        :param init_sigma:  Initial amplitude of the RBF kernel
        :param init_lamda:   Initial lengthscale of the RBF kernel
        :param init_noise: Initial value of the gaussian noise
        :param init_mean: Initial mean of the training observations
        """
        super(GP_Prior, self).__init__(init_sigma=init_sigma, init_lamda=init_lamda,
                                       init_noise=init_noise, init_mean=init_mean,
                                       kernel=kernel)

        self.list_priors = dict()  # Priors, log-mean and log-variance
        self.list_parameters = nn.ParameterDict()  # Log-variances

    def add_prior(self,
                  param_name: str,
                  prior: dict):
        """ Add a prior to some of the parameters of the GP kernel """
        # It can be the noise, the sigma or the lamda in the case of the RBF
        if f"{param_name}" in self._parameters.keys():
            self.list_priors[f"{param_name}"] = prior
            self.list_parameters[f"{param_name}"] = nn.Parameter(prior['logvar'], requires_grad=True)

    def loss(self,
             predicted: tuple,
             target: tensor):
        """
        Compute the loss, assuming gaussian noise
        :param
        """
        ll = likelihood_gp(predicted[0], predicted[1], target)

        kl = 0
        for key in self.list_parameters.keys():
            kl += kl_prior(self.list_priors[f"{key}"],  # Prior dictionary
                           eval(f"self.{key}"),  # Mean value
                           self.list_parameters[f"{key}"])  # Log-variance

        loss_dict = {'ll': -ll,
                     'kl': kl,
                     'total': -ll+kl}
        return loss_dict

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                x: tensor,
                y: tensor):
        """
        Using the current hyperparameters generate samples of the distribution to compute the likelihood of the
        parameters
        :param x: Training locations used to predict observations
        :param y: Training observations
        :return:
        """
        if "lamda" in self.list_parameters.keys():
            lamda = self.reparameterize(self.lamda, self.list_parameters["lamda"])
        else:
            lamda = self.lamda

        if "sigma" in self.list_parameters.keys():
            sigma = self.reparameterize(self.sigma, self.list_parameters["sigma"])
        else:
            sigma = self.sigma

        if "noise" in self.list_parameters.keys():
            noise = self.reparameterize(self.noise, self.list_parameters["noise"])
        else:
            noise = self.noise

        cov_train = self.kernel(x, x, sigma, lamda, noise=noise, add_noise=True)  # Compute covariance matrix
        try:
            sqrt_cov = torch.cholesky(cov_train)
        except RuntimeError as e:
            print(error_handling())
            print(cov_train)

        if self.training:
            # Store the info.
            self.cov_train = cov_train.detach()
            self.sqrt_cov = sqrt_cov.detach()
            self.x_train = x  # Store training locations
            self.y_train = y.squeeze()

        return self.mu, cov_train


class GP_HighD(GeneralGP):
    """ High-dimensional GP with RBF kernel """

    def __init__(self,
                 init_sigma: float = 0.1,
                 init_lamda: float = 0.1,
                 init_noise: float = 0.1,
                 init_mean: float = 0,
                 num_inp_features: int = 1,
                 kernel="RBF_M"):
        """
        :param init_sigma:  Initial amplitude of the RBF kernel
        :param init_lamda:   Initial lengthscale of the RBF kernel
        :param init_noise: Initial value of the gaussian noise
        :param init_mean: Initial mean of the training observations
        :param num_features: Number of dimensions
        """
        super(GP_HighD, self).__init__()

        # Dimension of the GP
        self._dim = num_inp_features

        self.sigma = nn.Parameter(tensor([init_sigma]), requires_grad=True)
        self.mu = nn.Parameter(tensor([init_mean]), requires_grad=False)

        if self._dim == 1 and kernel != "Lin":
            kernel = "RBF"

        self.multiple_params = list()
        if kernel == "RBF" or kernel == "Lin":
            self.kernel = kernel_rbf
            self.lamda = nn.Parameter(tensor([init_lamda]), requires_grad=True)
            self.noise = nn.Parameter(tensor([init_noise]), requires_grad=True)
        elif kernel == "RBF_M":
            self.kernel = kernel_rbf_mult_params
            self.noise = nn.ParameterList([nn.Parameter(tensor([init_noise]), requires_grad=True) for ix in range(self._dim)])
            self.lamda = nn.ParameterList([nn.Parameter(tensor([init_lamda]), requires_grad=True) for ix in range(self._dim)])
            self.multiple_params.append("lamda")
            self.multiple_params.append("noise")
        elif kernel == "RBF_ML":
            self.kernel = kernel_mult_lamdas
            self.lamda = nn.ParameterList([nn.Parameter(tensor([init_lamda]), requires_grad=True) for ix in range(self._dim)])
            self.noise = nn.Parameter(tensor([init_noise]), requires_grad=True)
            self.multiple_params.append("lamda")
        elif kernel == "RBF_MN":
            self.kernel = kernel_mult_noise
            self.noise = nn.ParameterList([nn.Parameter(tensor([init_noise]), requires_grad=True) for ix in range(self._dim)])
            self.multiple_params.append("noise")
            self.lamda = nn.Parameter(tensor([init_lamda]), requires_grad=True)
        elif kernel == "Lin_MC":
            self.kernel = kernel_linear
            self.lamda = nn.ParameterList([nn.Parameter(tensor([init_lamda]), requires_grad=True) for ix in range(self._dim)])
            self.noise = nn.Parameter(tensor([init_noise]), requires_grad=True)
            self.multiple_params.append("lamda")
        else:
            raise RuntimeError(f"Unexpected kernel type {kernel}")

        self.list_priors = dict()  # Priors, log-mean and log-variance
        self.list_parameters = nn.ParameterDict()  # Log-variances

    def add_prior(self,
                  param_name: str,
                  prior: dict):
        """ Add a prior to some of the parameters of the GP kernel """
        # It can be the noise, the sigma or the lamda in the case of the RBF
        for p_data in list(self.named_parameters()):
            p_name = p_data[0]
            if f"{param_name}" in p_name:
                self.list_priors[f"{param_name}"] = prior
                # NOT POSSIBLE _ THAT I KNOW
                # if f"{param_name}" in self.multiple_params:
                #     # One independent logvar for each dimension
                #     self.list_parameters[f"{param_name}"] = nn.ParameterList(
                #         [nn.Parameter(prior['logvar'], requires_grad=True) for ix in range(self._dim)])
                # else:
                self.list_parameters[f"{param_name}"] = nn.Parameter(prior['logvar'], requires_grad=True)

    def loss(self,
             predicted: tuple,
             target: tensor):
        """
        Compute the loss
        :param
        """
        ll = likelihood_gp(predicted[0], predicted[1], target).squeeze()

        kl = tensor([0])
        for key in self.list_parameters.keys():
            if f"{key}" in self.multiple_params:
                for ix in range(self._dim):
                    # kl += kl_prior(self.list_priors[f"{key}"],  # Prior dictionary
                    #                eval(f"self.{key}[{ix}]"),  # Mean value
                    #                self.list_parameters[f"{key}"][ix])  # Log-variance
                    kl += kl_prior(self.list_priors[f"{key}"],  # Prior dictionary
                                   eval(f"self.{key}[{ix}]"),  # Mean value
                                   self.list_parameters[f"{key}"])  # Log-variance
            else:
                kl += kl_prior(self.list_priors[f"{key}"],  # Prior dictionary
                               eval(f"self.{key}"),  # Mean value
                               self.list_parameters[f"{key}"])  # Log-variance

        kl = kl.squeeze()
        loss_dict = {'ll': -ll,
                     'kl': kl,
                     'total': -ll+kl}

        return loss_dict

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                x: tensor,
                y: tensor):
        """
        Using the current hyperparameters generate samples of the distribution to compute the likelihood of the
        parameters
        :param x: Training locations used to predict observations
        :param y: Training observations
        :return:
        """
        if "lamda" in self.list_parameters.keys():
            if "lamda" in self.multiple_params:
                # lamda = [self.reparameterize(self.lamda[ix], self.list_parameters["lamda"][ix]) for ix in range(self._dim)]
                lamda = [self.reparameterize(self.lamda[ix], self.list_parameters["lamda"]) for ix in range(self._dim)]
            else:
                lamda = self.reparameterize(self.lamda, self.list_parameters["lamda"])
        else:
            lamda = self.lamda

        if "sigma" in self.list_parameters.keys():
            if "sigma" in self.multiple_params:
                # sigma = [self.reparameterize(self.sigma[ix], self.list_parameters["sigma"][ix]) for ix in range(self._dim)]
                sigma = [self.reparameterize(self.sigma[ix], self.list_parameters["sigma"]) for ix in range(self._dim)]
            else:
                sigma = self.reparameterize(self.sigma, self.list_parameters["sigma"])
        else:
            sigma = self.sigma

        if "noise" in self.list_parameters.keys():
            if "noise" in self.multiple_params:
                # noise = [self.reparameterize(self.noise[ix], self.list_parameters["noise"][ix]) for ix in range(self._dim)]
                noise = [self.reparameterize(self.noise[ix], self.list_parameters["noise"]) for ix in range(self._dim)]
            else:
                noise = self.reparameterize(self.noise, self.list_parameters["noise"])
        else:
            noise = self.noise

        cov_train = self.kernel(x, x, sigma, lamda, noise, add_noise=True)  # Compute covariance matrix
        try:
            sqrt_cov = torch.cholesky(cov_train)
        except RuntimeError as e:
            log.error(f"The GP has crashed calculating the square root of the covariance. The value of the noise is: "
                      f"{torch.exp(noise).item():3f}")
            log.error(f"{e}")
            raise RuntimeError

        if self.training:
            # Store the info.
            self.cov_train = cov_train.detach()
            self.sqrt_cov = sqrt_cov.detach()
            self.x_train = x  # Store training locations
            self.y_train = y.squeeze()

        return self.mu, cov_train

    def predict(self,
                x: tensor):
        """
        Use conditional properties of gaussian to predict posterior mean and covariance of the testing points.
        mean_cond = mean_test + cov_test_train*( cov_train^-1)*(y_train - mean_train)
        cov_cond = cov_test - cov_test_train * (cov_train^-1) * cov_test_train.T
        :param x: posterior locations / test locations
        :return:
        """

        # Compute cov. matrix between training and test points / [Ns_test, Ns_train]
        cov_test_train = self.kernel(x, self.x_train, self.sigma, self.lamda, noise=self.noise,
                                     add_noise=False)

        # Need to solve inverse AX=B, to find X -[Ns_train, 1] - Do it for both sides to get the full expression.
        diff = self.y_train - self.mu
        tmp_a, _ = torch.solve(diff.unsqueeze(1), self.sqrt_cov)  # [Ns_train, 1]
        tmp_b, _ = torch.solve(cov_test_train.T, self.sqrt_cov)  # [Ns_train, Ns_test]

        mean_cond = (self.mu + torch.mm(tmp_b.T, tmp_a)).squeeze()  # [Ns_test, 1]

        cov_test = self.kernel(x, x, self.sigma, self.lamda, noise=self.noise, add_noise=True)  # [Ns_test, Ns_test]

        cov_cond = cov_test - torch.mm(tmp_b.T, tmp_b)  # [Ns_test, Ns_test]

        # std = torch.sqrt(torch.diag(cov_test) - torch.sum(tmp_b ** 2, dim=0))

        return mean_cond, cov_cond


class RegressionGP(GeneralGP):
    """ Multiple inputs and multiple outputs """
    def __init__(self,
                 init_sigma: float = 0.1,
                 init_lamda: float = 0.1,
                 init_noise: float = 0.1,
                 init_mean: float = 0,
                 input_dim: int = 1,
                 output_dim: int = 1,
                 kernel="RBF"):
        super(RegressionGP, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        # Initialize GPs list
        self._GP = nn.ModuleList([GP_HighD(init_sigma=init_sigma, init_lamda=init_lamda, init_mean=init_mean,
                                           init_noise=init_noise, num_inp_features=self._input_dim, kernel=kernel)
                                  for ix in range(self._output_dim)])

    def add_prior(self,
                  param_name: str,
                  prior: dict):
        for ix in range(self._output_dim):
            self._GP[ix].add_prior(param_name, prior)

    def loss(self, predicted, target):
        """ Loss function """

        def_value = tensor([0])

        # loss = tensor([0])
        # ll = tensor([0])
        # kl = tensor([0])
        loss = tensor(torch.zeros(1, self._output_dim))
        ll = tensor(torch.zeros(1, self._output_dim))
        kl = tensor(torch.zeros(1, self._output_dim))

        num_samples = target.shape[0]
        for ix in range(self._output_dim):
            # out_gp = (predicted[0][ix], predicted[1][ix])
            out_gp = (predicted[0][ix], predicted[1][:, :, ix])
            losses_gp = self._GP[ix].loss(out_gp, target[:, ix])
            # loss += losses_gp['total']
            # ll += losses_gp.get('ll', def_value)
            # kl += losses_gp.get('kl', def_value)
            loss[0, ix] = losses_gp['total']
            ll[0, ix] = losses_gp.get('ll', def_value)
            kl[0, ix] = losses_gp.get('kl', def_value)

        total = ll.sum() + kl.sum()

        loss_dict = dict(total=total,
                         ll_gp=ll.sum(),
                         kl_gp=kl.sum())
        # loss_dict = dict(total=loss,
        #                  ll_gp=ll,
        #                  kl_gp=kl)
        return loss_dict

    def forward(self,
                x: tensor,
                y: tensor):
        """ Forward """
        # y_mu = []
        # y_cov_matrix = []
        num_samples = x.shape[0]
        y_mu = tensor(torch.zeros(1, self._output_dim))
        # y_cov_matrix = [tensor() for ix in range(self._output_dim)]
        y_cov_matrix = tensor(torch.zeros(num_samples, num_samples, self._output_dim))
        for ix in range(self._output_dim):
            y_ix, cov_matrix_ix = self._GP[ix](x, y[:, ix])
            y_mu[:, ix] = y_ix
            # y_cov_matrix[ix] = cov_matrix_ix
            y_cov_matrix[:, :, ix] = cov_matrix_ix
            # y_mu.append(y_ix)
            # y_cov_matrix.append(cov_matrix_ix)

        return y_mu.squeeze(), y_cov_matrix

    def predict(self,
                x: tensor):
        """ Predict """

        num_samples = x.shape[0]
        y_pred = tensor(torch.zeros(num_samples, self._output_dim))
        # cov_cond = [tensor() for ix in range(self._output_dim)]
        cov_cond = tensor(torch.zeros(num_samples, num_samples, self._output_dim))
        for ix in range(self._output_dim):
            y_ix, cov_matrix_ix = self._GP[ix].predict(x)
            y_pred[:, ix] = y_ix
            # cov_cond[ix] = cov_matrix_ix
            cov_cond[:, :, ix] = cov_matrix_ix

        return y_pred, cov_cond

    def save_training_info(self):
        """ Generate list of dictionaries with the training information """

        training_info = [self._GP[ix].save_training_info() for ix in range(self._output_dim)]
        return training_info

    def load_training_info(self,
                           training_info: list):
        """ Since for the prediction we need information regarding the training data we will need to load it.
        Not only the parameters of the model """

        for ix in range(self._output_dim):
            self._GP[ix].load_training_info(training_info[ix])
