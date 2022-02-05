import numpy as np
import pandas as pd
import os
import torch
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from hdcob.virca.virca import VIRCA
from hdcob.vi.vi_models import CVAE, ICVAE
from hdcob.vi.test_autoencoders import train_vi
from hdcob.gp.gaussian_process import RegressionGP
from hdcob.gp.test_gp_single_input_output import train_gp
from hdcob.virca.test_model import train_virca
from hdcob.virca.generator_synthetic import SyntheticDataset
from hdcob.config import tensor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from hdcob.config import *


import argparse
from distutils.util import strtobool


parser = argparse.ArgumentParser(description='Running GP in test data')

parser.add_argument('--save_folder', type=str, default=os.path.join(os.getcwd(), "results"),
                    help='Folder where to save the results')

parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs')

parser.add_argument('--snr_miss', type=int, default=3,
                    help="Latent dimensions")

parser.add_argument('--snr_target', type=int, default=3,
                    help="Latent dimensions")

parser.add_argument('--lr_decay', type=int, default=3,
                    help="Latent dimensions")

parser.add_argument('--missing', type=int, default=1,
                    help="Latent dimensions")

parser.add_argument('--target', type=int, default=1,
                    help="Latent dimensions")

parser.add_argument('--input', type=int, default=1,
                    help="Latent dimensions")

parser.add_argument('--conditional', type=int, default=1,
                    help="Latent dimensions")

parser.add_argument('--latent', type=int, default=1,
                    help="Latent dimensions")

parser.add_argument('--hidden', type=int, default=0,
                    help="Hidden dimensions")

parser.add_argument('--batch_size', type=int, default=0,
                    help="Batch size, if 0 it means all the dataset")

parser.add_argument('--lr', type=float, default=5e-2,
                    help="Learning rate")

parser.add_argument('--epochs_kl', type=int, default=10,
                    help='Number of epochs to warm-up KL')

parser.add_argument('--print_epochs', type=int, default=100,
                    help="Print epochs")

parser.add_argument("--resume", type=strtobool, nargs='?', const=True, default=False,
                    help="Resume previous optimization if available")

parser.add_argument("--use_param", type=strtobool, nargs='?', const=True, default=True,
                    help="Use parameter for variance of the decoding distribution")


class DummyDataset(Dataset):
    """
    Dataset containing DummyData to test the GP classes
    """

    def __init__(self, input_data, cond_data, missing_data, target_data):
        """

        """
        self.input_data = input_data
        self.cond_data = cond_data
        self.missing_data = missing_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if len(self.cond_data) > 0:
            cond_data = self.cond_data[idx]
        else:
            cond_data = tensor([0])

        sample = {'input': self.input_data[idx],
                  'conditional': cond_data,
                  'missing': self.missing_data[idx],
                  'target': self.target_data[idx],
                  }
        return sample


class DummyDataset_GP(Dataset):
    """
    Dataset containing DummyData to test the GP classes
    """

    def __init__(self, x, y):
        """

        """
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x': self.x[idx],
                  'y': self.y[idx]
                  }
        return sample


class DummyDataset_CVAE(Dataset):
    """
    Dataset containing DummyData to test the VAE classes
    """

    def __init__(self, x, conditional=None):
        """
        x: Input data. In normal autoencoder it's all we need
        target: If instead of an autoencoder we want to use x to infer the target data
        conditional: If we want to condition the inference on some data
        """
        self.data = x
        self.conditional = conditional

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_input = self.data[idx]
        if self.conditional is not None:
            data_conditional = self.conditional[idx]
        else:
            data_conditional = tensor([0])  # Don't allow None

        sample = {'input': data_input,
                  'conditional': data_conditional
                  }
        return sample


if __name__ == '__main__':
    # ==================================================================================================================
    # ======================================= Prepare the data =========================================================

    data = pd.read_csv("Cornell.csv", header=0, index_col=0)
    header = list(data.columns)

    observed = ['X1', 'X2', 'X3', 'X4']
    missing = ['X5', 'X6']
    target = ['Y']
    condition = ["X7"]

    x_obs = data[observed].copy()  # Observations [available components]
    x_obs = tensor(x_obs.to_numpy())

    x_miss = data[missing].copy()  # Missing observations and observed
    x_miss = tensor(x_miss.to_numpy())

    cond = pd.get_dummies(data[condition]).copy()
    cond = tensor(cond.to_numpy())

    target = data[target].copy()  # Predictor judge scores
    target = tensor(target.to_numpy())

    INPUT_DIM = x_obs.shape[1]
    MISSING_DIM = x_miss.shape[1]
    COND_DIM = cond.shape[1]
    OUTPUT_DIM = target.shape[1]

    parsed_args = parser.parse_args()

    log = logging.getLogger(LOGGER_NAME)
    log.info("Comparison")

    runs_path = parsed_args.save_folder
    os.makedirs(runs_path, exist_ok=True)

    # Options
    EPOCHS = parsed_args.epochs
    PRINT_EPOCHS = parsed_args.print_epochs
    WARM_UP_KL = parsed_args.epochs_kl
    LR = parsed_args.lr
    LOAD_PREVIOUS = bool(parsed_args.resume)
    BATCH_SIZE = parsed_args.batch_size

    LATENT_DIM = parsed_args.latent
    HIDDEN_DIM = parsed_args.hidden
    PARAM = bool(parsed_args.use_param)

    LATENT_DIM = 1  # Latent dimension of the autoencoder
    BATCH_SIZE = 100
    SEED = 327  # For reproducibility
    EPOCHS = 500
    LOAD_PREVIOUS = False
    PRINT_EPOCHS = EPOCHS + 100
    WARM_UP_KL = 50
    LR_DECAY = int(EPOCHS / 3)
    # LR_DECAY = EPOCHS + 100
    PARAM = False
    LR = 5e-3

    # Initial conditions
    SIGMA_GP = 0
    LAMDA_GP = 0
    MEAN_GP = 0
    NOISE_GP = 0
    NOISE_VI = 1
    BIAS = False
    HIDDEN_DIM = 0
    KERNEL = "RBF_ML"
    # KERNEL = "Lin_MC"

    # Reg = "Linear"  # or GP
    Reg = "GP"

    # Save folder
    results_folder = os.path.join(os.getcwd(), "results_cornell")
    os.makedirs(results_folder, exist_ok=True)

    # Multiple input features to predict one output
    NUM_SAMPLES = data.shape[0]

    np.random.seed(SEED)  # Just in case we use shuffle option - reproducibility
    if SEED is not None: torch.manual_seed(SEED)
    DataSet = DummyDataset(x_obs, cond, x_miss, target)

    # Define the samplers
    indices = list(range(0, NUM_SAMPLES))
    split = int(np.floor(NUM_SAMPLES * 0.33))
    train_idx, test_idx = indices[split:], indices[:split]
    train_idx = indices
    test_idx = indices
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Standardize the data based on the training set
    mean_data_input_data = DataSet.input_data[train_idx].mean(dim=0)
    std_data_input_data = DataSet.input_data[train_idx].std(dim=0)
    DataSet.input_data = (DataSet.input_data - mean_data_input_data) / std_data_input_data

    mean_data_cond_data = DataSet.cond_data[train_idx].mean(dim=0)
    std_data_cond_data = DataSet.cond_data[train_idx].std(dim=0)
    DataSet.cond_data = (DataSet.cond_data - mean_data_cond_data) / std_data_cond_data

    mean_data_missing_data = DataSet.missing_data[train_idx].mean(dim=0)
    std_data_missing_data = DataSet.missing_data[train_idx].std(dim=0)
    DataSet.missing_data = (DataSet.missing_data - mean_data_missing_data) / std_data_missing_data

    mean_data_target_data = DataSet.target_data[train_idx].mean(dim=0)
    std_data_target_data = DataSet.target_data[train_idx].std(dim=0)
    DataSet.target_data = (DataSet.target_data - mean_data_target_data) / std_data_target_data

    # Get the data loaders
    if BATCH_SIZE == 0:
        train_loader = DataLoader(DataSet, batch_size=len(train_idx), sampler=train_sampler)
        test_loader = DataLoader(DataSet, batch_size=len(test_idx), sampler=test_sampler)
    else:
        train_loader = DataLoader(DataSet, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(DataSet, batch_size=len(test_idx), sampler=test_sampler)

    # ==================================================================================================================
    # =========================================== Run VIRCA ============================================================

    path_test = os.path.join(results_folder, f"VIRCA_L{LATENT_DIM}_H{HIDDEN_DIM}_K{KERNEL}_{LR}")
    virca_model = VIRCA(input_dim=INPUT_DIM,
                        miss_dim=MISSING_DIM,
                        cond_dim=COND_DIM,
                        latent_dim=LATENT_DIM,
                        hidden_dim=HIDDEN_DIM,
                        target_dim=OUTPUT_DIM,
                        init_sigma_gp=SIGMA_GP,
                        init_lamda_gp=LAMDA_GP,
                        init_mean_gp=MEAN_GP,
                        init_noise_gp=NOISE_GP,
                        init_noise_vi=NOISE_VI,
                        bias=BIAS,
                        kernel=f'{KERNEL}',
                        prior_noise=None,
                        use_param=PARAM)
    # virca_model.add_prior_gp("sigma", prior={'mean': tensor([0]), 'logvar': tensor([1])})  # Weak prior

    optim_options = {'epochs': EPOCHS,
                     'load_previous': LOAD_PREVIOUS,
                     'print_epochs': PRINT_EPOCHS,
                     'lr': LR,
                     'seed': SEED,
                     'latent_dim': LATENT_DIM,
                     'hidden_dim': HIDDEN_DIM,
                     'var_param': PARAM,
                     'batch_size': BATCH_SIZE,
                     'warm_up_kl': WARM_UP_KL,
                     'kernel': KERNEL,
                     'hp_params': {'lr': LR, 'latent': LATENT_DIM, 'hidden': HIDDEN_DIM, 'batch_size': BATCH_SIZE,
                                   'kernel': KERNEL}}

    groups = [dict(params=virca_model._VI.parameters(), lr=LR),
              dict(params=virca_model._Reg.parameters(), lr=LR)]
    optimizer = optim.Adam(groups, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY, gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[WARM_UP_KL], gamma=0.1)
    train_virca(virca_model, optimizer, path_test, train_loader, test_loader, optim_options, scheduler)

    virca_model.eval()
    if COND_DIM > 0:
        ours_target, _, ours_miss = virca_model.predict(DataSet.input_data[test_idx], DataSet.cond_data[test_idx])
    else:
        ours_target, _, ours_miss = virca_model.predict(DataSet.input_data[test_idx])

    # ==================================================================================================================
    # ========================================= Classic methods ========================================================

    # ====================> KNN for imputation
    data_train = 0
    if COND_DIM > 0:
        x_input_knn = torch.cat((DataSet.cond_data[train_idx], DataSet.input_data[train_idx]), dim=1).cpu().data.numpy()
        x_input_knn_test = torch.cat((DataSet.cond_data[test_idx], DataSet.input_data[test_idx]), dim=1).cpu().data.numpy()
    else:
        x_input_knn = DataSet.input_data[train_idx].cpu().data.numpy()
        x_input_knn_test = DataSet.input_data[test_idx].cpu().data.numpy()
    # x_input_knn = DataSet.cond_data[train_idx].cpu().data.numpy()
    y_knn = DataSet.missing_data[train_idx].cpu().data.numpy()

    # x_input_knn_test = DataSet.cond_data[test_idx].cpu().data.numpy()

    list_scores = list()
    max_neigh = min(len(train_idx), 50) - 3
    for n in range(0, max_neigh):
        scores = cross_val_score(KNeighborsRegressor(n_neighbors=n), x_input_knn, y_knn,
                                 scoring='neg_mean_squared_error', cv=5)
        list_scores.append(np.abs(scores.mean()))
        print(f"====> {n} <======")
        print(np.abs(scores.mean()))
    # best_k = np.argmin(list_scores)
    best_k = int(np.where(list_scores == pd.DataFrame(data=list_scores).min()[0])[0][0])
    print(f"{best_k}")

    knn_neigh = KNeighborsRegressor(n_neighbors=best_k)
    knn_neigh.fit(x_input_knn, y_knn)

    # ========================> Imputation (CVAE)
    path_test = os.path.join(results_folder, f"CVAE_{LR}_L{LATENT_DIM}_P{PARAM}")

    if COND_DIM > 0:
        DataSet_CVAE = DummyDataset_CVAE(
            DataSet.missing_data,
            # conditional=DataSet.cond_data,
            conditional=torch.cat((DataSet.cond_data, DataSet.input_data), dim=1),
        )
    else:
        DataSet_CVAE = DummyDataset_CVAE(
            DataSet.missing_data,
            # conditional=DataSet.cond_data,
            conditional=DataSet.input_data,
        )

    if BATCH_SIZE == 0:
        train_loader_cvae = DataLoader(DataSet_CVAE, batch_size=len(train_idx), sampler=train_sampler)
        test_loader_cvae = DataLoader(DataSet_CVAE, batch_size=len(test_idx), sampler=test_sampler)
    else:
        train_loader_cvae = DataLoader(DataSet_CVAE, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader_cvae = DataLoader(DataSet_CVAE, batch_size=len(test_idx), sampler=test_sampler)

    # Train the model
    model_cvae = CVAE(
        input_dim=MISSING_DIM,
        cond_dim=COND_DIM + INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        param=PARAM,
        init_noise=NOISE_VI
    )

    # model_cvae = ICVAE(
    #     input_dim=MISSING_DIM,
    #     cond_dim=COND_DIM + INPUT_DIM,
    #     hidden_dim=HIDDEN_DIM,
    #     latent_dim=LATENT_DIM,
    #     param=PARAM,
    #     init_noise=NOISE_VI
    # )

    optim_options = {'epochs': EPOCHS,
                     'load_previous': LOAD_PREVIOUS,
                     'print_epochs': PRINT_EPOCHS,
                     'lr': LR,
                     'seed': SEED,
                     'latent_dim': LATENT_DIM,
                     'hidden_dim': HIDDEN_DIM,
                     'var_param': PARAM,
                     'batch_size': BATCH_SIZE,
                     'warm_up_kl': WARM_UP_KL,
                     'hp_params': {'lr': LR, 'latent': LATENT_DIM, 'hidden': HIDDEN_DIM}}

    optimizer = optim.Adam(model_cvae.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[WARM_UP_KL], gamma=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY, gamma=0.1)
    train_vi(model_cvae, optimizer, path_test, train_loader_cvae, test_loader_cvae, optim_options, scheduler)
    model_cvae.eval()

    # ========================> Regression (GP)
    if COND_DIM > 0:
        cvae_pred = model_cvae.forward(torch.zeros_like(DataSet.missing_data),
                                       torch.cat((DataSet.cond_data, DataSet.input_data), dim=1))
    else:
        cvae_pred = model_cvae.forward(torch.zeros_like(DataSet.missing_data),
                                       DataSet.input_data)
    cvae_pred = cvae_pred[2]

    path_test = os.path.join(results_folder, f"GP_Reg_{KERNEL}_{LR}")

    # X = torch.cat((DataSet.input_data, DataSet.missing_data), dim=1)
    X = torch.cat((DataSet.input_data, cvae_pred.data), dim=1)
    Y = DataSet.target_data
    DataSet_Reg = DummyDataset_GP(X, Y)

    if BATCH_SIZE == 0:
        train_loader_reg = DataLoader(DataSet_Reg, batch_size=len(train_idx), sampler=train_sampler)
        test_loader_reg = DataLoader(DataSet_Reg, batch_size=len(test_idx), sampler=test_sampler)
    else:
        train_loader_reg = DataLoader(DataSet_Reg, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader_reg = DataLoader(DataSet_Reg, batch_size=len(test_idx), sampler=test_sampler)

    if Reg == "GP":
        # Train the model
        gp_model = RegressionGP(init_sigma=SIGMA_GP,
                                init_lamda=LAMDA_GP,
                                init_noise=NOISE_GP,
                                init_mean=MEAN_GP,
                                input_dim=X.shape[1],
                                output_dim=Y.shape[1],
                                kernel=f"{KERNEL}")
        # gp_model.add_prior("sigma", prior={'mean': tensor([0]), 'logvar': tensor([1])})  # Weak prior

        optim_options = {'epochs': EPOCHS*2,
                         'load_previous': LOAD_PREVIOUS,
                         'print_epochs': PRINT_EPOCHS,
                         'lr': LR,
                         'kernel': KERNEL,
                         'hp_params': {'lr': LR, 'kernel': KERNEL},  # Hyper-parameters to give to the tensorbaord
                         }

        optimizer = optim.Adam(gp_model.parameters(), lr=LR, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY, gamma=0.1)
        train_gp(gp_model, optimizer, path_test, train_loader_reg, test_loader_reg, optim_options)
        gp_model.eval()
    else:
        # Try a linear regression
        gp_model = torch.nn.Linear(MISSING_DIM+INPUT_DIM, OUTPUT_DIM, bias=BIAS)
        optimizer = optim.Adam(gp_model.parameters(), lr=LR, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY, gamma=0.1)

        from hdcob.utilities.metrics import mse
        for epoch in range(0, EPOCHS):
            gp_model.train()

            for batch_idx, data in enumerate(train_loader_reg):
                x_data = data['x'].to(DEVICE)
                y_data = data['y'].to(DEVICE)

                # with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                Y_pred = gp_model.forward(x_data)
                loss = mse(Y_pred, y_data)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

        gp_model.eval()

    # ========================> Median
    median_value = DataSet.missing_data[train_idx].median(dim=0)[0]

    # ========================> Mean
    mean_value = DataSet.missing_data[train_idx].mean(dim=0)

    # ========================> CVAE + GP
    # cvae_pred = model_cvae.forward(DataSet.missing_data[test_idx], DataSet.cond_data[test_idx])
    if COND_DIM > 0:
        cvae_pred = model_cvae.forward(torch.zeros_like(DataSet.missing_data[test_idx]),
                                       torch.cat((DataSet.cond_data[test_idx], DataSet.input_data[test_idx]), dim=1))
    else:
        cvae_pred = model_cvae.forward(torch.zeros_like(DataSet.missing_data[test_idx]),
                                       DataSet.input_data[test_idx])
    cvae_pred = cvae_pred[2]

    X_test = torch.cat((DataSet.input_data[test_idx], tensor(cvae_pred)), dim=1)
    if Reg == "GP":
        cvae_gp_pred, _ = gp_model.predict(X_test)
    else:
        cvae_gp_pred = gp_model(X_test)

    # ========================> KNN + GP
    gp_model.eval()
    y_pred_knn = knn_neigh.predict(x_input_knn_test)
    X_test = torch.cat((DataSet.input_data[test_idx], tensor(y_pred_knn)), dim=1)
    if Reg == "GP":
        knn_gp_pred, _ = gp_model.predict(X_test)
    else:
        knn_gp_pred = gp_model(X_test)

    # ==================================================================================================================
    # ========================================= Compute errors =========================================================

    # ========================> Imputation
    methods_list = ["Mean", "Median", "KNN", "CVAE", "Ours"]
    mean_imp_errors = pd.DataFrame([], columns=methods_list, index=["MSE"])

    # Squared error
    mean_se = ((DataSet.missing_data[test_idx] - mean_value) ** 2)
    median_se = ((DataSet.missing_data[test_idx] - median_value) ** 2)
    knn_se = ((DataSet.missing_data[test_idx] - y_pred_knn) ** 2)
    cvae_se = ((DataSet.missing_data[test_idx] - cvae_pred) ** 2)
    ours_se = ((DataSet.missing_data[test_idx] - ours_miss) ** 2)

    imp_errors = pd.DataFrame(data=np.concatenate((mean_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   median_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   knn_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   cvae_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   ours_se.sum(dim=1).unsqueeze(1).cpu().data.numpy()), axis=1),
                              columns=methods_list)
    mean_imp_errors.loc["MSE"] = imp_errors.mean()

    # ========================> Emulation
    methods_list = ["KNN+GP", "CVAE+GP", "Ours"]
    mean_emulation_errors = pd.DataFrame([], columns=methods_list, index=["MSE"])

    # Squared error
    knn_gp_se = ((DataSet.target_data[test_idx] - knn_gp_pred) ** 2)
    cvae_gp_se = ((DataSet.target_data[test_idx] - cvae_gp_pred) ** 2)
    ours_em_se = ((DataSet.target_data[test_idx] - ours_target) ** 2)

    emulation_errors = pd.DataFrame(data=np.concatenate((knn_gp_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                         cvae_gp_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                         ours_em_se.sum(dim=1).unsqueeze(1).cpu().data.numpy()), axis=1),
                                    columns=methods_list)
    mean_emulation_errors.loc["MSE"] = emulation_errors.mean()

    # ==================================================================================================================
    # ========================================= Plot comparison ========================================================
    list_missing_features = [f"Miss. {ix}" for ix in range(MISSING_DIM)]
    list_target_features = [f"Targ. {ix}" for ix in range(OUTPUT_DIM)]

    # Errors by variable / dimension
    err_mean_dim = pd.DataFrame(mean_se.cpu().data.numpy(), columns=list_missing_features)
    err_mean_dim["Index"] = "Mean"

    err_median_dim = pd.DataFrame(median_se.cpu().data.numpy(), columns=list_missing_features)
    err_median_dim["Index"] = "Median"

    err_knn_dim = pd.DataFrame(knn_se.cpu().data.numpy(), columns=list_missing_features)
    err_knn_dim["Index"] = "KNN"

    err_cvi_dim = pd.DataFrame(cvae_se.cpu().data.numpy(), columns=list_missing_features)
    err_cvi_dim["Index"] = "CVAE"

    err_ours_dim = pd.DataFrame(ours_se.cpu().data.numpy(), columns=list_missing_features)
    err_ours_dim["Index"] = "Ours"

    # imp_errors_dim = pd.concat([err_mean_dim, err_median_dim, err_knn_dim, err_cvi_dim, err_ours_dim], axis=0)
    imp_errors_dim = pd.concat([err_knn_dim, err_cvi_dim, err_ours_dim], axis=0)
    # imp_errors_dim = pd.concat([err_cvi_dim, err_ours_dim], axis=0)

    err_knn_gp_dim = pd.DataFrame(knn_gp_se.cpu().data.numpy(), columns=list_target_features)
    err_knn_gp_dim["Index"] = "KNN+GP"

    err_cvi_gp_dim = pd.DataFrame(cvae_gp_se.cpu().data.numpy(), columns=list_target_features)
    err_cvi_gp_dim["Index"] = "CVAE+GP"

    err_ours_em_dim = pd.DataFrame(ours_em_se.cpu().data.numpy(), columns=list_target_features)
    err_ours_em_dim["Index"] = "Ours"

    em_errors_dim = pd.concat([err_knn_gp_dim, err_cvi_gp_dim, err_ours_em_dim], axis=0)

    # Actual plotting
    imputations = pd.melt(imp_errors_dim, id_vars="Index")
    emulations = pd.melt(em_errors_dim, id_vars="Index")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxenplot(data=imputations, x="variable", y="value", hue="Index", ax=ax[0])
    sns.boxenplot(data=emulations, x="variable", y="value", hue="Index", ax=ax[1])

    min_val = scipy.percentile(imputations["value"], 1)
    max_val = scipy.percentile(imputations["value"], 99)
    ax[0].set_ylim([min_val, max_val])

    min_val = scipy.percentile(emulations["value"], 1)
    max_val = scipy.percentile(emulations["value"], 99)
    ax[1].set_ylim([min_val, max_val])

    ax[0].set_xticklabels(list_missing_features)
    ax[1].set_xticklabels(list_target_features)
    ax[0].set_title("Imputation")
    ax[1].set_title("Emulated")

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].legend(ncol=2, title="", loc=3)
    ax[1].legend(ncol=2, title="", loc=3)
    ax[1].set_xlabel("")
    ax[0].set_xlabel("")

    ax[0].set_ylabel("$log_{10}$(Squared Error)")
    ax[1].set_ylabel("$log_{10}$(Squared Error)")
    fig.savefig(os.path.join(results_folder, "figure_errors.png"))

    # Compute significance
    p_thres = 0.05

    # Imputation
    p_thres = p_thres / MISSING_DIM
    df_pvalues = pd.DataFrame(data=None, columns=["KNN", "CVAE"], index=list_missing_features)
    for var in list_missing_features:
        df_pvalues["KNN"].loc[f"{var}"] = scipy.stats.ranksums(err_knn_dim[f"{var}"], err_ours_dim[f"{var}"])[-1]
        df_pvalues["CVAE"].loc[f"{var}"] = scipy.stats.ranksums(err_cvi_dim[f"{var}"], err_ours_dim[f"{var}"])[-1]
    mask_sig = df_pvalues < p_thres
    mask_sig.to_csv(os.path.join(results_folder, "imputation_sig.csv"))
    print(mask_sig)

    # Emulation
    p_thres = p_thres / OUTPUT_DIM
    df_pvalues = pd.DataFrame(data=None, columns=["KNN+GP", "CVAE+GP"], index=list_target_features)
    for var in list_target_features:
        df_pvalues["KNN+GP"].loc[f"{var}"] = scipy.stats.ranksums(err_knn_gp_dim[f"{var}"], err_ours_em_dim[f"{var}"])[-1]
        df_pvalues["CVAE+GP"].loc[f"{var}"] = scipy.stats.ranksums(err_cvi_gp_dim[f"{var}"], err_ours_em_dim[f"{var}"])[-1]
    mask_sig = df_pvalues < p_thres
    mask_sig.to_csv(os.path.join(results_folder, "emulation_sig.csv"))
    print(mask_sig)

