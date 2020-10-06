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
from hdcob.vi.vi_models import CVAE, CVI
from hdcob.vi.test_autoencoders import train_vi
from hdcob.gp.gaussian_process import RegressionGP
from hdcob.gp.test_gp_single_input_output import train_gp
from hdcob.virca.test_model import train_virca
from hdcob.virca.generator_synthetic import SyntheticDataset
from hdcob.config import tensor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


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


class DummyDataset_CVI(Dataset):
    """
    Dataset containing DummyData to test the VAE classes
    """

    def __init__(self, x, conditional=None, target=None):
        """
        x: Input data. In normal autoencoder it's all we need
        target: If instead of an autoencoder we want to use x to infer the target data
        conditional: If we want to condition the inference on some data
        target: If target and input are different
        """
        self.data = x
        self.conditional = conditional
        self.target = target

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

        if self.target is not None:
            data_target = self.target[idx]
        else:
            data_target = tensor([0])

        sample = {'input': data_input,
                  'conditional': data_conditional,
                  'target': data_target,
                  }
        return sample


if __name__ == '__main__':
    # ==================================================================================================================
    # ======================================= Prepare the data =========================================================
    LATENT_DIM = 1  # Latent dimension of the variational inference
    G_LATENT_DIM = 1  # True latent dimensions
    NOISE_LVL = 0.1  # Noise lvl in the generated data
    BATCH_SIZE = 30  # 0 loads the entire batch
    SEED = 5  # For reproducibility
    EPOCHS = 200
    LOAD_PREVIOUS = True
    PRINT_EPOCHS = EPOCHS + 100
    WARM_UP_KL = 10
    GRADIENT_DECAY = EPOCHS / 3
    PARAM = True
    LR = 5e-2

    # Initial conditions
    SIGMA_GP = 0
    LAMDA_GP = 0
    MEAN_GP = 0
    NOISE_GP = 1
    NOISE_VI = 1
    BIAS = False
    HIDDEN_DIM = 0
    KERNEL = "RBF_ML"

    # Save folder
    results_folder = os.path.join(os.getcwd(), "results")
    os.makedirs(results_folder, exist_ok=True)

    # Multiple input features to predict one output
    NUM_SAMPLES = 500
    MISSING_DIM = 3
    COND_DIM = 5
    INPUT_DIM = 1
    OUTPUT_DIM = 3

    np.random.seed(SEED)  # Just in case we use shuffle option - reproducibility
    if SEED is not None: torch.manual_seed(SEED)
    DataSet = SyntheticDataset(num_samples=NUM_SAMPLES, lat_dim=G_LATENT_DIM,
                               input_dim=INPUT_DIM,
                               cond_dim=COND_DIM,
                               miss_dim=MISSING_DIM,
                               target_dim=OUTPUT_DIM,
                               seed=SEED,
                               noise_lvl=NOISE_LVL,
                               vae=False)

    # Define the samplers
    indices = list(range(0, NUM_SAMPLES))
    split = int(np.floor(NUM_SAMPLES * 0.33))
    train_idx, test_idx = indices[split:], indices[:split]
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

    path_test = os.path.join(results_folder, f"VIRCA_L{LATENT_DIM}_H{HIDDEN_DIM}_K{KERNEL}_{LR}_P{PARAM}")
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
              dict(params=virca_model._GP.parameters(), lr=LR)]
    optimizer = optim.Adam(groups)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=GRADIENT_DECAY, gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[WARM_UP_KL], gamma=0.1)
    train_virca(virca_model, optimizer, path_test, train_loader, test_loader, optim_options, scheduler)

    virca_model.eval()
    ours_target, _, ours_miss = virca_model.predict(DataSet.input_data[test_idx], DataSet.cond_data[test_idx])

    # ==================================================================================================================
    # ========================================= Classic methods ========================================================

    # ====================> KNN for imputation
    data_train = 0
    x_input_knn = torch.cat((DataSet.cond_data[train_idx], DataSet.input_data[train_idx]), dim=1).cpu().data.numpy()
    # x_input_knn = DataSet.cond_data[train_idx].cpu().data.numpy()
    y_knn = DataSet.missing_data[train_idx].cpu().data.numpy()
    x_input_knn_test = torch.cat((DataSet.cond_data[test_idx], DataSet.input_data[test_idx]), dim=1).cpu().data.numpy()
    # x_input_knn_test = DataSet.cond_data[test_idx].cpu().data.numpy()

    list_scores = list()
    for n in range(2, 50):
        scores = cross_val_score(KNeighborsRegressor(n_neighbors=n), x_input_knn, y_knn,
                                 scoring='neg_mean_squared_error', cv=5)
        list_scores.append(np.abs(scores.mean()))
        print(f"====> {n} <======")
        print(np.abs(scores.mean()))
    best_k = np.argmin(list_scores)
    print(f"{best_k}")

    knn_neigh = KNeighborsRegressor(n_neighbors=best_k)
    knn_neigh.fit(x_input_knn, y_knn)

    # ========================> Imputation (CVI)
    path_test = os.path.join(results_folder, f"CVI_{LR}_L{LATENT_DIM}_P{PARAM}")

    DataSet_CVI = DummyDataset_CVI(
        DataSet.input_data,
        target=DataSet.missing_data,
        conditional=DataSet.cond_data,
        # conditional=torch.cat((DataSet.cond_data, DataSet.input_data), dim=1),
    )

    if BATCH_SIZE == 0:
        train_loader_cvi = DataLoader(DataSet_CVI, batch_size=len(train_idx), sampler=train_sampler)
        test_loader_cvi = DataLoader(DataSet_CVI, batch_size=len(test_idx), sampler=test_sampler)
    else:
        train_loader_cvi = DataLoader(DataSet_CVI, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader_cvi = DataLoader(DataSet_CVI, batch_size=len(test_idx), sampler=test_sampler)

    # Train the model
    model_cvi = CVI(
        input_dim=INPUT_DIM,
        cond_dim=COND_DIM,
        out_dim=MISSING_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        param=PARAM,
        init_noise=NOISE_VI
    )

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

    optimizer = optim.Adam(model_cvi.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[WARM_UP_KL], gamma=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=GRADIENT_DECAY, gamma=0.1)
    train_vi(model_cvi, optimizer, path_test, train_loader_cvi, test_loader_cvi, optim_options, scheduler,
             input_output=True)
    model_cvi.eval()

    # ========================> Regression (GP)
    path_test = os.path.join(results_folder, f"GP_Reg_{KERNEL}_{LR}")

    X = torch.cat((DataSet.input_data, DataSet.missing_data), dim=1)
    Y = DataSet.target_data
    DataSet_Reg = DummyDataset_GP(X, Y)

    if BATCH_SIZE == 0:
        train_loader_reg = DataLoader(DataSet_Reg, batch_size=len(train_idx), sampler=train_sampler)
        test_loader_reg = DataLoader(DataSet_Reg, batch_size=len(test_idx), sampler=test_sampler)
    else:
        train_loader_reg = DataLoader(DataSet_Reg, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader_reg = DataLoader(DataSet_Reg, batch_size=len(test_idx), sampler=test_sampler)

    # Train the model
    gp_model = RegressionGP(init_sigma=SIGMA_GP,
                            init_lamda=LAMDA_GP,
                            init_noise=NOISE_GP,
                            init_mean=MEAN_GP,
                            input_dim=X.shape[1],
                            output_dim=Y.shape[1],
                            kernel=f"{KERNEL}")
    # gp_model.add_prior("sigma", prior={'mean': tensor([0]), 'logvar': tensor([1])})  # Weak prior

    optim_options = {'epochs': EPOCHS,
                     'load_previous': LOAD_PREVIOUS,
                     'print_epochs': PRINT_EPOCHS,
                     'lr': LR,
                     'kernel': KERNEL,
                     'hp_params': {'lr': LR, 'kernel': KERNEL},  # Hyper-parameters to give to the tensorbaord
                     }

    optimizer = optim.Adam(gp_model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=GRADIENT_DECAY, gamma=0.1)
    train_gp(gp_model, optimizer, path_test, train_loader_reg, test_loader_reg, optim_options)
    gp_model.eval()

    # ========================> Median
    median_value = DataSet.missing_data[train_idx].median(dim=0)[0]

    # ========================> Mean
    mean_value = DataSet.missing_data[train_idx].mean(dim=0)

    # ========================> CVI + GP
    cvi_pred = model_cvi.forward(DataSet.input_data[test_idx], DataSet.cond_data[test_idx])
    cvi_pred = cvi_pred[2]

    X_test = torch.cat((DataSet.input_data[test_idx], tensor(cvi_pred)), dim=1)
    cvi_gp_pred, _ = gp_model.predict(X_test)

    # ========================> KNN + GP
    y_pred_knn = knn_neigh.predict(x_input_knn_test)
    X_test = torch.cat((DataSet.input_data[test_idx], tensor(y_pred_knn)), dim=1)
    knn_gp_pred, _ = gp_model.predict(X_test)

    # ==================================================================================================================
    # ========================================= Compute errors =========================================================

    # ========================> Imputation
    methods_list = ["Mean", "Median", "KNN", "CVI", "Ours"]
    mean_imp_errors = pd.DataFrame([], columns=methods_list, index=["MSE"])

    # Squared error
    mean_se = ((DataSet.missing_data[test_idx] - mean_value) ** 2)
    median_se = ((DataSet.missing_data[test_idx] - median_value) ** 2)
    knn_se = ((DataSet.missing_data[test_idx] - y_pred_knn) ** 2)
    cvi_se = ((DataSet.missing_data[test_idx] - cvi_pred) ** 2)
    ours_se = ((DataSet.missing_data[test_idx] - ours_miss) ** 2)

    imp_errors = pd.DataFrame(data=np.concatenate((mean_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   median_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   knn_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   cvi_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                   ours_se.sum(dim=1).unsqueeze(1).cpu().data.numpy()), axis=1),
                              columns=methods_list)
    mean_imp_errors.loc["MSE"] = imp_errors.mean()

    # ========================> Emulation
    methods_list = ["KNN+GP", "CVI+GP", "Ours"]
    mean_emulation_errors = pd.DataFrame([], columns=methods_list, index=["MSE"])

    # Squared error
    knn_gp_se = ((DataSet.target_data[test_idx] - knn_gp_pred) ** 2)
    cvi_gp_se = ((DataSet.target_data[test_idx] - cvi_gp_pred) ** 2)
    ours_em_se = ((DataSet.target_data[test_idx] - ours_target) ** 2)

    emulation_errors = pd.DataFrame(data=np.concatenate((knn_gp_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
                                                         cvi_gp_se.sum(dim=1).unsqueeze(1).cpu().data.numpy(),
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

    err_cvi_dim = pd.DataFrame(cvi_se.cpu().data.numpy(), columns=list_missing_features)
    err_cvi_dim["Index"] = "CVI"

    err_ours_dim = pd.DataFrame(ours_se.cpu().data.numpy(), columns=list_missing_features)
    err_ours_dim["Index"] = "Ours"

    imp_errors_dim = pd.concat([err_mean_dim, err_median_dim, err_knn_dim, err_cvi_dim, err_ours_dim], axis=0)

    err_knn_gp_dim = pd.DataFrame(knn_gp_se.cpu().data.numpy(), columns=list_target_features)
    err_knn_gp_dim["Index"] = "KNN+GP"

    err_cvi_gp_dim = pd.DataFrame(cvi_gp_se.cpu().data.numpy(), columns=list_target_features)
    err_cvi_gp_dim["Index"] = "CVI+GP"

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
    df_pvalues = pd.DataFrame(data=None, columns=["KNN", "CVI"], index=list_missing_features)
    for var in list_missing_features:
        df_pvalues["KNN"].loc[f"{var}"] = scipy.stats.ranksums(err_knn_dim[f"{var}"], err_ours_dim[f"{var}"])[-1]
        df_pvalues["CVI"].loc[f"{var}"] = scipy.stats.ranksums(err_cvi_dim[f"{var}"], err_ours_dim[f"{var}"])[-1]
    mask_sig = df_pvalues < p_thres
    mask_sig.to_csv(os.path.join(results_folder, "imputation_sig.csv"))
    print(mask_sig)

    # Emulation
    p_thres = p_thres / OUTPUT_DIM
    df_pvalues = pd.DataFrame(data=None, columns=["KNN+GP", "CVI+GP"], index=list_target_features)
    for var in list_target_features:
        df_pvalues["KNN+GP"].loc[f"{var}"] = scipy.stats.ranksums(err_knn_gp_dim[f"{var}"], err_ours_em_dim[f"{var}"])[-1]
        df_pvalues["CVI+GP"].loc[f"{var}"] = scipy.stats.ranksums(err_cvi_gp_dim[f"{var}"], err_ours_em_dim[f"{var}"])[-1]
    mask_sig = df_pvalues < p_thres
    mask_sig.to_csv(os.path.join(results_folder, "emulation_sig.csv"))
    print(mask_sig)

