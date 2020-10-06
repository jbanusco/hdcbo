from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from hdcob.gp.gaussian_process import RegressionGP
from hdcob.gp.test_gp_single_input_output import train_gp
from hdcob.gp.generator_synthetic import SyntheticDataset
from hdcob.config import *
import argparse
import logging
import os
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='Running GP in test data')

parser.add_argument('--save_folder', type=str, default=os.path.join(os.getcwd(), "runs"),
                    help='Folder where to save the results')

parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs')

parser.add_argument('--lr', type=float, default=1e-2,
                    help="Learning rate")

parser.add_argument('--kernel', type=str, default="RBF_ML",
                    help="Kernel used in the GP")

parser.add_argument('--print_epochs', type=int, default=100,
                    help="Resume previous optimization if available")

parser.add_argument("--resume", type=strtobool, nargs='?', const=True, default=False,
                    help="Resume previous optimization if available")

if __name__ == '__main__':
    """ Test that our multi-dimensional GP implementation works """
    SEED = 5
    parsed_args = parser.parse_args()

    log = logging.getLogger(LOGGER_NAME)
    log.info("Test GP multi dimensional")

    runs_path = os.path.join(os.getcwd(), "runs")
    os.makedirs(runs_path, exist_ok=True)

    # Options
    EPOCHS = parsed_args.epochs
    PRINT_EPOCHS = parsed_args.print_epochs
    LR = parsed_args.lr
    LOAD_PREVIOUS = parsed_args.resume
    KERNEL = parsed_args.kernel
    optim_options = {'epochs': EPOCHS,
                     'load_previous': LOAD_PREVIOUS,
                     'print_epochs': PRINT_EPOCHS,
                     'lr': LR,
                     'kernel': KERNEL,
                     'hp_params': {'lr': LR, 'kernel': KERNEL},  # Hyper-parameters to give to the tensorbaord
                     'save_folder': runs_path}

    log.info("==> Optimization options <==")
    for key in optim_options.keys():
        log.info(f"{key}: {optim_options[f'{key}']}")

    # Multiple input features to predict one output
    NUM_SAMPLES = 500
    DIM_X = 3
    DIM_Y = 2

    np.random.seed(SEED)  # Just in case we use shuffle option - reproducibility
    torch.manual_seed(SEED)
    DataSet = SyntheticDataset(NUM_SAMPLES, DIM_X, DIM_Y, SEED)

    # Define the samplers
    indices = list(range(0, NUM_SAMPLES))
    split = int(np.floor(NUM_SAMPLES * 0.33))
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Standardize the data based on the training set
    mean_data_x = DataSet.x[train_idx].mean(dim=0)
    std_data_x = DataSet.x[train_idx].std(dim=0)
    DataSet.x = (DataSet.x - mean_data_x) / std_data_x

    mean_data_y = DataSet.y[train_idx].mean(dim=0)
    std_data_y = DataSet.y[train_idx].std(dim=0)
    DataSet.y = (DataSet.y - mean_data_y) / std_data_y

    # Get the data loaders
    train_loader = DataLoader(DataSet, batch_size=len(train_idx), sampler=train_sampler)
    test_loader = DataLoader(DataSet, batch_size=len(test_idx), sampler=test_sampler)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Test multiple d. gps  ---------------------------------------------------

    path_test = os.path.join(runs_path, f"test_gp_multiple_output_{KERNEL}_{LR}")

    # Train the model
    model = RegressionGP(init_sigma=1,
                         init_lamda=0,
                         init_noise=1,
                         init_mean=0,
                         input_dim=DIM_X,
                         output_dim=DIM_Y,
                         kernel=f"{KERNEL}")
    # model.add_prior("lamda", prior={'mean': tensor([0.7]), 'logvar': tensor([0])})
    # model.add_prior("noise", prior={'mean': tensor([-10]), 'logvar': tensor([0])})
    # model.add_prior("sigma", prior={'mean': tensor([0]), 'logvar': tensor([-2])})

    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_gp(model, optimizer, path_test, train_loader, test_loader, optim_options)

    # print(DataSet.generator.lamdas)
    # print(DataSet.generator.sigma)
    # print(DataSet.generator.noise)
