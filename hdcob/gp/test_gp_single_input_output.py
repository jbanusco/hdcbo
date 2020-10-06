from torch import optim
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from hdcob.gp.gaussian_process import GP, GP_Prior
from hdcob.gp.generator_synthetic import SyntheticDataset
from hdcob.gp.plots_gp import plot_predictions, plot_residual, plot_multiple_predictions, plot_multiple_residual, plot_strip
from hdcob.utilities.metrics import mae, mse
from hdcob.config import *
import argparse
from distutils.util import strtobool
import logging
import os

parser = argparse.ArgumentParser(description='Running GP in test data')

parser.add_argument('--save_folder', type=str, default=os.path.join(os.getcwd(), "runs"),
                    help='Folder where to save the results')

parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs')

parser.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate")

parser.add_argument('--kernel', type=str, default="RBF",
                    help="Kernel used in the GP")

parser.add_argument('--print_epochs', type=int, default=100,
                    help="Resume previous optimization if available")

parser.add_argument("--resume", type=strtobool, nargs='?', const=True, default=False,
                    help="Resume previous optimization if available")


def load_data(model, optimizer, filename, scheduler=None):
    loaded_checkpoint = torch.load(filename)
    model.load_state_dict(loaded_checkpoint["model_state"])
    model.to(DEVICE)
    optimizer.load_state_dict(loaded_checkpoint["optim_state"])
    model.load_training_info(loaded_checkpoint["training_info"])  # Just because it is a GP
    model.eval()

    if scheduler is not None:
        scheduler.load_state_dict(loaded_checkpoint["scheduler_state"])

    # Other information
    misc = (loaded_checkpoint["losses"],
            loaded_checkpoint["metrics"]["mae"],
            loaded_checkpoint["metrics"]["mse"],
            loaded_checkpoint["epoch"],
            loaded_checkpoint['time'])
    return misc


def train_gp(model, optimizer, save_path, train_loader, test_loader, options,
             scheduler=None):
    """ Train GP model """
    log = logging.getLogger(LOGGER_NAME)
    log.info("Training GP")

    # Writer where to save the log files
    os.makedirs(save_path, exist_ok=True)

    torch.save(options, os.path.join(save_path, f"optim_options.pth"))
    writer = SummaryWriter(save_path)

    data = iter(train_loader).next()
    x_data = data['x']
    y_data = data['y']
    writer.add_graph(model, [x_data, y_data])
    writer.close()

    losses = list()
    mae_list = list()
    mse_list = list()
    # running_loss = 0
    checkpoint_filename = os.path.join(save_path, "checkpoint.pth")
    final_filename = os.path.join(save_path, "trained_model.pth")

    # This will be used at the end of the training to store a big covaraince matrix that will be used
    # for future predictions [ -- we need to do this we load the data in batches -- ]
    all_x_data_train = tensor([])
    all_y_data_train = tensor([])
    for batch_idx, data in enumerate(train_loader):
        all_x_data_train = torch.cat((all_x_data_train,  data['x'].to(DEVICE)), dim=0)
        all_y_data_train = torch.cat((all_y_data_train,  data['y'].to(DEVICE)), dim=0)

    if os.path.isfile(final_filename) and options['load_previous']:
        losses, mae_list, mse_list, epoch, time_optim = load_data(model, optimizer, final_filename, scheduler)
    else:
        if os.path.isfile(checkpoint_filename) and options['load_previous']:
            losses, mae_list, mse_list, epoch, time_optim = load_data(model, optimizer, checkpoint_filename, scheduler)
        else:
            epoch = 0
            time_optim = 0

        for epoch in range(epoch, options['epochs']):
            model.train()
            init_epoch_t = time.time()

            running_loss = 0
            running_mse = 0
            running_mae = 0
            num_batches = len(train_loader)
            for batch_idx, data in enumerate(train_loader):
                x_data = data['x'].to(DEVICE)
                y_data = data['y'].to(DEVICE)

                # with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                Y_pred = model.forward(x_data, y_data)
                loss = model.loss(Y_pred, y_data)
                loss['total'].backward()
                optimizer.step()

                running_mae += mae(Y_pred[0], y_data)
                running_mse += mse(Y_pred[0], y_data)
                running_loss += loss['total'].item()

            if scheduler is not None:
                scheduler.step()

            # Save the metrics
            running_mae /= num_batches; running_mse /= num_batches; running_loss /= num_batches
            losses.append(running_loss)
            mae_list.append(running_mae)
            mse_list.append(running_mse)

            time_optim += (time.time() - init_epoch_t)

            if (epoch + 1) % int(options['print_epochs']) == 0:
                # Metrics
                log.info(f"\nEpoch: {epoch + 1}\tLoss: {running_loss:.6f}\tMSE: {running_mse:.6f}\tMAE: {running_mae:.6f}")

                # Parameters
                [log.debug(f"{key}: {model.state_dict()[key].detach().numpy()[0]:.6f}") for key in model.state_dict()]

                if hasattr(model, 'list_parameters'):
                    for key in model.list_parameters.keys():
                        log.info(f"\tLogvar {key}: {model.list_parameters[f'{key}'].detach().numpy()}\n")

                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "losses": losses,
                    "metrics": {'mae': mae_list, 'mse': mse_list},
                    "training_info": model.save_training_info(),
                    "location": DEVICE,
                    "time": time_optim
                }
                if scheduler is not None:
                    checkpoint["scheduler_state"] = scheduler.state_dict()
                torch.save(checkpoint, checkpoint_filename)
                model.eval()

                # Add it to the tensorboard
                for key in loss.keys():
                    writer.add_scalar(f'{key}', loss[f"{key}"], epoch)
                writer.add_scalar('MSE', running_mse, epoch)
                writer.add_scalar('MAE', running_mae, epoch)

                # Try histogram
                for name, w in model.named_parameters():
                    writer.add_histogram(name, w, epoch)

                # Add to the tensorboard and save it
                data = iter(test_loader).next()
                x_data_test = data['x'].to(DEVICE)
                y_data_test = data['y'].to(DEVICE)
                y_rec, _ = model.predict(x_data_test)

                if len(y_data_test.shape) == 1: y_data_test = y_data_test.unsqueeze(1)

                if y_data_test.shape[1] > 1:
                        fig_pred, ax_pred = plot_multiple_predictions(model, x_data_test, y_data_test, num_samples=0, shade=True)
                else:
                    fig_pred, ax_pred = plot_predictions(model, x_data_test, y_data_test, num_samples=0, shade=True)
                writer.add_figure('predictions', fig_pred, global_step=epoch + 1)

                if y_data_test.shape[1] > 1:
                    fig_res, ax_res = plot_multiple_residual(model, x_data_test, y_data_test)
                else:
                    fig_res, ax_res = plot_residual(model, x_data_test, y_data_test)
                writer.add_figure('residuals', fig_res, global_step=epoch + 1)

                # Strip plot
                fig_strip = plot_strip(y_data_test, y_rec, x_names=None, y_names=None, save_path=None)
                writer.add_figure('strip', fig_strip, global_step=epoch + 1)

                writer.close()
                plt.close('all')

        # Get all the training data
        model.train()
        _ = model.forward(all_x_data_train, all_y_data_train)
        model.eval()

        # Save the final model
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "losses": losses,
            "metrics": {'mae': mae_list, 'mse': mse_list},
            "training_info": model.save_training_info(),
            "location": DEVICE,
            "time": time_optim
        }
        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()
        torch.save(checkpoint, final_filename)

    # Try in the test model
    data = iter(test_loader).next()
    x_data = data['x'].to(DEVICE)
    y_data = data['y'].to(DEVICE)

    y_pred, cov_pred = model.predict(x_data)
    test_mae = mae(y_pred, y_data)
    test_mse = mse(y_pred, y_data)

    writer.add_hparams(options['hp_params'],
                       {'mse': mse_list[-1], 'mae': mae_list[-1], 'loss': losses[-1],
                        'test_mse': test_mse, 'test_mae': test_mae})

    # Try histogram
    for name, w in model.named_parameters():
        writer.add_histogram(name, w, epoch)

    # Add to the tensorboard and save it
    if len(y_data.shape) == 1: y_data = y_data.unsqueeze(1)

    fig_pred, ax_pred = plot_multiple_predictions(model, x_data, y_data, num_samples=0, shade=True)
    writer.add_figure('predictions', fig_pred, global_step=options['epochs'])

    # if y_data.shape[1] > 1:
    fig_res, ax_res = plot_multiple_residual(model, x_data, y_data)
    # else:
    #     fig_res, ax_res = plot_residual(model, x_data, y_data)
    writer.add_figure('residuals', fig_res, global_step=options['epochs'])

    # Strip plot
    fig_strip = plot_strip(y_data, y_pred, x_names=None, y_names=None, save_path=None)
    writer.add_figure('strip', fig_strip, global_step=epoch + 1)

    writer.close()
    fig_res.savefig(os.path.join(save_path, "residuals.png"))
    fig_pred.savefig(os.path.join(save_path, "prediction.png"))
    fig_strip.savefig(os.path.join(save_path, "strip.png"))

    # Plot the loss
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Loss")
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(os.path.join(save_path, "loss.png"))

    log.info(f" ===> FINISHED in {time_optim:.3f} <===")

    log.info(f"\n====> Final Values <====\n")
    [log.info(f"{key}: {model.state_dict()[key].detach().numpy()[0]:.6f}") for key in model.state_dict()]


if __name__ == '__main__':
    """ Test that our simple GP implementation works """
    SEED = 5
    parsed_args = parser.parse_args()

    log = logging.getLogger(LOGGER_NAME)
    log.info("Test GP")

    runs_path = parsed_args.save_folder
    os.makedirs(runs_path, exist_ok=True)

    # Options
    EPOCHS = parsed_args.epochs
    PRINT_EPOCHS = parsed_args.print_epochs
    LR = parsed_args.lr
    LOAD_PREVIOUS = bool(parsed_args.resume)
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

    # Initial conditions
    NUM_SAMPLES = 500
    DIM_X = 1
    DIM_Y = 1

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
    # -------------------------------------------- Test simple GP ------------------------------------------------------
    path_test = os.path.join(runs_path, f"test_gp_{KERNEL}_{LR}")

    model = GP(init_noise=-5, kernel=KERNEL)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_gp(model, optimizer, path_test, train_loader, test_loader, optim_options)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Test priors  --------------------------------------------------------
    # ==> Lamda
    var_prior = "lamda"
    path_test = os.path.join(runs_path, f"test_gp_{KERNEL}_Prior{var_prior}_{LR}")

    model = GP_Prior(init_noise=NOISE, kernel=f"{KERNEL}")
    model.add_prior(f"{var_prior}", prior={'mean': tensor([0.7]), 'logvar': tensor([0])})
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_gp(model, optimizer, path_test, train_loader, test_loader, optim_options)

    # ==> Sigma
    var_prior = "sigma"
    path_test = os.path.join(runs_path, f"test_gp_{KERNEL}_Prior{var_prior}_{LR}")

    model = GP_Prior(init_noise=NOISE, kernel=f"{KERNEL}")
    model.add_prior(f"{var_prior}", prior={'mean': tensor([0]), 'logvar': tensor([-2])})
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_gp(model, optimizer, path_test, train_loader, test_loader, optim_options)

    # ==> Noise
    var_prior = "noise"
    path_test = os.path.join(runs_path, f"test_gp_{KERNEL}_Prior{var_prior}_{LR}")

    model = GP_Prior(init_noise=NOISE, kernel=f"{KERNEL}")
    model.add_prior(f"{var_prior}", prior={'mean': tensor([-10]), 'logvar': tensor([0])})
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_gp(model, optimizer, path_test, train_loader, test_loader, optim_options)


