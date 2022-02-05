import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from hdcob.utilities.metrics import mae, mse
from hdcob.vi.plots_vi import plot_residual, plot_latent, plot_predictions, plot_strip
from hdcob.vi.vi_models import VAE, CVAE, ICVAE
from hdcob.vi.generator_synthetic import SyntheticDataset
from hdcob.config import *

import argparse
import matplotlib.pyplot as plt
from distutils.util import strtobool


parser = argparse.ArgumentParser(description='Running GP in test data')

parser.add_argument('--save_folder', type=str, default=os.path.join(os.getcwd(), "runs"),
                    help='Folder where to save the results')

parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs')

parser.add_argument('--latent', type=int, default=1,
                    help="Latent dimensions")

parser.add_argument('--hidden', type=int, default=0,
                    help="Hidden dimensions")

parser.add_argument('--batch_size', type=int, default=0,
                    help="Batch size, if 0 it means all the dataset")

parser.add_argument('--lr', type=float, default=1e-2,
                    help="Learning rate")

parser.add_argument('--epochs_kl', type=int, default=200,
                    help='Number of epochs to warm-up KL')

parser.add_argument('--print_epochs', type=int, default=100,
                    help="Resume previous optimization if available")

parser.add_argument("--resume", type=strtobool, nargs='?', const=True, default=False,
                    help="Resume previous optimization if available")

parser.add_argument("--use_param", type=strtobool, nargs='?', const=True, default=False,
                    help="Use parameter for variance of the decoding distribution")


def load_data(model, optimizer, filename, scheduler = None):
    loaded_checkpoint = torch.load(filename)
    model.load_state_dict(loaded_checkpoint["model_state"])
    model.to(DEVICE)
    optimizer.load_state_dict(loaded_checkpoint["optim_state"])
    model.load_training_info(loaded_checkpoint["training_info"])
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


def train_vi(model, optimizer, save_path, train_loader, test_loader, options,
             scheduler=None):
    """ Train variational inference model """
    log = logging.getLogger(LOGGER_NAME)
    log.info("Training VI")

    # Writer where to save the log files
    os.makedirs(save_path, exist_ok=True)
    torch.save(options, os.path.join(save_path, f"optim_options.pth"))
    writer = SummaryWriter(save_path)

    # Don't use prob. functions to be able to trace back
    data = iter(train_loader).next()
    input_data = data['input'].to(DEVICE)
    if type(model) == VAE:
        writer.add_graph(model, [input_data])
    elif type(model) == CVAE or type(model) == ICVAE:
        cond_data = data['conditional'].to(DEVICE)
        writer.add_graph(model, [input_data, cond_data])
    else:
        raise RuntimeError("Unexpected type of model")
    writer.close()

    losses = list()
    mae_list = list()
    mse_list = list()
    checkpoint_filename = os.path.join(save_path, "checkpoint.pth")
    final_filename = os.path.join(save_path, "trained_model.pth")

    if os.path.isfile(final_filename) and options['load_previous']:
        losses, mae_list, mse_list, epoch, time_optim = load_data(model, optimizer, final_filename, scheduler)
    else:
        if os.path.isfile(checkpoint_filename) and options['load_previous']:
            losses, mae_list, mse_list, epoch, time_optim = load_data(model, optimizer, checkpoint_filename, scheduler)
        else:
            epoch = 0
            time_optim = 0

        total_epochs = options.get('warm_up_kl', 0) + options.get('epochs', 0)
        for epoch in range(epoch, total_epochs):
            model.train()
            init_epoch_t = time.time()

            running_loss = 0
            running_mse = 0
            running_mae = 0
            num_batches = len(train_loader)
            for batch_idx, data in enumerate(train_loader):
                recon_data = data['input'].to(DEVICE)
                if type(model) == VAE:
                    input_data = [recon_data]
                elif type(model) == CVAE or type(model) == ICVAE:
                    cond_data = data['conditional'].to(DEVICE)
                    input_data = [recon_data, cond_data]
                else:
                    raise RuntimeError(f"Unexpected model type: {type(model)}")

                rec_data = model.forward(*input_data)
                # with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                loss = model.loss(rec_data, recon_data)
                if epoch < options.get('warm_up_kl', 0):
                    loss['ll'].backward()
                else:
                    loss['total'].backward()
                optimizer.step()

                running_mae += mae(rec_data[2], recon_data)
                running_mse += mse(rec_data[2], recon_data)
                running_loss += loss['total'].item()

            if scheduler is not None:
                scheduler.step()

            # Save the metrics
            running_mae /= num_batches; running_mse /= num_batches; running_loss /= num_batches
            losses.append(running_loss)
            mae_list.append(running_mae)
            mse_list.append(running_mse)

            # Update time
            time_optim += (time.time() - init_epoch_t)

            # Print / Save info
            if (epoch + 1) % int(options['print_epochs']) == 0:
                # Metrics
                log.info(f"\nEpoch: {epoch + 1}\tLoss: {running_loss:.6f}\tMSE: {running_mse:.6f}\tMAE: {running_mae:.6f}")
                [print(f'{key}: {loss[f"{key}"]}') for key in loss.keys()]
                # Add it to the tensorboard
                for key in loss.keys():
                    writer.add_scalar(f'{key}', loss[f"{key}"], epoch)
                writer.add_scalar('MSE', running_mse, epoch)
                writer.add_scalar('MAE', running_mae, epoch)

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
                    checkpoint["scheduler_state"] =  scheduler.state_dict()
                torch.save(checkpoint, checkpoint_filename)

                # Plot results with the test data and add them to the tensorboard
                with torch.no_grad():
                    model.eval()

                    data = iter(test_loader).next()
                    if type(model) == VAE:
                        input_data = [data['input'].to(DEVICE)]
                    elif type(model) == CVAE or type(model) == ICVAE:
                        input_data = [data['input'].to(DEVICE), data['conditional'].to(DEVICE)]
                    else:
                        raise RuntimeError(f"Unexpected model type: {type(model)}")

                    fig_lat = plot_latent(model, input_data, save_path=None)
                    fig_res = plot_residual(model, input_data, save_path=None)
                    fig_pred = plot_predictions(model, input_data, save_path=None)
                    fig_strip = plot_strip(model, input_data, save_path=None)

                    writer.add_figure('strip', fig_strip, global_step=epoch + 1)
                    writer.add_figure('latent', fig_lat, global_step=epoch + 1)
                    writer.add_figure('prediction', fig_pred, global_step=epoch + 1)
                    writer.add_figure('residual', fig_res, global_step=epoch + 1)
                    writer.close()

                    plt.close('all')

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

    # Save the final results in the test model
    model.eval()
    data = iter(test_loader).next()
    recon_data_test = data['input'].to(DEVICE)
    if type(model) == VAE:
        input_data = [data['input'].to(DEVICE)]
    elif type(model) == CVAE or type(model) == ICVAE:
        input_data = [data['input'].to(DEVICE), data['conditional'].to(DEVICE)]
    else:
        raise RuntimeError(f"Unexpected model type: {type(model)}")

    rec_data = model.forward(*input_data)
    test_mae = mae(rec_data[2], recon_data_test)
    test_mse = mse(rec_data[2], recon_data_test)

    writer.add_hparams(options['hp_params'],
                       {'mse': mse_list[-1], 'mae': mae_list[-1], 'loss': losses[-1],
                        'test_mse': test_mse, 'test_mae': test_mae})

    # Try histogram
    # for name, w in model.named_parameters():
    #     writer.add_histogram(name, w, epoch)

    # Residual's boxplot
    fig_lat = plot_latent(model, input_data, save_path=os.path.join(save_path, "latent.png"))
    fig_res = plot_residual(model, input_data, save_path=os.path.join(save_path, "residuals.png"))
    fig_pred = plot_predictions(model, input_data, save_path=os.path.join(save_path, "prediction.png"))
    fig_strip = plot_strip(model, input_data, save_path=os.path.join(save_path, "strip_plot.png"))

    writer.add_figure('strip', fig_strip, global_step=epoch + 1)
    writer.add_figure('latent', fig_lat, global_step=epoch + 1)
    writer.add_figure('prediction', fig_pred, global_step=epoch + 1)
    writer.add_figure('residual', fig_res, global_step=epoch + 1)

    # Plot the loss
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Loss")
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(os.path.join(save_path, "loss.png"))

    log.info(f"\n====> OPTIMIZATION FINISHED <====\n")
    log.info(f"\nTraining: \tLoss: {losses[-1]:.6f}\tMSE: {mse_list[-1]:.6f}\tMAE: {mae_list[-1]:.6f}"
             f"\nTest: \tMSE: {test_mse:.6f}\tMAE: {test_mae:.6f}")


if __name__ == '__main__':
    """ Test if imnplementation of VAE and CVAE works """

    SEED = 5  # For reproducibility
    parsed_args = parser.parse_args()

    log = logging.getLogger(LOGGER_NAME)
    log.info("Test VI")

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

    log.info("==> Optimization options <==")
    for key in optim_options.keys():
        log.info(f"{key}: {optim_options[f'{key}']}")

    # Initial conditions
    INIT_NOISE = -3  # Used when we learn the variance as a parameter
    NUM_SAMPLES = 500
    INPUT_DIM = 3
    COND_DIM = 2

    np.random.seed(SEED)  # Just in case we use shuffle option - reproducibility
    torch.manual_seed(SEED)
    DataSet = SyntheticDataset(NUM_SAMPLES, LATENT_DIM, INPUT_DIM, COND_DIM, SEED)

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
    if COND_DIM > 0:
        mean_data_cond = DataSet.cond[train_idx].mean(dim=0)
        std_data_cond = DataSet.cond[train_idx].std(dim=0)
        DataSet.cond = (DataSet.cond - mean_data_cond) / std_data_cond

    # Get the data loaders
    if BATCH_SIZE == 0:
        train_loader = DataLoader(DataSet, batch_size=len(train_idx), sampler=train_sampler)
        test_loader = DataLoader(DataSet, batch_size=len(test_idx), sampler=test_sampler)
    else:
        train_loader = DataLoader(DataSet, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(DataSet, batch_size=len(test_idx), sampler=test_sampler)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- Test simple VAE ------------------------------------------------------
    path_test = os.path.join(runs_path, f"test_vae_L{LATENT_DIM}_H{HIDDEN_DIM}_P{PARAM}_{LR}")

    model = VAE(input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM,
                latent_dim=LATENT_DIM,
                param=PARAM,
                init_noise=INIT_NOISE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_vi(model, optimizer, path_test, train_loader, test_loader, optim_options)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- Test simple CVAE -------------------------------------------------------
    path_test = os.path.join(runs_path, f"test_cvae_L{LATENT_DIM}_H{HIDDEN_DIM}_P{PARAM}_{LR}")

    model = CVAE(input_dim=INPUT_DIM,
                 cond_dim=COND_DIM,
                 hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM,
                 param=PARAM,
                 init_noise=INIT_NOISE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_vi(model, optimizer, path_test, train_loader, test_loader, optim_options)
