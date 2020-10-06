from torch import optim
import time
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from hdcob.vi.plots_vi import plot_latent
from hdcob.gp.plots_gp import plot_ard
from hdcob.virca.plots_virca import plot_residual, plot_predictions, plot_boxplot_params, plot_correlations
from hdcob.virca.generator_synthetic import SyntheticDataset
from hdcob.virca.virca import VIRCA
from hdcob.utilities.metrics import mae, mse
from hdcob.config import *
import argparse
import matplotlib.pyplot as plt
from distutils.util import strtobool
import logging
import os

parser = argparse.ArgumentParser(description='Running joint model in test data')

parser.add_argument('--save_folder', type=str, default=os.path.join(os.getcwd(), "runs"),
                    help='Folder where to save the results')

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs')

parser.add_argument('--latent', type=int, default=1,
                    help="Latent dimensions")

parser.add_argument('--hidden', type=int, default=0,
                    help="Hidden dimensions")

parser.add_argument('--batch_size', type=int, default=32,
                    help="Batch size, if 0 it means all the dataset")

parser.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate")

parser.add_argument('--epochs_kl', type=int, default=30,
                    help='Number of epochs to warm-up KL')

parser.add_argument('--kernel', type=str, default="RBF_ML",
                    help="Kernel used in the GP: RBF, Lin, RBF_M, RBF_ML, RBF_MN, Lin_MC")

parser.add_argument('--print_epochs', type=int, default=50,
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
    model.load_training_info(loaded_checkpoint["training_info"])  # Just because it is a GP
    model.eval()

    if scheduler is not None:
        scheduler.load_state_dict(loaded_checkpoint["scheduler_state"])

    # Other information
    misc = (loaded_checkpoint["losses"],
            loaded_checkpoint["gp_loss"],
            loaded_checkpoint["vi_loss"],
            loaded_checkpoint["metrics"]["mae_missing"],
            loaded_checkpoint["metrics"]["mse_missing"],
            loaded_checkpoint["metrics"]["mae_target"],
            loaded_checkpoint["metrics"]["mse_target"],
            loaded_checkpoint["epoch"],
            loaded_checkpoint['time'])
    return misc


def train_virca(model, optimizer, save_path, train_loader, test_loader, options,
                 scheduler=None):
    """ Train variational inference model """
    log = logging.getLogger(LOGGER_NAME)
    log.info("Training VIRCA")

    # Writer where to save the log files
    os.makedirs(save_path, exist_ok=True)
    torch.save(options, os.path.join(save_path, f"optim_options.pth"))
    writer = SummaryWriter(save_path)

    # Get type of class
    data = iter(train_loader).next()
    input_data = data['input'].to(DEVICE)
    conditional_data = data['conditional'].to(DEVICE)
    missing_data = data['missing'].to(DEVICE)
    target_data = data['target'].to(DEVICE)

    writer.add_graph(model, [input_data, target_data, missing_data, conditional_data])
    writer.close()

    losses = list();
    vi_loss = list(); gp_loss = list();
    mae_list_missing = list(); mse_list_missing = list()
    mae_list_target = list(); mse_list_target = list()

    # This will be used at the end of the training to store a big covaraince matrix that will be used
    # for future predictions [ -- we need to do this we load the data in batches -- ]
    all_input_data_train = tensor([])
    all_conditional_data_train = tensor([])
    all_missing_data_train = tensor([])
    all_target_data_train = tensor([])
    for batch_idx, data in enumerate(train_loader):
        all_input_data_train = torch.cat((all_input_data_train,  data['input'].to(DEVICE)), dim=0)
        all_conditional_data_train = torch.cat((all_conditional_data_train,  data['conditional'].to(DEVICE)), dim=0)
        all_missing_data_train = torch.cat((all_missing_data_train,  data['missing'].to(DEVICE)), dim=0)
        all_target_data_train = torch.cat((all_target_data_train,  data['target'].to(DEVICE)), dim=0)

    checkpoint_filename = os.path.join(save_path, "checkpoint.pth")
    final_filename = os.path.join(save_path, "trained_model.pth")

    if os.path.isfile(final_filename) and options['load_previous']:
        losses, gp_loss, vi_loss, mae_list_missing, mse_list_missing, mae_list_target, mse_list_target, \
        epoch, time_optim = load_data(model, optimizer, final_filename, scheduler)
    else:
        if os.path.isfile(checkpoint_filename) and options['load_previous']:
            losses, gp_loss, vi_loss, mae_list_missing, mse_list_missing, mae_list_target, mse_list_target, \
            epoch, time_optim = load_data(model, optimizer, checkpoint_filename, scheduler)
        else:
            epoch = 0
            time_optim = 0

        total_epochs = options.get('warm_up_kl', 0) + options.get('epochs', 0)
        num_batches = len(train_loader)
        for epoch in range(epoch, total_epochs):
            model.train()
            init_epoch_t = time.time()

            running_loss = 0
            running_vi_loss = 0
            running_gp_loss = 0
            running_mse_missing = 0
            running_mae_missing = 0
            running_mse_target = 0
            running_mae_target = 0

            if epoch < options.get('warm_up_kl', 0):  # Train only in LL
                model._VI.training = True
                model._GP.training = False
            else:
                model._VI.training = True
                model._GP.training = True

            for batch_idx, data in enumerate(train_loader):
                model.train()

                input_data = data['input'].to(DEVICE)
                conditional_data = data['conditional'].to(DEVICE)
                missing_data = data['missing'].to(DEVICE)  # Imputation using VI - input + conditional
                target_data = data['target'].to(DEVICE)  # Prediction using GP regression - missing + input

                # with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                output = model.forward(input_data, target_data,
                                       x_imp=missing_data,
                                       x_cond=conditional_data)
                loss_dict = model.loss(missing_data, target_data, output)

                if epoch < options.get('warm_up_kl', 0):  # Train only in LL
                    loss_dict['ll_vi'].backward()
                else:
                    loss_dict['total'].backward()

                running_loss += loss_dict['total'].item()
                running_vi_loss += loss_dict['total_vi'].item()
                running_gp_loss += loss_dict['total_gp'].item()

                # print(f"==>Losses<==\n")
                # for key in loss_dict.keys():
                #     print(f"{key}', {loss_dict[f'{key}']}")

                optimizer.step()

                running_mae_missing += mae(output[2], missing_data)
                running_mse_missing += mse(output[2], missing_data)
                running_mae_target += mae(output[4], target_data)
                running_mse_target += mse(output[4], target_data)

            if scheduler is not None:
                scheduler.step()

            # Save the metrics
            running_mae_missing /= num_batches; running_mse_missing /= num_batches;
            running_mae_target /= num_batches; running_mse_target /= num_batches;
            running_loss /= num_batches; running_vi_loss /= num_batches; running_gp_loss /= num_batches

            losses.append(running_loss); gp_loss.append(running_gp_loss); vi_loss.append(running_vi_loss)
            mae_list_missing.append(running_mae_missing); mse_list_missing.append(running_mse_missing)
            mae_list_target.append(running_mae_target); mse_list_target.append(running_mse_target)

            time_optim += (time.time() - init_epoch_t)

            if (epoch + 1) % int(options['print_epochs']) == 0:
                # Metrics
                log.info(f"\n==> Summary <==\n"
                         f"Epoch: {epoch + 1}\tLoss: {running_loss:.3f}\n"
                         f"==> Missing: MSE: {running_mse_missing:.3f}\tMAE: {running_mae_missing:.3f}\n"
                         f"==> Target: MSE: {running_mse_target:.3f}\tMAE: {running_mae_target:.3f}\n")

                # Parameters
                # Maybe divide, between GP and VI ?
                [log.debug(f"{key}: {model.state_dict()[key].detach().numpy().squeeze()}") for key in model.state_dict()]

                # Losses
                log.info(f"==>Losses<==")
                for key in loss_dict.keys():
                    log.info(f"{key}', {loss_dict[f'{key}'].item():.3f}")

                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "losses": losses,
                    "gp_loss": gp_loss,
                    "vi_loss": vi_loss,
                    "metrics": {'mae_missing': mae_list_missing, 'mse_missing': mse_list_missing,
                                'mae_target': mae_list_target, 'mse_target': mse_list_target},
                    "training_info": model.save_training_info(),
                    "location": DEVICE,
                    "time": time_optim
                }
                if scheduler is not None:
                    checkpoint["scheduler_state"] =  scheduler.state_dict()
                torch.save(checkpoint, checkpoint_filename)

                # Add it to the tensorboard
                for key in loss_dict.keys():
                    writer.add_scalar(f'{key}', loss_dict[f"{key}"], epoch)
                writer.add_scalar('Loss', running_loss, epoch)
                writer.add_scalar('MSE missing', running_mse_missing, epoch)
                writer.add_scalar('MAE missing', running_mae_missing, epoch)
                writer.add_scalar('MSE target', running_mse_target, epoch)
                writer.add_scalar('MAE target', running_mae_target, epoch)

                # Try histogram
                for name, w in model.named_parameters():
                    writer.add_histogram(name, w, epoch)

                # Get the training data
                with torch.no_grad():
                    output = model.forward(all_input_data_train, all_target_data_train,
                                           x_imp=all_missing_data_train,
                                           x_cond=all_conditional_data_train)

                # Add to the tensorboard and save it
                model.eval()
                data = iter(test_loader).next()
                input_data = data['input'].to(DEVICE)
                conditional_data = data['conditional'].to(DEVICE)
                missing_data = data['missing'].to(DEVICE)  # Imputation using VI - input + conditional
                target_data = data['target'].to(DEVICE)  # Prediction using GP regression - missing + input

                pred_target, pred_cov, pred_missing = model.predict(input_data, conditional_data)
                fig_box = plot_boxplot_params(input_data, pred_missing, pred_target,
                                              ground_missing=missing_data,
                                              ground_target=target_data,
                                              names_input=None, names_missing=None, names_target=None)

                if model.condition:
                    fig_lat = plot_latent(model._VI,
                                          [missing_data, torch.cat((conditional_data, input_data), dim=1)],
                                          # [missing_data, conditional_data],
                                          save_path=None)
                else:
                    fig_lat = plot_latent(model._VI, [missing_data], save_path=None)

                fig_res_vi, fig_res_gp = plot_residual(model, input_data, target_data, x_cond=conditional_data,
                                                       x_miss=missing_data, save_folder=None)

                fig_pred_vi, fig_pred_gp = plot_predictions(model, input_data, target_data, x_cond=conditional_data,
                                                            x_miss=missing_data, save_folder=None, shade=True)

                fig_corr_vi, fig_corr_gp = plot_correlations(model, input_data, target_data, x_cond=conditional_data,
                                                             x_miss=missing_data, save_folder=None)

                if 'lamda' in model._GP._GP[0].multiple_params:
                    fig_ard = plot_ard(model._GP, input_names=None, target_names=None)
                    writer.add_figure('ARD', fig_ard, global_step=epoch + 1)

                writer.add_figure('boxplot', fig_box.fig, global_step=epoch + 1)
                writer.add_figure('latent', fig_lat, global_step=epoch + 1)

                writer.add_figure('prediction VI', fig_pred_vi, global_step=epoch + 1)
                writer.add_figure('prediction GP', fig_pred_gp, global_step=epoch + 1)

                writer.add_figure('residual VI', fig_res_vi, global_step=epoch + 1)
                writer.add_figure('residual GP', fig_res_gp, global_step=epoch + 1)

                writer.add_figure('correlation VI', fig_corr_vi, global_step=epoch + 1)
                writer.add_figure('correlation GP', fig_corr_gp, global_step=epoch + 1)

                writer.close()

                plt.close('all')

            # torch.save({'losses': losses}, os.path.join(path_test, "losses.pth"))
            # If used in a model, it saves the WHOLE model

        # Get all the training data
        model.train()
        output = model.forward(all_input_data_train, all_target_data_train,
                               x_imp=all_missing_data_train,
                               x_cond=all_conditional_data_train)

        # Save the final model
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "losses": losses,
            "gp_loss": gp_loss,
            "vi_loss": vi_loss,
            "metrics": {'mae_missing': mae_list_missing, 'mse_missing': mse_list_missing,
                        'mae_target': mae_list_target, 'mse_target': mse_list_target},
            "training_info": model.save_training_info(),
            "location": DEVICE,
            "time": time_optim
        }
        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()
        torch.save(checkpoint, final_filename)

    # Problem with .save and .load is that the serialization is bound to the specific classes and the exact directory
    # structure used when the model is saved
    # load_state_dict, loads only the parameters, we can do the same with the model

    # Try in the test model
    model.eval()
    data = iter(test_loader).next()
    input_data = data['input'].to(DEVICE)
    conditional_data = data['conditional'].to(DEVICE)
    missing_data = data['missing'].to(DEVICE)  # Imputation using VI - input + conditional
    target_data = data['target'].to(DEVICE)  # Prediction using GP regression - missing + input

    pred_target, pred_cov, pred_missing = model.predict(input_data, conditional_data)
    fig_box = plot_boxplot_params(input_data, pred_missing, pred_target,
                                  ground_missing=missing_data,
                                  ground_target=target_data,
                                  names_input=None, names_missing=None, names_target=None)
    fig_box.savefig(os.path.join(save_path, "boxplot_params.png"))

    if model.condition:
        fig_lat = plot_latent(model._VI,
                              [missing_data, torch.cat((conditional_data, input_data), dim=1)],
                              # [missing_data, conditional_data],
                              save_path=os.path.join(save_path, "latent.png"))
    else:
        fig_lat = plot_latent(model._VI, [missing_data], save_path=os.path.join(save_path, "latent.png"))

    fig_res_vi, fig_res_gp = plot_residual(model, input_data, target_data, x_cond=conditional_data,
                                           x_miss=missing_data, save_folder=save_path)

    fig_pred_vi, fig_pred_gp = plot_predictions(model, input_data, target_data, x_cond=conditional_data,
                                                x_miss=missing_data, save_folder=save_path, shade=True)

    fig_corr_vi, fig_corr_gp = plot_correlations(model, input_data, target_data, x_cond=conditional_data,
                                                 x_miss=missing_data, save_folder=save_path)

    if 'lamda' in model._GP._GP[0].multiple_params:
        fig_ard = plot_ard(model._GP, input_names=None, target_names=None, save_path=os.path.join(save_path, "ARD.png"))
        writer.add_figure('ARD', fig_ard, global_step=epoch + 1)

    # Try histogram
    for name, w in model.named_parameters():
        writer.add_histogram(name, w, epoch)

    # Residual's boxplot
    writer.add_figure('boxplot', fig_box.fig, global_step=epoch + 1)
    writer.add_figure('latent', fig_lat, global_step=epoch + 1)

    writer.add_figure('prediction VI', fig_pred_vi, global_step=epoch + 1)
    writer.add_figure('prediction GP', fig_pred_gp, global_step=epoch + 1)

    writer.add_figure('residual VI', fig_res_vi, global_step=epoch + 1)
    writer.add_figure('residual GP', fig_res_gp, global_step=epoch + 1)

    writer.add_figure('correlation VI', fig_corr_vi, global_step=epoch + 1)
    writer.add_figure('correlation GP', fig_corr_gp, global_step=epoch + 1)

    # Plot the loss
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Losses")

    color = 'tab:red'
    ax.plot(vi_loss, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VI Loss", color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('GP Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(gp_loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.savefig(os.path.join(save_path, "losses.png"))

    # Hyperparams
    test_mae_missing = mae(pred_missing, missing_data)
    test_mse_missing = mse(pred_missing, missing_data)
    test_mae_target = mae(pred_target, target_data)
    test_mse_target = mse(pred_target, target_data)

    writer.add_hparams(options['hp_params'],
                       {'loss': losses[-1],
                        'mse_missing': mse_list_missing[-1], 'mae_missing': mae_list_missing[-1],
                        'mse_target': mse_list_target[-1], 'mae_target': mae_list_target[-1],
                        'test_mse_missing': test_mse_missing, 'test_mae_missing': test_mae_missing,
                        'test_mse_target': test_mse_target, 'test_mae_target': test_mae_target})

    log.info(f"\n====> Final Values <====\n")
    [log.info(f"{key}: {model.state_dict()[key].detach().numpy().squeeze()}") for key in model.state_dict()]


if __name__ == '__main__':
    """ Test the joint model in dummy data """
    SEED = 5
    parsed_args = parser.parse_args()

    log = logging.getLogger(LOGGER_NAME)
    log.info("Testing Joint Model")

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
    KERNEL = parsed_args.kernel

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

    log.info("==> Optimization options <==")
    for key in optim_options.keys():
        log.info(f"{key}: {optim_options[f'{key}']}")

    # Multiple input features to predict one output
    NUM_SAMPLES = 500
    MISSING_DIM = 3
    COND_DIM = 3
    INPUT_DIM = 2
    OUTPUT_DIM = 5

    np.random.seed(SEED)  # Just in case we use shuffle option - reproducibility
    torch.manual_seed(SEED)
    DataSet = SyntheticDataset(num_samples=NUM_SAMPLES, lat_dim=LATENT_DIM,
                               input_dim=INPUT_DIM,
                               cond_dim=COND_DIM,
                               miss_dim=MISSING_DIM,
                               target_dim=OUTPUT_DIM,
                               seed=SEED)

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

    # Initial conditions
    SIGMA_GP = 0
    LAMDA_GP = 0
    MEAN_GP = 0
    NOISE_GP = 0
    NOISE_VI = 0
    BIAS = False

    path_test = os.path.join(runs_path, f"test_L{LATENT_DIM}_H{HIDDEN_DIM}_K{KERNEL}_{LR}")
    model = VIRCA(input_dim=INPUT_DIM,
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

    # model.add_prior_gp("lamda", prior={'mean': tensor([1]), 'logvar': tensor([0])})  # Weak prior
    # model.add_prior_gp("sigma", prior={'mean': tensor([-1]), 'logvar': tensor([-1])})  # Weak prior

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    groups = [dict(params=model._VI.parameters(), lr=LR*1),
              dict(params=model._GP.parameters(), lr=LR*1)]
    optimizer = optim.Adam(groups)
    train_virca(model, optimizer, path_test, train_loader, test_loader, optim_options)
