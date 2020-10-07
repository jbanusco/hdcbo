import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
from typing import List
import numpy as np
import torch


def plot_strip(model, input, feature_names=None, save_path=None):
    model.eval()
    output = model.forward(*input)
    x_rec = output[2]

    num_rec = x_rec.shape[1]
    x_original = pd.DataFrame(input[0].cpu().data.numpy())
    x_rec = pd.DataFrame(x_rec.cpu().data.numpy())

    x_original["set"] = "Original"
    x_rec["set"] = "Rec."
    x_all = pd.concat([x_original, x_rec], axis=0)

    g = sns.catplot(data=pd.melt(x_all, id_vars='set'), col='variable', x='set', y='value', kind='strip', sharey=False)
    if save_path is not None:
        g.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        g.fig.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
        plt.close(g.fig)

    return g.fig


def plot_latent(model, input,  save_path=None):
    """ Plot the latent space """

    output = model.forward(*input)
    z = output[0]  # Mean
    z_logvar = output[1]  # Log-variance

    num_samples = z.shape[0]
    num_dim = z.shape[1]
    x_plot = np.arange(0, num_samples)

    mean_add_std = z + 1*torch.sqrt(torch.exp(z_logvar))
    mean_sub_std = z - 1*torch.sqrt(torch.exp(z_logvar))

    fig, axes = plt.subplots(num_dim, 1, figsize=(3 * 1, 3 * num_dim), squeeze=False)
    for ix in range(num_dim):
        axes[ix, 0].plot(x_plot, z[:, ix].cpu().data.numpy(), marker='.', ms=4, label="Z", linestyle='')
        axes[ix, 0].plot(x_plot, mean_add_std[:, ix].cpu().data.numpy(), marker='o', linestyle='', ms=1,
                         label="Z+Std")
        axes[ix, 0].plot(x_plot, mean_sub_std[:, ix].cpu().data.numpy(), marker='o', linestyle='', ms=1,
                         label="Z-Std")
        axes[ix, 0].set_xlabel(f"Lat.{ix}")
    axes[ix, 0].legend()

    if save_path is not None:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig)

    return fig


def plot_predictions(model,
                     input,
                     features_names: List = None,
                     save_path=None):

    data_x = input[0]
    num_samples = data_x.shape[0]

    output = model.forward(*input)
    pred = output[2]
    logvar = output[3]

    mean_add_std = pred + 1*torch.sqrt(torch.exp(logvar))
    mean_sub_std = pred - 1*torch.sqrt(torch.exp(logvar))

    num_dim = pred.shape[1]
    fig, axes = plt.subplots(num_dim, 1, figsize=(3*1, 3*num_dim), squeeze=False, sharex=False, sharey=False)

    for ix in range(num_dim):
        axes[ix, 0].plot(data_x[:, ix].cpu().data.numpy(), pred[:, ix].cpu().data.numpy(), marker='o', ms=4,
                         label="Mean", linestyle='')
        axes[ix, 0].plot(data_x[:, ix].cpu().data.numpy(), mean_add_std[:, ix].cpu().data.numpy(), marker='o',
                         linestyle='', ms=1, label="Mean+Std")
        axes[ix, 0].plot(data_x[:, ix].cpu().data.numpy(), mean_sub_std[:, ix].cpu().data.numpy(), marker='o',
                         linestyle='', ms=1, label="Mean-Std")

        min_val = scipy.percentile(data_x[:, ix], 1)
        max_val = scipy.percentile(data_x[:, ix], 99)
        axes[ix, 0].set_xlim([min_val, max_val])

        min_val = scipy.percentile(mean_sub_std[:, ix].cpu().data.numpy(), 1)
        max_val = scipy.percentile(mean_add_std[:, ix].cpu().data.numpy(), 99)
        axes[ix, 0].set_ylim([min_val, max_val])

        if features_names is None:
            axes[ix, 0].set_xlabel(f"Rec f{ix}")
        else:
            axes[ix, 0].set_xlabel(f"{features_names[ix]}")

    axes[ix, 0].legend()

    if save_path is not None:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig)

    return fig


def plot_residual(model,
                  input,
                  features_names: List = None,
                  save_path=None):

    output = model.forward(*input)
    pred = output[2]
    num_dim = pred.shape[1]

    data_x = input[0]
    residual = pred - data_x

    fig, axes = plt.subplots(num_dim, 1, figsize=(3*1, 3*num_dim), squeeze=False, sharex=False, sharey=False)

    for ix in range(num_dim):
        axes[ix, 0].axhline(y=0, xmin=-1, xmax=1, linestyle="--", color='red')
        axes[ix, 0].plot(data_x[:, ix].cpu().data.numpy(), residual[:, ix].cpu().data.numpy(), ms=4, marker=".",
                         linestyle="")

        min_val = scipy.percentile(data_x[:, ix], 1)
        max_val = scipy.percentile(data_x[:, ix], 99)
        axes[ix, 0].set_xlim([min_val, max_val])

        min_val = scipy.percentile(residual[:, ix].cpu().data.numpy(), 1)
        max_val = scipy.percentile(residual[:, ix].cpu().data.numpy(), 99)
        axes[ix, 0].set_ylim([min_val, max_val])

        axes[ix, 0].set_ylabel(f"residual_{ix}")
        if features_names is None:
            axes[ix, 0].set_xlabel(f"missing_{ix}")
        else:
            axes[ix, 0].set_title(f"{features_names[ix]}")

    fig.suptitle("Residuals")
    # plt.show()

    if save_path is not None:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig)

    return fig


def plot_correlations_vi(model,
                         input,
                         features_names: List = None,
                         save_path=None):

    """ Plot the correlations """

    output = model.forward(*input)
    pred = output[2]
    num_dim = pred.shape[1]
    data_x = input[0]

    fig, axes = plt.subplots(num_dim, 1, figsize=(3*1, 3*num_dim), squeeze=False, sharex=False, sharey=False)
    for ix in range(num_dim):
        axes[ix, 0].axhline(y=0, xmin=-1, xmax=1, linestyle="--", color='red')
        axes[ix, 0].plot(data_x[:, ix].cpu().data.numpy(), pred[:, ix].cpu().data.numpy(), ms=4, marker=".", linestyle="")

        min_val = scipy.percentile(data_x[:, ix], 1)
        max_val = scipy.percentile(data_x[:, ix], 99)
        axes[ix, 0].set_xlim([min_val, max_val])

        min_val = scipy.percentile(pred[:, ix].cpu().data.numpy(), 1)
        max_val = scipy.percentile(pred[:, ix].cpu().data.numpy(), 99)
        axes[ix, 0].set_ylim([min_val, max_val])

        if features_names is None:
            axes[ix, 0].set_xlabel(f"target_{ix}")
            axes[ix, 0].set_ylabel(f"pred_{ix}")
        else:
            axes[ix, 0].set_xlabel(f"{features_names[ix]}")
            axes[ix, 0].set_ylabel(f"pred_{features_names[ix]}")

    fig.suptitle("Correlation between predictor and indicator")
    # plt.show()

    if save_path is not None:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig)

    return fig
