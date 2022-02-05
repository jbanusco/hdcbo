from hdcob.config import *
from typing import List
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_strip(y_or, y_rec, x_names = None, y_names=None, save_path=None):

    y_original = pd.DataFrame(y_or.cpu().data.numpy())
    y_recon = pd.DataFrame(y_rec.cpu().data.numpy())

    y_original["set"] = "Original"
    y_recon["set"] = "Rec."
    y_all = pd.concat([y_original, y_recon], axis=0)

    g = sns.catplot(data=pd.melt(y_all, id_vars='set'), col='variable', x='set', y='value', kind='strip', sharey=False)
    if save_path is not None:
        g.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        g.fig.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
        plt.close(g.fig)

    return g.fig


def plot_residual(model: torch.nn.Module,
                  x: tensor,
                  y: tensor,
                  x_names: List = None,
                  y_names: List = None,
                  axes=None,
                  fig=None,
                  save_filename=None):

    """ Plot the residual """
    num_features = x.shape[1]
    mean_cond, cov_cond = model.predict(x)

    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    residual = mean_cond.squeeze() - y.squeeze()

    if axes is None:
        fig, axes = plt.subplots(num_features, 1, figsize=(3 * 1, 3 * num_features), squeeze=False,
                                 sharey=False, sharex=False)
    else:
        if len(axes.shape) == 1: axes = np.expand_dims(axes, 1)

    if num_features == 1:
        axes = np.asarray(axes)

    for ix in range(num_features):
        axes[ix, 0].axhline(y=0, xmin=-1, xmax=1, linestyle="--", color='red')
        axes[ix, 0].plot(x[:, ix].cpu().data.numpy(), residual.cpu().data.numpy(), ms=4, marker=".", linestyle="")
        if x_names is not None:
            axes[ix, 0].set_xlabel(f"{x_names[ix]}")
        else:
            axes[ix, 0].set_xlabel(f"x{ix}")
        axes[ix, 0].set_ylabel(f"residual_{ix}")

    if fig is not None:
        fig.suptitle("Prediction residuals")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_filename is not None:
        fig.savefig(save_filename, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig)

    return fig, axes


def plot_predictions(model: torch.nn.Module,
                     x: tensor,
                     y: tensor,
                     num_samples: int = 1,
                     shade: bool = False,
                     x_names: List = None,
                     y_names: List = None,
                     axes=None,
                     fig=None,
                     save_filename=None):
    """
    Plot the predictions and the confidence interval
    :param x:
    :param y:
    :param: transform: Transform jointly X, Y (usually coming from a same dataset)
    :param: features_names: List of names for the features
    :return:
    """

    # Plot the prediction of the GP
    mean_cond, cov_cond = model.predict(x)

    # Sample some predictions
    if num_samples <= 0:
        samples = None
    else:
        samples = model.sample(x, num_samples=num_samples)

    if len(mean_cond.shape) == 1:
        mean_cond = mean_cond.unsqueeze(1)

    num_features = x.shape[1]
    # if num_features > 1:
    #     shade = False

    if len(cov_cond.shape) == 3: cov_cond = cov_cond.squeeze()
    if len(mean_cond.shape) == 0: mean_cond = mean_cond.unsqueeze(0)
    mean_add_std = mean_cond + cov_cond.mm(torch.ones((mean_cond.shape[0], 1)).to(DEVICE))
    mean_sub_std = mean_cond - cov_cond.mm(torch.ones((mean_cond.shape[0], 1)).to(DEVICE))

    x_train = model.x_train
    y_train = model.y_train

    if axes is None:
        fig, axes = plt.subplots(num_features, 1, figsize=(3 * 1, 3 * num_features), squeeze=False,
                                 sharey=False, sharex=False)
    else:
        if len(axes.shape) == 1: axes = np.expand_dims(axes, 1)

    if num_features == 1:
        axes = np.asarray(axes)

    for ix in range(num_features):
        axes[ix, 0].plot(x_train[:, ix].cpu().data.numpy(), y_train.cpu().data.numpy(), marker='.', ms=2,
                      label="Training", linestyle='')
        if samples is not None:
            axes[ix, 0].plot(x[:, ix].cpu().data.numpy(), samples.cpu().data.numpy(), ms=4, marker=".", linestyle="",
                          label="Sample")

        axes[ix, 0].plot(x[:, ix].cpu().data.numpy(), mean_cond.cpu().data.numpy(), marker='o', ms=4, label="Pred.",
                      linestyle="")
        axes[ix, 0].plot(x[:, ix].cpu().data.numpy(), y.cpu().data.numpy(), '.', ms=4, label="Ground", linestyle='')

        if shade:
            ind_sort = x[:, ix].sort(descending=False)[-1]
            axes[ix, 0].fill_between(x[ind_sort, ix].cpu().data.numpy(),
                                  mean_sub_std.cpu().data.numpy().squeeze()[ind_sort],
                                  mean_add_std.cpu().data.numpy().squeeze()[ind_sort], color="#dddddd")

        # axes[ix].set_title(f"Samples from the GP posterior")
        if x_names is not None:
            axes[ix, 0].set_xlabel(f"{x_names[ix]}")

        if y_names is not None:
            axes[ix, 0].set_ylabel(f"{y_names}")

    # plt.show()
    axes[-1, 0].legend()

    if fig is not None:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_filename is not None:
        fig.savfig(save_filename, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig)

    return fig, axes


def plot_multiple_predictions(model: torch.nn.Module,
                     x: tensor,
                     y: tensor,
                     num_samples: int = 0,
                     shade: bool = False,
                     x_names: List = None,
                     y_names: List = None,
                     save_filename=None):

    fig_gp, axes_gp = plt.subplots(model._output_dim, model._input_dim,
                                   figsize=(3*model._input_dim, 3*model._output_dim), squeeze=False,
                                   sharey=False, sharex=False)

    if y_names is None:
        y_names = [f"target_{ix}" for ix in range(model._output_dim)]

    for ix in range(model._output_dim):
        _, _ = plot_predictions(model._GP[ix], x, y[:, ix], num_samples=num_samples, shade=shade,
                                x_names=x_names, y_names=[y_names[ix]], axes=axes_gp[ix, :])

    if save_filename is not None:
        fig_gp.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_gp.savefig(save_filename, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig_gp)

    return fig_gp, axes_gp


def plot_multiple_residual(model: torch.nn.Module,
                  x: tensor,
                  y: tensor,
                  x_names: List = None,
                  y_names: List = None,
                  save_filename=None):

    fig_gp, axes_gp = plt.subplots(model._output_dim, model._input_dim,
                                   figsize=(3 * model._input_dim, 3 * model._output_dim), squeeze=False,
                                   sharey=False, sharex=False)

    if y_names is None:
        y_names = [f"target_{ix}" for ix in range(model._output_dim)]

    for ix in range(model._output_dim):
        _, _ = plot_residual(model._GP[ix], x, y[:, ix],
                                          x_names=x_names,
                                          y_names=[y_names[ix]],
                                          axes=axes_gp[ix, :])

    if save_filename is not None:
        fig_gp.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_gp.savefig(save_filename, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig_gp)

    return fig_gp, axes_gp


def plot_correlations_gp(model: torch.nn.Module,
                         x: tensor,
                         y: tensor,
                         x_names: List = None,
                         y_names: List = None,
                         axes=None,
                         fig=None,
                         save_filename=None):

    """ Plot the correlations """
    mean_cond, cov_cond = model.predict(x)

    if len(y.shape) == 1: y = y.unsqueeze(1)
    num_features = y.shape[1]

    if axes is None:
        fig, axes = plt.subplots(num_features, 1, figsize=(3 * 1, 3 * num_features), squeeze=False,
                                 sharex=False, sharey=False)
    else:
        if len(axes.shape) == 1: axes = np.expand_dims(axes, 1)

    if num_features == 1:
        axes = np.asarray(axes)

    if y_names is None:
        y_names = [f"target_{ix}" for ix in range(model._output_dim)]

    for ix in range(num_features):
        axes[ix, 0].plot(mean_cond[:, ix].cpu().data.numpy(), y[:, ix].cpu().data.numpy(), ms=4, marker=".", linestyle="")
        if x_names is not None:
            axes[ix, 0].set_xlabel(f"{x_names[ix]}")
        else:
            axes[ix, 0].set_xlabel(f"pred_{ix}")
        axes[ix, 0].set_ylabel(f"{y_names[ix]}")

    if fig is not None:
        fig.suptitle("Correlation between predictor and indicator")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_filename is not None:
        fig.savefig(save_filename, bbox_inches='tight', format='png', dpi=200)
        plt.close(fig)

    return fig, axes


def plot_ard(model: torch.nn.Module,
             input_names: List = None,
             target_names: List = None,
             save_path: str = None):
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- Test ARD -----------------------------------------------------------
    if input_names is None:
        input_names = [f"input_{ix}" for ix in range(model._input_dim)]

    if target_names is None:
        target_names = [f"target_{ix}" for ix in range(model._output_dim)]

    df_lamdas = pd.DataFrame(data=None, index=[r"$\alpha$"] + input_names, columns=target_names)

    for ix_t, t_var in enumerate(target_names):
        for ix_p, p_var in enumerate(input_names):
            df_lamdas[t_var].loc[p_var] = 1 / np.exp(model._GP[ix_t].lamda[ix_p].data.numpy()[0])
        df_lamdas[t_var].loc[r"$\alpha$"] = np.exp(model._GP[ix_t].sigma.data.numpy()[0])

    g = sns.catplot(data=pd.melt(df_lamdas.reset_index(), id_vars='index'), y="index", x="value", col="variable",
                    kind='bar', sharey=False)
    g.set_titles("{col_name}")
    g.set_xlabels("")

    if save_path is not None:
        fig = g.fig
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
        # plt.close(fig)

    return g.fig
