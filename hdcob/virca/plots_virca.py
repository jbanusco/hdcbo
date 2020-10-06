import os
import pandas as pd
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from hdcob.gp.plots_gp import plot_residual as plot_residual_gp
from hdcob.gp.plots_gp import plot_multiple_residual as plot_multiple_residual_gp
from hdcob.gp.plots_gp import plot_predictions as plot_predictions_gp
from hdcob.gp.plots_gp import plot_multiple_predictions as plot_multiple_predictions_gp
from hdcob.gp.plots_gp import plot_correlations_gp
from hdcob.vi.plots_vi import plot_residual as plot_residual_vi
from hdcob.vi.plots_vi import plot_predictions as plot_predictions_vi
from hdcob.vi.plots_vi import plot_correlations_vi
from hdcob.config import *


def plot_boxplot_params(input_data,
                        missing_data,
                        target_data,
                        ground_missing=None,
                        ground_target=None,
                        names_input=None,
                        names_missing=None,
                        names_target=None):

    if names_input is None:
        names_input = [f"I{ix}" for ix in range(input_data.shape[1])]

    if names_missing is None:
        names_missing = [f"M{ix}" for ix in range(missing_data.shape[1])]

    if names_target is None:
        names_target = [f"T{ix}" for ix in range(target_data.shape[1])]

    # Get dataframes
    data_input = pd.DataFrame(data=input_data.cpu().data.numpy(), columns=names_input)
    data_missing = pd.DataFrame(data=missing_data.cpu().data.numpy(), columns=names_missing)
    data_target = pd.DataFrame(data=target_data.cpu().data.numpy(), columns=names_target)

    # data_estimated = pd.concat([data_input, data_missing, data_target], axis=1)
    data_estimated = pd.concat([data_missing, data_target], axis=1)
    data_estimated["type"] = "Estim."

    # Get the original data if available (or ground truth)
    if ground_missing is not None and ground_target is not None:
        g_data_missing = pd.DataFrame(data=ground_missing.cpu().data.numpy(), columns=names_missing)
        g_data_target = pd.DataFrame(data=ground_target.cpu().data.numpy(), columns=names_target)

        # data_ground = pd.concat([data_input, g_data_missing, g_data_target], axis=1)
        data_ground = pd.concat([g_data_missing, g_data_target], axis=1)
        data_ground["type"] = "Target"

        data_all = pd.concat([data_estimated, data_ground], axis=0)
    else:
        data_all = data_estimated

    final_df = pd.melt(data_all, id_vars=["type"])
    g = sns.catplot(data=final_df, x="type", y="value", col="variable", sharey=False, kind='boxen', col_wrap=4,
                    sharex=False)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    g.set_ylabels("Value")

    import scipy
    for ix, ax in enumerate(g.axes.flatten()):
        min_val = scipy.percentile(data_all[f"{ax.get_title()}"], 1)
        max_val = scipy.percentile(data_all[f"{ax.get_title()}"], 99)
        ax.set_ylim([min_val, max_val])

    return g


def plot_predictions(model,
                     x: tensor,
                     y: tensor,
                     x_cond: tensor = None,
                     x_miss: tensor = None,
                     num_samples: int = 1,
                     shade: bool = False,
                     input_features_names: List = None,
                     miss_features_names: List = None,
                     targ_features_names: List = None,
                     save_folder=None):

    if save_folder is not None:
        save_vi_filename = os.path.join(save_folder, f"predictions_VI.png")
        save_gp_filename = os.path.join(save_folder, f"predictions_GP.png")
    else:
        save_vi_filename = None
        save_gp_filename = None

    if model.condition:
        # input_data = [x_miss, x_cond]
        input_data = [x_miss, torch.cat((x_cond, x), dim=1)]
    else:
        input_data = [x_miss]

    # VI
    fig_pred_vi = plot_predictions_vi(model._VI, input_data, save_path=save_vi_filename)

    # GP
    with torch.no_grad():
        model._VI.eval()
        mu, logvar, x_hat_mu, x_hat_logvar = model._VI(*input_data)
        x_train = torch.cat((x, x_hat_mu), dim=1)

    if input_features_names is None:
        input_features_names = [f"input_{ix}" for ix in range(model._input_dim)]

    if miss_features_names is None:
        miss_features_names = [f"missing_{ix}" for ix in range(model._miss_dim)]

    x_names = input_features_names + miss_features_names

    if y.shape[1] > 1:
        fig_pred_gp, _ = plot_multiple_predictions_gp(model._GP, x_train, y, num_samples=0, shade=True,
                                                      x_names=x_names, save_filename=save_gp_filename)
    else:
        fig_pred_gp, _ = plot_predictions_gp(model._GP, x_train, y, num_samples=0, shade=True,
                                             save_filename=save_gp_filename, x_names=x_names)

    return fig_pred_vi, fig_pred_gp


def plot_residual(model,
                  x: tensor,
                  y: tensor,
                  x_cond: tensor = None,
                  x_miss: tensor = None,
                  input_features_names: List = None,
                  miss_features_names: List = None,
                  pred_features_names: List = None,
                  save_folder=None):

    if save_folder is not None:
        save_vi_filename = os.path.join(save_folder, "residuals_VI.png")
        save_gp_filename = os.path.join(save_folder, f"residuals_GP.png")
    else:
        save_vi_filename = None
        save_gp_filename = None

    if model.condition:
        # input_data = [x_miss, x_cond]
        input_data = [x_miss, torch.cat((x_cond, x), dim=1)]
    else:
        input_data = [x_miss]

    fig_res_vi = plot_residual_vi(model._VI, input_data, save_path=save_vi_filename)

    # GP
    if input_features_names is None:
        input_features_names = [f"input_{ix}" for ix in range(model._input_dim)]

    if miss_features_names is None:
        miss_features_names = [f"missing_{ix}" for ix in range(model._miss_dim)]

    x_names = input_features_names + miss_features_names

    with torch.no_grad():
        model._VI.eval()
        mu, logvar, x_hat_mu, x_hat_logvar = model._VI(*input_data)
        x_train = torch.cat((x, x_hat_mu), dim=1)

    if y.shape[1] > 1:
        fig_res_gp, _ = plot_multiple_residual_gp(model._GP, x_train, y,
                                                  x_names=x_names, save_filename=save_gp_filename)
    else:
        fig_res_gp, _ = plot_residual_gp(model._GP, x_train, y, save_filename=save_gp_filename,
                                         x_names=x_names)

    return fig_res_vi, fig_res_gp


def plot_correlations(model,
                      x: tensor,
                      y: tensor,
                      x_cond: tensor = None,
                      x_miss: tensor = None,
                      target_features_names: List = None,
                      miss_features_names: List = None,
                      save_folder=None):

    if save_folder is not None:
        save_vi_filename = os.path.join(save_folder, "correlations_VI.png")
        save_gp_filename = os.path.join(save_folder, f"correlations_GP.png")
    else:
        save_vi_filename = None
        save_gp_filename = None

    if model.condition:
        # input_data = [x_miss, x_cond]
        input_data = [x_miss, torch.cat((x_cond, x), dim=1)]
    else:
        input_data = [x_miss]

    # VI
    fig_corr_vi = plot_correlations_vi(model._VI, input_data,
                                       save_path=save_vi_filename,
                                       features_names=miss_features_names)

    # GP
    if target_features_names is None:
        target_features_names = [f"target_{ix}" for ix in range(model._target_dim)]

    pred_features_names = [f"rec_{name}" for name in target_features_names]

    with torch.no_grad():
        model._VI.eval()
        mu, logvar, x_hat_mu, x_hat_logvar = model._VI(*input_data)
        x_train = torch.cat((x, x_hat_mu), dim=1)

    fig_corr_gp, _ = plot_correlations_gp(model._GP,
                                          x_train, y,
                                          save_filename=save_gp_filename,
                                          x_names=target_features_names,
                                          y_names=pred_features_names)

    return fig_corr_vi, fig_corr_gp
