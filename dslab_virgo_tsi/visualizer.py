import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import style
from scipy.stats import norm

from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import moving_average_std, mission_day_to_year, downsample_signal


class Visualizer(object):

    def __init__(self):
        style.use(Const.MATPLOTLIB_STYLE)
        mpl.rcParams['lines.linewidth'] = Const.MATPLOTLIB_STYLE_LINEWIDTH
        mpl.rcParams["savefig.format"] = Const.OUT_FORMAT
        mpl.rcParams["savefig.bbox"] = Const.OUT_BBOX
        mpl.rcParams["savefig.dpi"] = Const.OUT_DPI
        mpl.rcParams.update({'font.size': Const.TITLE_FONT_SIZE})
        mpl.rc('xtick', labelsize=Const.XTICK_SIZE)
        mpl.rc('ytick', labelsize=Const.YTICK_SIZE)
        mpl.rc('axes', labelsize=Const.AXES_FONT_SIZE)

    @staticmethod
    def set_figsize(size=Const.FIG_SIZE):
        mpl.rcParams['figure.figsize'] = list(size)

    @staticmethod
    def plot_signals(signal_fourplets, results_dir, title, x_ticker=None, legend=None, y_lim=None,
                     x_label=None, y_label=None, max_points=1e5):

        fig = plt.figure()
        for signal_fourplet in signal_fourplets:
            if not signal_fourplet:
                continue

            t = signal_fourplet[0]
            x = signal_fourplet[1]
            label = signal_fourplet[2]
            scatter = signal_fourplet[3]

            if x_label == Const.YEAR_UNIT:
                t = np.array(list(map(mission_day_to_year, t)))

            if x.shape[0] > max_points:
                downsampling_factor = int(np.floor(float(x.shape[0]) / float(max_points)))
                t = downsample_signal(t, downsampling_factor)
                x = downsample_signal(x, downsampling_factor)

            if scatter:
                plt.scatter(t, x, label=label, marker="x", color="tab:red")
            else:
                plt.plot(t, x, label=label)

        plt.title(title)

        if x_ticker:
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))

        if legend:
            plt.legend(loc=legend)
        else:
            plt.legend()

        if y_lim:
            plt.ylim(y_lim)

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if results_dir:
            plt.savefig(os.path.join(results_dir, title))
            logging.info(f"Plot {title} generated.")

        return fig

    @staticmethod
    def plot_signals_mean_std(signal_fourplets, results_dir, title, x_ticker=None, legend=None, y_lim=None,
                              x_label=None, y_label=None, confidence=0.95, alpha=0.5, max_points=1e5):

        factor = norm.ppf(1 / 2 + confidence / 2)  # 0.95 % -> 1.959963984540054

        fig = plt.figure()
        for signal_fourplet in signal_fourplets:

            t = signal_fourplet[0]
            x = signal_fourplet[1]
            label = signal_fourplet[2]
            window = signal_fourplet[3]

            if x.shape[0] > max_points:
                downsampling_factor = int(np.floor(float(x.shape[0]) / float(max_points)))
                t = downsample_signal(t, downsampling_factor)
                x = downsample_signal(x, downsampling_factor)

            if window:
                x_ma, x_std = moving_average_std(x, t, w=window)

                if x_label == Const.YEAR_UNIT:
                    t = np.array(list(map(mission_day_to_year, t)))

                plt.plot(t, x_ma, label=label)
                plt.fill_between(t, x_ma - factor * x_std, x_ma + factor * x_std, alpha=alpha,
                                 label='{}_{}_conf_interval'.format(label, confidence))
            else:
                if x_label == Const.YEAR_UNIT:
                    t = np.array(list(map(mission_day_to_year, t)))

                plt.plot(t, x, label=label)

        plt.title(title)

        if x_ticker:
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))

        if legend:
            plt.legend(loc=legend)
        else:
            plt.legend()

        if y_lim:
            plt.ylim(y_lim)

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if results_dir:
            plt.savefig(os.path.join(results_dir, title))
            logging.info(f"Plot {title} generated.")

        return fig

    @staticmethod
    def plot_signals_mean_std_precompute(signal_fourplets, results_dir, title, x_ticker=None, legend=None, y_lim=None,
                                         x_label=None, y_label=None, ground_truth_triplet=None,
                                         data_points_triplets=None, confidence=0.95, alpha=0.5, max_points=1e5,
                                         max_points_scatter=1e4, inducing_points=None):

        factor = norm.ppf(1 / 2 + confidence / 2)  # 0.95 % -> 1.959963984540054

        x_mean = None
        fig = plt.figure()
        for signal_fourplet in signal_fourplets:
            t = signal_fourplet[0]
            x_mean = signal_fourplet[1]
            x_std = signal_fourplet[2]
            label = signal_fourplet[3]

            if x_mean.shape[0] > max_points:
                downsampling_factor = int(np.floor(float(x_mean.shape[0]) / float(max_points)))
                t = downsample_signal(t, downsampling_factor)
                x_mean = downsample_signal(x_mean, downsampling_factor)
                x_std = downsample_signal(x_std, downsampling_factor)

            if x_label == Const.YEAR_UNIT:
                t = np.array(list(map(mission_day_to_year, t)))

            plt.plot(t, x_mean, label=label)
            plt.fill_between(t, x_mean - factor * x_std, x_mean + factor * x_std, alpha=alpha,
                             label='{}_{}_conf_interval'.format(label, confidence))

        if isinstance(inducing_points, np.ndarray):
            inducing_points = inducing_points[:, 0]
            inducing_points_positions = x_mean.mean() * np.ones_like(inducing_points)

            if x_label == Const.YEAR_UNIT:
                inducing_points = np.array(list(map(mission_day_to_year, inducing_points)))

            plt.plot(inducing_points, inducing_points_positions, 'k|', mew=1, label="SVGP_INDUCING_POINTS")

        if data_points_triplets:
            for data_points_triplet in data_points_triplets:

                t_data_points = data_points_triplet[0]
                if x_label == Const.YEAR_UNIT:
                    t_data_points = np.array(list(map(mission_day_to_year, t_data_points)))

                signal_data_points = data_points_triplet[1]
                label_data_points = data_points_triplet[2]

                if t_data_points.shape[0] > max_points_scatter:
                    downsampling_factor = int(np.floor(float(t_data_points.shape[0]) / float(max_points_scatter)))
                    t_data_points = downsample_signal(t_data_points, downsampling_factor)
                    signal_data_points = downsample_signal(signal_data_points, downsampling_factor)

                plt.scatter(t_data_points, signal_data_points,
                            label=label_data_points,
                            marker=Const.MATPLOTLIB_STYLE_MARKER,
                            s=Const.MATPLOTLIB_STYLE_MARKERSIZE)

        if ground_truth_triplet:
            t_ground_truth = ground_truth_triplet[0]
            if x_label == Const.YEAR_UNIT:
                t_ground_truth = np.array(list(map(mission_day_to_year, t_ground_truth)))

            signal_ground_truth = ground_truth_triplet[1]
            label_ground_truth = ground_truth_triplet[2]

            plt.plot(t_ground_truth, signal_ground_truth, label=label_ground_truth)

        plt.title(f"{title}_{int(confidence * 100)}%_conf_interval")

        if x_ticker:
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))

        if legend:
            plt.legend(loc=legend)
        else:
            plt.legend()

        if y_lim:
            plt.ylim(y_lim)

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if results_dir:
            name = "{}_{}_conf_interval".format(title, int(confidence * 100))
            plt.savefig(os.path.join(results_dir, name))
            logging.info(f"Plot {name} generated.")

        return fig

    @staticmethod
    def plot_signal_history(t_mutual, history_fitresults, results_dir, title, ground_truth_triplet=None, x_ticker=None,
                            legend=None, y_lim=None, x_label=None, y_label=None):

        t_ground_truth = None
        signal_ground_truth = None
        label_ground_truth = None

        if ground_truth_triplet:
            t_ground_truth = ground_truth_triplet[0]
            if x_label == Const.YEAR_UNIT:
                t_ground_truth = np.array(list(map(mission_day_to_year, t_ground_truth)))

            signal_ground_truth = ground_truth_triplet[1]
            label_ground_truth = ground_truth_triplet[2]

        max_plots = 10

        if len(history_fitresults) > max_plots:
            selected_triplet_indices = np.arange(1, len(history_fitresults) - 1,
                                                 np.ceil((len(history_fitresults) - 1) / max_plots))
        else:
            selected_triplet_indices = np.arange(1, len(history_fitresults) - 1, 1)

        max_plots = len(selected_triplet_indices) + 2

        if x_label == Const.YEAR_UNIT:
            t_mutual = np.array(list(map(mission_day_to_year, t_mutual)))

        fig, axs = plt.subplots(max_plots, 2, figsize=(16, 4 * max_plots))

        selected_step = 0
        for step, fitresult in enumerate(history_fitresults):
            if step in selected_triplet_indices or step in [0, len(history_fitresults) - 1]:
                a = fitresult.a_mutual_nn_corrected
                b = fitresult.b_mutual_nn_corrected
                r = np.divide(a, b)

                axs[selected_step, 0].plot(t_mutual, a, label=f"step_{step}_A")
                axs[selected_step, 0].plot(t_mutual, b, label=f"step_{step}_B")

                if ground_truth_triplet:
                    axs[selected_step, 0].plot(t_ground_truth, signal_ground_truth, label=label_ground_truth)

                axs[selected_step, 1].plot(t_mutual, r, label=f"step_{step}_RATIO")
                axs[selected_step, 1].plot(t_mutual, fitresult.ratio_a_b_mutual_nn_corrected,
                                           label=f"step_{step}_RATIO_fitted")

                if x_label:
                    axs[selected_step, 0].set_xlabel(x_label)
                    axs[selected_step, 1].set_xlabel(x_label)

                if y_label:
                    axs[selected_step, 0].set_ylabel(y_label)
                    axs[selected_step, 1].set_ylabel("RATIO")

                if x_ticker:
                    axs[selected_step, 0].xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))
                    axs[selected_step, 1].xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))

                if y_lim:
                    axs[selected_step, 0].ylim(y_lim)

                if y_label != Const.TSI_UNIT:
                    axs[selected_step, 1].set_ylim([0, 1.2])

                if legend:
                    axs[selected_step, 0].legend(loc=legend)
                    axs[selected_step, 1].legend(loc=legend)
                else:
                    axs[selected_step, 0].legend()
                    axs[selected_step, 1].legend()

                selected_step = selected_step + 1

        if results_dir:
            plt.savefig(os.path.join(results_dir, title))
            logging.info(f"Plot {title} generated.")

        return fig

    @staticmethod
    def plot_iter_loglikelihood(iter_loglikelihood, results_dir, title, legend=None, x_label=None, y_label=None):

        iterations = [pair[0] for pair in iter_loglikelihood]
        loglikelihood = [pair[1] for pair in iter_loglikelihood]

        fig = plt.figure()
        plt.plot(iterations, loglikelihood, label="LOG_LIKELIHOOD_SVGP")
        plt.title(title)

        if legend:
            plt.legend(loc=legend)
        else:
            plt.legend()

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if results_dir:
            plt.savefig(os.path.join(results_dir, title))
            logging.info(f"Plot {title} generated.")

        return fig



