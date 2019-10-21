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

    @staticmethod
    def set_figsize(size=Const.FIG_SIZE):
        mpl.rcParams['figure.figsize'] = list(size)

    @staticmethod
    def plot_signals(signal_fourplets, results_dir, title, x_ticker=None, legend=None, y_lim=None,
                     x_label=None, y_label=None, max_points=1e4):

        fig = plt.figure()
        for signal_fourplet in signal_fourplets:
            t = signal_fourplet[0]
            x = signal_fourplet[1]
            label = signal_fourplet[2]
            scatter = signal_fourplet[3]

            if x_label == Const.YEAR_UNIT:
                t = np.array(list(map(mission_day_to_year, t)))

            if x.shape[0] > max_points:
                downsampling_factor = np.ceil(x.shape[0] / max_points)
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

        return fig

    @staticmethod
    def plot_signals_mean_std(signal_fourplets, results_dir, title, x_ticker=None, legend=None, y_lim=None,
                              x_label=None, y_label=None, confidence=0.95, alpha=0.5):

        fig = plt.figure()
        for signal_fourplet in signal_fourplets:
            factor = norm.ppf(confidence) - norm.ppf(1 - confidence)
            t = signal_fourplet[0]

            x = signal_fourplet[1]
            label = signal_fourplet[2]

            if signal_fourplet[3]:
                x_ma, x_std = moving_average_std(x, t, w=signal_fourplet[3], center=True)

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

        return fig
