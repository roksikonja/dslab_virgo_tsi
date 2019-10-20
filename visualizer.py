from constants import Constants as C
from data_utils import moving_average_std, mission_day_to_year

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.stats import norm
import numpy as np

import os


class Visualizer(object):

    def __init__(self):
        style.use(C.MATPLOTLIB_STYLE)
        mpl.rcParams['lines.linewidth'] = C.MATPLOTLIB_STYLE_LINEWIDTH
        mpl.rcParams["savefig.format"] = C.OUT_FORMAT
        mpl.rcParams["savefig.bbox"] = C.OUT_BBOX
        mpl.rcParams["savefig.dpi"] = C.OUT_DPI

    @staticmethod
    def set_figsize(size=C.FIG_SIZE):
        mpl.rcParams['figure.figsize'] = list(size)

    @staticmethod
    def plot_signals(signal_fourplets, results_dir, title, x_ticker=None, legend=None, y_lim=None,
                     x_label=None, y_label=None):

        fig = plt.figure()
        for signal_fourplet in signal_fourplets:
            t = signal_fourplet[0]
            if x_label == C.YEAR_UNIT:
                t = np.array(list(map(mission_day_to_year, t)))

            x = signal_fourplet[1]
            label = signal_fourplet[2]

            if not signal_fourplet[3]:
                plt.plot(t, x, label=label)
            else:
                plt.scatter(t, x, label=label)
        
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

                if x_label == C.YEAR_UNIT:
                    t = np.array(list(map(mission_day_to_year, t)))

                plt.plot(t, x_ma, label=label)
                plt.fill_between(t, x_ma - factor * x_std, x_ma + factor * x_std, alpha=alpha,
                                 label='{}_{}_conf_interval'.format(label, confidence))
            else:
                if x_label == C.YEAR_UNIT:
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
