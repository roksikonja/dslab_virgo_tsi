from constants import Constants as C

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import matplotlib.ticker as ticker

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
    def plot_signals(signal_triplets, results_dir, title, x_ticker=None, legend=None, y_lim=None,
                     x_label=None, y_label=None):

        fig = plt.figure()
        for signal_triplet in signal_triplets:
            t = signal_triplet[0]
            x = signal_triplet[1]
            label = signal_triplet[2]
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
