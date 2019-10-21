import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.interpolate import interp1d

from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import load_data, make_dir, notnan_indices, mission_day_to_year, \
    get_sampling_intervals, detect_outliers
from dslab_virgo_tsi.visualizer import Visualizer

visualizer = Visualizer()

data_dir = Const.DATA_DIR
virgo_file = Const.VIRGO_FILE
results_dir = Const.RESULTS_DIR
results_dir = make_dir(os.path.join(results_dir, datetime.datetime.now().strftime("data_analysis_%Y-%m-%d")))

parser = argparse.ArgumentParser()
parser.add_argument("--visualize", action="store_true", help="Flag for visualizing results.")
parser.add_argument("--sampling", action="store_true", help="Flag for sampling analysis.")
parser.add_argument("--fft", action="store_true", help="Flag for FFT analysis.")
parser.add_argument("--t_early_increase", type=int, default=100, help="Early increase time span.")
parser.add_argument("--sampling_window", type=int, default=5, help="Minimum size of sampling gap.")
parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")

ARGS = parser.parse_args()

SAMPLING_WINDOW = ARGS.sampling_window
T_EARLY_INCREASE = ARGS.t_early_increase
OUTLIER_FRACTION = ARGS.outlier_fraction


data = load_data(data_dir, virgo_file)
t = data[Const.T].values
pmo_a = data[Const.A].values
pmo_b = data[Const.B].values
temp = data[Const.TEMP].values

# Filter outliers
outliers_a = notnan_indices(pmo_a)
outliers_a[outliers_a] = detect_outliers(pmo_a[notnan_indices(pmo_a)], None,
                                         outlier_fraction=OUTLIER_FRACTION)

outliers_b = notnan_indices(pmo_b)
outliers_b[outliers_b] = detect_outliers(pmo_b[notnan_indices(pmo_b)], None,
                                         outlier_fraction=OUTLIER_FRACTION)

pmo_a_outliers = pmo_a.copy()
pmo_b_outliers = pmo_b.copy()

pmo_a[outliers_a] = np.nan
pmo_b[outliers_b] = np.nan
pmo_a_outliers[~outliers_a] = np.nan
pmo_b_outliers[~outliers_b] = np.nan

data[Const.A] = pmo_a
data[Const.B] = pmo_b

# Filter nan values
data_nn = data[[Const.T, Const.A, Const.B]].dropna()
t_nn = data_nn[Const.T].values
pmo_a_nn = data_nn[Const.A].values
pmo_b_nn = data_nn[Const.B].values

visualizer.set_figsize()

x_a_outliers = pmo_a_outliers[notnan_indices(pmo_a_outliers)]
t_a_outliers = t.copy()[notnan_indices(pmo_a_outliers)]
x_b_outliers = pmo_b_outliers[notnan_indices(pmo_b_outliers)]
t_b_outliers = t.copy()[notnan_indices(pmo_b_outliers)]

x_a = pmo_a[notnan_indices(pmo_a)]
t_a = t[notnan_indices(pmo_a)]
x_b = pmo_b[notnan_indices(pmo_b)]
t_b = t[notnan_indices(pmo_b)]
x_t = temp[notnan_indices(temp)]
t_t = t[notnan_indices(temp)]
t_e = t[t <= T_EARLY_INCREASE]
x_a_e = pmo_a[t <= T_EARLY_INCREASE]
x_b_e = pmo_b[t <= T_EARLY_INCREASE]
ratio_a_b_nn = np.divide(pmo_a_nn, pmo_b_nn)

figs = []
fig = visualizer.plot_signals([(t_a, x_a, Const.A, False)], results_dir, "{}_raw".format(Const.A), x_ticker=1,
                              legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_a, x_a, Const.A, False),
                               (t_a_outliers, x_a_outliers, Const.A, True)],
                              results_dir, "{}_raw_outliers".format(Const.A), x_ticker=1, legend="upper right",
                              x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_a, x_a, Const.A, False)], results_dir, "{}_raw_closeup".format(Const.A), x_ticker=1,
                              legend="upper right", y_lim=[1357, 1367], x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_b, x_b, Const.B, False)], results_dir, "{}_raw".format(Const.B), x_ticker=1,
                              legend="upper right",
                              x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_b, x_b, Const.B, False),
                               (t_b_outliers, x_b_outliers, Const.B, True)],
                              results_dir, "{}_raw_outliers".format(Const.B), x_ticker=1, legend="upper right",
                              x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_a, x_a, Const.A, False), (t_b, x_b, Const.B, False)], results_dir,
                              "{}_{}_raw".format(Const.A, Const.B), x_ticker=1,
                              legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_a, x_a, Const.A, False), (t_b, x_b, Const.B, False)], results_dir,
                              "{}_{}_raw_closeup".format(Const.A, Const.B),
                              x_ticker=1, legend="upper right", y_lim=[1357, 1369],
                              x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_t, x_t, Const.TEMP, False)], results_dir, "{}_raw".format(Const.TEMP), x_ticker=1,
                              legend="lower right", x_label=Const.YEAR_UNIT, y_label=Const.TEMP_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_e, x_a_e, Const.A, False), (t_e, x_b_e, Const.B, False)], results_dir,
                              "{}_{}_raw_early_increase".format(Const.A, Const.B), x_ticker=5, legend="upper right",
                              x_label=Const.DAY_UNIT, y_label=Const.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_nn, ratio_a_b_nn, "RATIO_{}_{}_nn".format(Const.A, Const.B), False)], results_dir,
                              "RATIO_{}_{}_nn".format(Const.A, Const.B),
                              x_ticker=1, legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.RATIO_UNIT)
figs.append(fig)

if ARGS.visualize:
    for fig in figs:
        fig.show()

if ARGS.sampling:
    sampling_intervals_a = get_sampling_intervals(t, pmo_a)
    sampling_intervals_b = get_sampling_intervals(t, pmo_b)

    counts_a = np.array([interval[1] - interval[0] for interval in sampling_intervals_a
                         if (interval[1] - interval[0]) > SAMPLING_WINDOW])
    counts_b = np.array([interval[1] - interval[0] for interval in sampling_intervals_b
                         if (interval[1] - interval[0]) > SAMPLING_WINDOW])

    starts_a = np.array(list(map(lambda x: x[0], sampling_intervals_a)))
    starts_b = np.array(list(map(lambda x: x[0], sampling_intervals_b)))

    diffs_a = t[starts_a[1:]] - t[starts_a[:-1]]
    diffs_a = diffs_a[diffs_a < 1] * 24 * 60
    diffs_b = t[starts_b[1:]] - t[starts_b[:-1]]

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(16, 16))
    pd.DataFrame(diffs_a).hist(bins=100, ax=ax[0], ec="black")
    ax[0].set_title("Sampling gaps Distribution - {}".format(Const.A))

    pd.DataFrame(counts_a).hist(bins=100, ax=ax[1], ec="black")
    ax[1].set_title("Sampling block lengths Distribution - {}".format(Const.A))

    pd.DataFrame(diffs_b).hist(bins=100, ax=ax[2], ec="black")
    ax[2].set_title("Sampling gaps Distribution - {}".format(Const.B))

    pd.DataFrame(counts_b).hist(bins=100, ax=ax[3], ec="black")
    ax[3].set_title("Sampling block lengths Distribution - {}".format(Const.B))

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "sampling_diffs_counts_raw"))
    if ARGS.visualize:
        plt.show()

    sampling_diffs_b = []
    for interval in sampling_intervals_b:
        if len(interval) > 1:
            interval = np.arange(interval[0], interval[1])
            interval = t[interval]
            diffs = interval[1:] - interval[:-1]
            sampling_diffs_b.extend(diffs)

    sampling_diffs_a = []
    for interval in sampling_intervals_a:
        if len(interval) > 1:
            interval = np.arange(interval[0], interval[1])
            interval = t[interval]
            diffs = interval[1:] - interval[:-1]
            sampling_diffs_a.extend(diffs)

    sampling_diffs_a = np.array(sampling_diffs_a) * 24 * 3600
    sampling_diffs_b = np.array(sampling_diffs_b) * 24 * 3600

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    pd.DataFrame(sampling_diffs_a).hist(bins=100, ax=ax[0], ec="black")
    ax[0].set_title("Sampling period Distribution - {}".format(Const.A))
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10))

    pd.DataFrame(sampling_diffs_b).hist(bins=100, ax=ax[1], ec="black")
    ax[1].set_title("Sampling period Distribution - {}".format(Const.B))
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "sampling_period_distributions_raw"))
    if ARGS.visualize:
        plt.show()

if ARGS.fft:
    # NOT FINISHED
    SAMPLING_PERIOD_B = 1 / 24
    INTERPOLATION = "linear"

    x_b = pmo_b[notnan_indices(pmo_b)]
    t_b = t[notnan_indices(pmo_b)]

    pmo_b_inter_func = interp1d(t_b, x_b, kind=INTERPOLATION)
    t_b_inter = np.arange(t_b.min(), t_b.max(), SAMPLING_PERIOD_B)
    x_b_inter = pmo_b_inter_func(t_b_inter)

    t_b = np.array(list(map(mission_day_to_year, t_b)))
    t_b_inter = np.array(list(map(mission_day_to_year, t_b_inter)))

    fig = visualizer.plot_signals([(t_b, x_b, Const.B, False),
                                   (t_b_inter, x_b_inter,
                                    "{}_{}_interpolation".format(Const.B, INTERPOLATION), False)],
                                  None, "{}_raw_{}_interpolation".format(Const.B, INTERPOLATION), legend="upper right",
                                  x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    if ARGS.visualize:
        fig.show()

    N = x_b_inter.shape[0]
    X_FFT = np.fft.fft(x_b_inter - x_b.mean()) / N
    S = np.square(np.abs(X_FFT))

    F_0 = 1 / N
    fs = 1 / SAMPLING_PERIOD_B
    k = np.arange(N)
    frq = k * F_0 * fs
    frq_range = fs / 128

    plt.figure()
    fig = visualizer.plot_signals([(frq[frq <= frq_range], S[frq <= frq_range], "{}_spectrum".format(Const.B))],
                                  results_dir, "{}_spectrum".format(Const.B), legend="upper right",
                                  x_label=Const.FREQ_DAY_UNIT, y_label=Const.SPECTRUM_UNIT)

    if ARGS.visualize:
        plt.show()
