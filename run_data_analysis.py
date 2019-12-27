import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.interpolate import interp1d

from dslab_virgo_tsi.base import ExposureMethod, ModelFitter, BaseSignals
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import load_data, notnan_indices, \
    get_sampling_intervals, fft_spectrum
from dslab_virgo_tsi.run_utils import create_logger, create_results_dir
from dslab_virgo_tsi.visualizer import Visualizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="virgo", help="Choose data file.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")
    parser.add_argument("--exposure_mode", type=str, default="measurements", help="Exposure computing method.")

    parser.add_argument("--sampling", action="store_true", help="Flag for sampling analysis.")
    parser.add_argument("--fft", action="store_true", help="Flag for FFT analysis.")
    parser.add_argument("--t_early_increase", type=int, default=100, help="Early increase time span.")
    return parser.parse_args()


def plot_base_virgo_tsi_signals(other_res_):
    other_tsi = other_res_[Const.PMO6V_OLD].values
    other_t = other_res_[Const.T].values
    other_fourplet = (other_t, other_tsi, f"{Const.PMO6V_OLD}_corrected", False)

    visualizer.plot_signals([other_fourplet],
                            results_dir_path, "{}_raw".format(Const.PMO6V_OLD), x_ticker=Const.XTICKER,
                            legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)


def plot_base_virgo_signals(base_sig):
    logging.info("Plotting results ...")

    visualizer.plot_signals([(base_sig.t_a_nn, base_sig.a_nn, Const.A, False)],
                            results_dir_path, "{}_raw".format(Const.A), x_ticker=Const.XTICKER,
                            legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals([(base_sig.t_a_nn, base_sig.a_nn, Const.A, False)],
                            results_dir_path, "{}_raw_closeup".format(Const.A), x_ticker=Const.XTICKER,
                            y_lim=[1357, 1367],
                            legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals([(base_sig.t_b_nn, base_sig.b_nn, Const.B, False)],
                            results_dir_path, "{}_raw".format(Const.B), x_ticker=Const.XTICKER,
                            legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals([(base_sig.t_a_nn, base_sig.a_nn, Const.A, False),
                             (base_sig.t_b_nn, base_sig.b_nn, Const.B, False)],
                            results_dir_path, "{}_{}_raw_closeup".format(Const.A, Const.B), x_ticker=Const.XTICKER,
                            y_lim=[1357, 1368], legend="upper right",
                            x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals([(base_sig.t_mutual_nn, np.divide(base_sig.a_mutual_nn, base_sig.b_mutual_nn),
                              "RATIO_{}_{}_nn".format(Const.A, Const.B), False)],
                            results_dir_path, "RATIO_{}_{}_nn".format(Const.A, Const.B),
                            x_ticker=Const.XTICKER, legend="upper right", x_label=Const.YEAR_UNIT,
                            y_label=Const.RATIO_UNIT)

    visualizer.plot_signals([(base_sig.t_temp_nn, base_sig.temp_nn, Const.TEMP, False)],
                            results_dir_path, "{}_raw".format(Const.TEMP), x_ticker=Const.XTICKER,
                            legend="lower right", x_label=Const.YEAR_UNIT, y_label=Const.TEMP_UNIT)

    visualizer.plot_signals([(base_sig.t_ei, base_sig.a_ei, Const.A, False),
                             (base_sig.t_ei, base_sig.b_ei, Const.A, False)],
                            results_dir_path, "{}_{}_raw_early_increase".format(Const.A, Const.B), x_ticker=5,
                            legend="upper right", x_label=Const.DAY_UNIT, y_label=Const.TSI_UNIT)


def analyse_virgo_data(data_, base_sig: BaseSignals, t_early_increase=100):
    logging.info("Analysing virgo data.")
    t = data_[Const.T].values
    temp_nn = data_[Const.TEMP].values

    t_temp_nn = t[notnan_indices(temp_nn)]
    temp_nn = temp_nn[notnan_indices(temp_nn)]

    t_ei = t[t <= t_early_increase]
    a_ei = data_[Const.A][t <= t_early_increase]
    b_ei = data_[Const.B][t <= t_early_increase]

    base_sig.t_temp_nn = t_temp_nn
    base_sig.temp_nn = temp_nn
    base_sig.t_ei = t_ei
    base_sig.a_ei = a_ei
    base_sig.b_ei = b_ei
    return base_sig


def plot_analyse_virgo_sampling(data_, sampling_window=5):
    logging.info("Sampling analysis started.")

    t = data_[Const.T]
    a = data_[Const.A]
    b = data_[Const.B]

    sampling_intervals_a = get_sampling_intervals(t, a)
    sampling_intervals_b = get_sampling_intervals(t, b)

    counts_a = np.array([interval[1] - interval[0] for interval in sampling_intervals_a
                         if (interval[1] - interval[0]) > sampling_window])
    counts_b = np.array([interval[1] - interval[0] for interval in sampling_intervals_b
                         if (interval[1] - interval[0]) > sampling_window])

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
    fig.savefig(os.path.join(results_dir_path, "sampling_diffs_counts_raw"))
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
    fig.savefig(os.path.join(results_dir_path, "sampling_period_distributions_raw"))
    if ARGS.visualize:
        plt.show()


def plot_analyse_virgo_fft(base_sig: BaseSignals):
    logging.info("FFT analysis started.")
    sampling_period = 100 / 24

    a_inter_func = interp1d(base_sig.t_a_nn, base_sig.a_nn, kind="linear")
    b_inter_func = interp1d(base_sig.t_b_nn, base_sig.b_nn, kind="linear")

    t_a_inter = np.arange(base_sig.t_a_nn.min(), base_sig.t_a_nn.max(), sampling_period)
    t_b_inter = np.arange(base_sig.t_b_nn.min(), base_sig.t_b_nn.max(), sampling_period)

    a_inter = a_inter_func(t_a_inter)
    b_inter = b_inter_func(t_b_inter)

    a_s, fs, frq = fft_spectrum(a_inter, sampling_period)
    frq_range = fs / 2

    visualizer.plot_signals([(frq[frq <= frq_range], a_s[frq <= frq_range], "{}_spectrum".format(Const.A), False)],
                            results_dir_path, "{}_spectrum".format(Const.A), legend="upper right",
                            y_label=Const.SPECTRUM_UNIT)

    b_s, fs, frq = fft_spectrum(b_inter, sampling_period)
    frq_range = fs / 2

    visualizer.plot_signals([(frq[frq <= frq_range], b_s[frq <= frq_range], "{}_spectrum".format(Const.B), False)],
                            results_dir_path, "{}_spectrum".format(Const.B), legend="upper right",
                            y_label=Const.SPECTRUM_UNIT)


if __name__ == "__main__":
    ARGS = parse_arguments()
    results_dir_path = create_results_dir(Const.RESULTS_DIR, "data_analysis")
    create_logger(results_dir_path)

    visualizer = Visualizer()
    visualizer.set_figsize()

    # Load data
    data = None
    if ARGS.data_file == "premos":
        pass
    elif ARGS.data_file == "virgo_tsi":
        other_res = load_data(Const.DATA_DIR, Const.VIRGO_TSI_FILE, "virgo_tsi")
        plot_base_virgo_tsi_signals(other_res)
    else:
        data = load_data(Const.DATA_DIR, Const.VIRGO_FILE)
        logging.info(f"Data {Const.VIRGO_FILE} loaded.")

        if ARGS.exposure_mode == "measurements":
            exposure_mode = ExposureMethod.NUM_MEASUREMENTS
        else:
            exposure_mode = ExposureMethod.EXPOSURE_SUM

        fitter = ModelFitter(data=data,
                             t_field_name=Const.T,
                             a_field_name=Const.A,
                             b_field_name=Const.B,
                             exposure_method=ExposureMethod.NUM_MEASUREMENTS,
                             outlier_fraction=ARGS.outlier_fraction)

        base_signals = analyse_virgo_data(data, fitter.base_signals)
        plot_base_virgo_signals(base_signals)

        if ARGS.fft:
            plot_analyse_virgo_fft(fitter.base_signals)

        if ARGS.sampling:
            plot_analyse_virgo_sampling(data)
