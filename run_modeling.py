import argparse
import datetime
import os

import matplotlib.pyplot as plt

from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import load_data, make_dir
from dslab_virgo_tsi.models import ExposureMode, ExpModel, ExpLinModel, ModelingResult
from dslab_virgo_tsi.visualizer import Visualizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="exp_lin", help="Model to train.")
    parser.add_argument("--visualize", action="store_true", help="Flag for visualizing results.")
    parser.add_argument("--window", type=int, default=81, help="Moving average window size.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")

    return parser.parse_args()


def create_results_dir():
    results_dir = os.path.join(Const.RESULTS_DIR, datetime.datetime.now().strftime("modeling_%Y-%m-%d"))
    make_dir(results_dir)
    return results_dir


def plot_results(results: ModelingResult, results_dir, model_name, window_size):
    figs = []
    fig = Visualizer.plot_signals([(results.t_mutual_nn, results.history_mutual_nn[0].iteration_ratio_a_b,
                                    "RATIO_{}_{}_raw".format(Const.A, Const.B), False)],
                                  results_dir, "RATIO_{}_{}_raw_initial_fit".format(Const.A, Const.B), x_ticker=1,
                                  legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = Visualizer.plot_signals(
        [(results.t_mutual_nn, results.history_mutual_nn[0].iteration_signal_a, "{}_raw_nn".format(Const.A), False),
         (results.t_mutual_nn, results.history_mutual_nn[0].iteration_signal_b, "{}_raw_nn".format(Const.B), False),
         (results.t_mutual_nn, results.history_mutual_nn[-1].iteration_signal_a, "{}_raw_nn_corrected".format(Const.A),
          False),
         (
             results.t_mutual_nn, results.history_mutual_nn[-1].iteration_signal_b,
             "{}_raw_nn_corrected".format(Const.B), False)],
        results_dir, "{}_{}_{}_raw_corrected".format(model_name, Const.A, Const.B), x_ticker=1,
        legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = Visualizer.plot_signals([(results.t_mutual_nn, results.history_mutual_nn[0].iteration_ratio_a_b,
                                    "RATIO_{}_{}_raw".format(Const.A, Const.B), False),
                                   (results.t_mutual_nn, results.history_mutual_nn[-1].iteration_ratio_a_b,
                                    "RATIO_{}_{}_corrected".format(Const.A, Const.B), False)],
                                  results_dir,
                                  "{}_RATIO_DEGRADATION_{}_{}_raw_corrected".format(model_name, Const.A, Const.B),
                                  x_ticker=1, legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = Visualizer.plot_signals([(results.t_a_nn, results.signal_a_nn, "{}_raw".format(Const.A), False),
                                   (results.t_b_nn, results.signal_b_nn, "{}_raw".format(Const.B), False),
                                   (results.t_a_nn, results.signal_a_nn_corrected, "{}_raw_corrected".format(Const.A),
                                    False),
                                   (results.t_b_nn, results.signal_b_nn_corrected, "{}_raw_corrected".format(Const.B),
                                    False)],
                                  results_dir, "{}_{}_{}_raw_corrected_full".format(model_name, Const.A, Const.B),
                                  x_ticker=1,
                                  legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = Visualizer.plot_signals_mean_std(
        [(results.t_a_nn, results.signal_a_nn, "{}_conf_int".format(Const.A), window_size),
         (results.t_b_nn, results.signal_b_nn, "{}_conf_int".format(Const.B), window_size),
         (results.t_a_nn, results.signal_a_nn_corrected, "{}_corrected_conf_int".format(Const.A), window_size),
         (results.t_b_nn, results.signal_b_nn_corrected, "{}_corrected_conf_int".format(Const.B), window_size)],
        results_dir, "{}_{}_{}_raw_corrected_full_conf_int".format(model_name, Const.A, Const.B),
        x_ticker=1, legend="lower left", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)


if __name__ == "__main__":
    ARGS = parse_arguments()
    results_dir_path = create_results_dir()

    # Load data
    data_pmo6v = load_data(os.path.join(Const.DATA_DIR, Const.VIRGO_FILE))

    # Perform modeling
    model = None
    if ARGS.model_type == "exp_lin":
        model = ExpLinModel(data=data_pmo6v,
                            timestamp_field_name=Const.T,
                            signal_a_field_name=Const.A,
                            signal_b_field_name=Const.B,
                            exposure_mode=ExposureMode.NUM_MEASUREMENTS)
    elif ARGS.model_type == "exp":
        model = ExpModel(data=data_pmo6v,
                         timestamp_field_name=Const.T,
                         signal_a_field_name=Const.A,
                         signal_b_field_name=Const.B,
                         exposure_mode=ExposureMode.EXPOSURE_SUM)

    result = model.get_result()
    plot_results(result, results_dir_path, ARGS.model_type, ARGS.window)

    if ARGS.visualize:
        plt.show()
