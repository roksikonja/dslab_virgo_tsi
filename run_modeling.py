import argparse
import datetime
import os
import pickle

import matplotlib.pyplot as plt

from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import load_data, make_dir
from dslab_virgo_tsi.models import ExposureMode, ExpModel, ExpLinModel, ModelingResult, IterationResult, SplineModel
from dslab_virgo_tsi.visualizer import Visualizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="spline", help="Model to train.")
    parser.add_argument("--reuse", action="store_true", help="Flag for reusing previous results.")
    parser.add_argument("--visualize", action="store_true", help="Flag for visualizing results.")
    parser.add_argument("--window", type=int, default=81, help="Moving average window size.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")

    return parser.parse_args()


def create_results_dir():
    results_dir = make_dir(os.path.join(Const.RESULTS_DIR, datetime.datetime.now().strftime("modeling_%Y-%m-%d")))
    return results_dir


def save_modeling_result(results_dir, model_results, model_name):
    with open(os.path.join(results_dir, "{}_modeling_result.pkl".format(model_name)), 'wb') as f:
        pickle.dump(model_results, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_modeling_result(results_dir, model_name):
    with open(os.path.join(results_dir, "{}_modeling_result.pkl".format(model_name)), 'rb') as f:
        model_results = pickle.load(f)
    return model_results


def plot_results(results: ModelingResult, results_dir, model_name, window_size):
    first_iteration: IterationResult = results.history_mutual_nn[0]
    last_iteration: IterationResult = results.history_mutual_nn[-1]

    print("plotting results ...")
    figs = []

    fig = visualizer.plot_signals(
        [(results.t_mutual_nn, first_iteration.ratio_a_b, f"RATIO_{Const.A}_{Const.B}_raw", False)],
        results_dir, f"RATIO_{Const.A}_{Const.B}_raw_initial_fit", x_ticker=1, legend="upper right",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals(
        [(results.t_a_nn, results.degradation_a, f"DEGRADATION_{Const.A}", False),
         (results.t_b_nn, results.degradation_b, f"DEGRADATION_{Const.B}", False)],
        results_dir, f"DEGRADATION_{Const.A}_{Const.B}_{model_name}", x_ticker=1, legend="upper right",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals(
        [(results.t_mutual_nn, first_iteration.a, f"{Const.A}_raw_nn", False),
         (results.t_mutual_nn, first_iteration.b, f"{Const.B}_raw_nn", False),
         (results.t_mutual_nn, last_iteration.a, f"{Const.A}_raw_nn_corrected", False),
         (results.t_mutual_nn, last_iteration.b, f"{Const.B}_raw_nn_corrected", False)],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_raw_corrected", x_ticker=1, legend="upper right",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals(
        [(results.t_mutual_nn, first_iteration.ratio_a_b, f"RATIO_{Const.A}_{Const.B}_raw", False),
         (results.t_mutual_nn, last_iteration.ratio_a_b, f"RATIO_{Const.A}_{Const.B}_corrected", False)],
        results_dir, f"{model_name}_RATIO_DEGRADATION_{Const.A}_{Const.B}_raw_corrected", x_ticker=1,
        legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals(
        [(results.t_a_nn, results.a_nn, f"{Const.A}_raw", False),
         (results.t_b_nn, results.b_nn, f"{Const.B}_raw", False),
         (results.t_a_nn, results.a_nn_corrected, f"{Const.A}_raw_corrected", False),
         (results.t_b_nn, results.b_nn_corrected, f"{Const.B}_raw_corrected", False)],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_raw_corrected_full", x_ticker=1, y_lim=[1357, 1369],
        legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals_mean_std_precompute(
        [(results.t_daily_out, results.signal_daily_out, results.signal_std_daily_out, f"TSI_daily_{model_name}")],
        results_dir, f"TSI_daily_{model_name}", x_ticker=1, legend="upper left", y_lim=[1357, 1369],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals_mean_std_precompute(
        [(results.t_hourly_out, results.signal_hourly_out, results.signal_std_hourly_out, f"TSI_hourly_{model_name}")],
        results_dir, f"TSI_hourly_{model_name}", x_ticker=1, legend="upper left", y_lim=[1357, 1369],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals_mean_std(
        [(results.t_a_nn, results.a_nn, f"{Const.A}_conf_int", window_size),
         (results.t_b_nn, results.b_nn, f"{Const.B}_conf_int", window_size),
         (results.t_a_nn, results.a_nn_corrected, f"{Const.A}_corrected_conf_int", window_size),
         (results.t_b_nn, results.b_nn_corrected, f"{Const.B}_corrected_conf_int", window_size)],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_raw_corrected_full_conf_int", x_ticker=1, legend="lower left",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, y_lim=[1357, 1369])
    figs.append(fig)

    """
    fig = visualizer.plot_signals(
        [(results.t_hourly_out, results.signal_hourly_out, f"TSI_hourly_{model_name}", False),
         (results.t_a_nn, results.a_nn_corrected, f"{Const.A}_raw_corrected", False),
         (results.t_b_nn, results.b_nn_corrected, f"{Const.B}_raw_corrected", False)],
        results_dir, f"TSI_{model_name}_{Const.A}_hourly_{Const.B}", x_ticker=1, legend="upper left",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, y_lim=[1357, 1369])
    figs.append(fig)

    fig = visualizer.plot_signals(
        [(results.t_hourly_out, results.signal_hourly_out, f"TSI_hourly_{model_name}", False)],
        results_dir, f"TSI_hourly_{model_name}", x_ticker=1, legend="upper left", y_lim=[1357, 1369],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)

    fig = visualizer.plot_signals(
        [(results.t_daily_out, results.signal_daily_out, "TSI_daily_{}".format(model_name), False)],
        results_dir, f"TSI_daily_{model_name}", x_ticker=1, legend="upper left", y_lim=[1357, 1369],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)
    figs.append(fig)
    """


if __name__ == "__main__":
    ARGS = parse_arguments()
    results_dir_path = create_results_dir()

    # Load data
    data_pmo6v = load_data(Const.DATA_DIR, Const.VIRGO_FILE)

    visualizer = Visualizer()
    visualizer.set_figsize()

    # Perform modeling
    model = None
    if not ARGS.reuse:
        if ARGS.model_type == "exp_lin":
            model = ExpLinModel(data=data_pmo6v,
                                t_field_name=Const.T,
                                a_field_name=Const.A,
                                b_field_name=Const.B,
                                exposure_mode=ExposureMode.NUM_MEASUREMENTS,
                                moving_average_window=ARGS.window,
                                outlier_fraction=ARGS.outlier_fraction)
        elif ARGS.model_type == "exp":
            model = ExpModel(data=data_pmo6v,
                             t_field_name=Const.T,
                             a_field_name=Const.A,
                             b_field_name=Const.B,
                             exposure_mode=ExposureMode.EXPOSURE_SUM,
                             moving_average_window=ARGS.window,
                             outlier_fraction=ARGS.outlier_fraction)
        elif ARGS.model_type == "spline":
            model = SplineModel(data=data_pmo6v,
                                t_field_name=Const.T,
                                a_field_name=Const.A,
                                b_field_name=Const.B,
                                exposure_mode=ExposureMode.EXPOSURE_SUM,
                                moving_average_window=ARGS.window,
                                outlier_fraction=ARGS.outlier_fraction)

        result = model.get_result()
        save_modeling_result(results_dir_path, result, ARGS.model_type)
    else:
        result = load_modeling_result(results_dir_path, ARGS.model_type)

    # result.downsample_signals(k_a=1000, k_b=10)
    plot_results(result, results_dir_path, ARGS.model_type, ARGS.window)

    if ARGS.visualize:
        plt.show()
