import argparse
import logging

import numpy as np
import pandas as pd

from dslab_virgo_tsi.base import Result, ModelFitter, FitResult, BaseSignals, OutResult, \
    FinalResult
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.generator import SignalGenerator
from dslab_virgo_tsi.run_utils import setup_run, create_results_dir, create_logger, save_modeling_result
from dslab_virgo_tsi.visualizer import Visualizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_length", type=int, default=100000, help="Generated signal length.")
    parser.add_argument("--degradation_model", type=str, default="exp", help="Model to train.")
    parser.add_argument("--degradation_rate", type=float, default=1.0, help="Tuning parameter for degradation.")
    parser.add_argument("--random_seed", type=int, default=0, help="Set random seed.")

    parser.add_argument("--save_plots", action="store_true", help="Flag for saving plots.")
    parser.add_argument("--save_signals", action="store_true", help="Flag for saving computed signals.")

    parser.add_argument("--model_type", type=str, default="smooth_monotonic", help="Model to train.")
    parser.add_argument("--correction_method", type=str, default="one", help="Iterative correction method.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")
    parser.add_argument("--exposure_mode", type=str, default="measurements", help="Exposure computing method.")
    parser.add_argument("--output_method", type=str, default="svgp", help="Exposure computing method.")

    return parser.parse_args()


def plot_results(t_, x_, result_: Result, results_dir, model_name):
    before_fit: FitResult = result_.history_mutual_nn[0]
    last_iter: FitResult = result_.history_mutual_nn[-1]

    base_sig: BaseSignals = result_.base_signals
    final_res: FinalResult = result_.final
    out_res: OutResult = result_.out

    logging.info("Plotting results ...")
    visualizer.plot_signals(
        [
            (base_sig.t_a_nn, final_res.degradation_a_nn, "DEGRADATION_A", False),
            (base_sig.t_b_nn, final_res.degradation_b_nn, "DEGRADATION_B", False)
        ],
        results_dir, f"DEGRADATION_{model_name}", legend="upper right",
        x_label="t", y_label="d(t)")

    visualizer.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.a_mutual_nn_corrected, "A_mutual_nn", False),
            (base_sig.t_mutual_nn, before_fit.b_mutual_nn_corrected, "B_mutual_nn", False),
            (base_sig.t_mutual_nn, last_iter.a_mutual_nn_corrected, "A_mutual_nn_corrected", False),
            (base_sig.t_mutual_nn, last_iter.b_mutual_nn_corrected, "B_mutual_nn_corrected", False),
            (t_, x_, "ground_truth", False),
        ],
        results_dir, f"{model_name}_mutual_corrected", legend="upper right",
        x_label="t", y_label="x(t)")

    visualizer.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.ratio_a_b_mutual_nn_corrected, f"RATIO_A_B_raw", False),
            (base_sig.t_mutual_nn, last_iter.ratio_a_b_mutual_nn_corrected, f"RATIO_A_not_B_corrected",
             False),
            (base_sig.t_mutual_nn, np.divide(last_iter.a_mutual_nn_corrected, last_iter.b_mutual_nn_corrected),
             f"RATIO_A_corrected_B_corrected", False)
        ],
        results_dir, f"{model_name}_RATIO_DEGRADATION_A_B_raw_corrected",
        legend="upper right", x_label="t", y_label="r(t)")

    visualizer.plot_signals(
        [
            (base_sig.t_a_nn, base_sig.a_nn, "A_raw", False),
            (base_sig.t_b_nn, base_sig.b_nn, "B_raw", False),
            (base_sig.t_a_nn, final_res.a_nn_corrected, "A_raw_corrected", False),
            (base_sig.t_b_nn, final_res.b_nn_corrected, "B_raw_corrected", False),
            (t_, x_, "ground_truth", False),
        ],
        results_dir, f"{model_name}_A_B_raw_corrected_full",
        legend="upper right", x_label="t", y_label="x(t)")

    visualizer.plot_signal_history(base_sig.t_mutual_nn, result_.history_mutual_nn,
                                   results_dir, f"{model_name}_history",
                                   ground_truth_triplet=(t_, x_, "ground_truth"),
                                   legend="upper right", x_label="t", y_label="x(t)")

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"gen_hourly_{model_name}")
        ],
        results_dir, f"gen_hourly_{model_name}", ground_truth_triplet=(t_, x_, "ground_truth"),

        legend="upper left", x_label="t", y_label="x(t)")

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_daily_out, out_res.signal_daily_out, out_res.signal_std_daily_out, f"gen_daily_{model_name}")
        ],
        results_dir, f"gen_daily_{model_name}", ground_truth_triplet=(t_, x_, "ground_truth"),
        legend="upper left", x_label="t", y_label="x(t)")

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"gen_hourly_{model_name}")
        ],
        results_dir, f"gen_hourly_{model_name}_points", ground_truth_triplet=(t_, x_, "ground_truth"),
        data_points_triplets=[
            (base_sig.t_a_nn, final_res.a_nn_corrected, "A_raw_corrected"),
            (base_sig.t_b_nn, final_res.b_nn_corrected, "B_raw_corrected")
        ],
        legend="upper left", x_label="t", y_label="x(t)")

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_daily_out, out_res.signal_daily_out, out_res.signal_std_daily_out, f"gen_daily_{model_name}")
        ],
        results_dir, f"gen_daily_{model_name}_points", ground_truth_triplet=(t_, x_, "ground_truth"),
        data_points_triplets=[
            (base_sig.t_a_nn, final_res.a_nn_corrected, "A_raw_corrected"),
            (base_sig.t_b_nn, final_res.b_nn_corrected, "B_raw_corrected")
        ],
        legend="upper left", x_label="t", y_label="x(t)")


if __name__ == "__main__":
    ARGS = parse_arguments()

    data_dir = Const.DATA_DIR
    results_dir_path = create_results_dir(Const.RESULTS_DIR, f"gen_{ARGS.model_type}")
    create_logger(results_dir_path)

    visualizer = Visualizer()
    visualizer.set_figsize()

    model, mode, model_type, correction_method, exposure_mode, output_method, outlier_fraction \
        = setup_run(ARGS, "gen", results_dir_path)

    # Generator
    Generator = SignalGenerator(ARGS.signal_length, ARGS.random_seed, exposure_mode)
    t = Generator.time
    x = Generator.x
    x_a_raw, x_b_raw, _ = Generator.generate_raw_signal(x, 5, rate=ARGS.degradation_rate)

    T, X_A, X_B = "t", "x_a", "x_b"
    data_gen = pd.DataFrame()
    data_gen[T] = t
    data_gen[X_A] = x_a_raw
    data_gen[X_B] = x_b_raw
    logging.info(f"Data generator loaded.")

    fitter = ModelFitter(mode=mode,
                         data=data_gen,
                         t_field_name=T,
                         a_field_name=X_A,
                         b_field_name=X_B,
                         exposure_mode=exposure_mode,
                         outlier_fraction=outlier_fraction)

    result: Result = fitter(model=model,
                            correction_method=correction_method,
                            output_method=output_method)

    if ARGS.save_signals:
        save_modeling_result(results_dir_path, result, f"gen_{ARGS.model_type}")

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(t, x, result, results_dir_path, f"gen_{ARGS.model_type}")

    logging.info("Application finished.")
