import logging
import os

import numpy as np

from dslab_virgo_tsi.base import Result, ModelFitter, FitResult, BaseSignals, OutResult, \
    FinalResult, Mode
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.run_utils import setup_run, create_results_dir, create_logger, save_modeling_result, \
    parse_arguments, load_data_run
from dslab_virgo_tsi.visualizer import Visualizer

"""
--random_seed = 0
--save_plots = store_true
--save_signals = store_true

--model_type = "smooth_monotonic"
--correction_method = "one"
--outlier_fraction = 0.0
--exposure_mode = "measurements"
--output_method = "svgp
"""


def plot_results(ground_truth_, result_: Result, results_dir, model_name):
    before_fit: FitResult = result_.history_mutual_nn[0]
    last_iter: FitResult = result_.history_mutual_nn[-1]

    base_sig: BaseSignals = result_.base_signals
    final_res: FinalResult = result_.final
    out_res: OutResult = result_.out

    t_, x_ = ground_truth_

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
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"gen_hourly_{model_name}")
        ],
        results_dir, f"gen_hourly_{model_name}_points", ground_truth_triplet=(t_, x_, "ground_truth"),
        data_points_triplets=[
            (base_sig.t_a_nn, final_res.a_nn_corrected, "A_raw_corrected"),
            (base_sig.t_b_nn, final_res.b_nn_corrected, "B_raw_corrected")
        ],
        legend="upper left", x_label="t", y_label="x(t)")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    visualizer = Visualizer()
    visualizer.set_figsize()

    ARGS = parse_arguments()
    results_dir_path = create_results_dir(Const.RESULTS_DIR, f"gen_{ARGS.model_type}")
    create_logger(results_dir_path)

    mode = Mode.GENERATOR

    data, t_field_name, a_field_name, b_field_name, ground_truth = load_data_run(ARGS, mode)
    model, model_type, correction_method, exposure_method, output_method, outlier_fraction \
        = setup_run(ARGS, mode, results_dir_path)

    fitter = ModelFitter(mode=mode,
                         data=data,
                         t_field_name=t_field_name,
                         a_field_name=a_field_name,
                         b_field_name=b_field_name,
                         exposure_method=exposure_method,
                         outlier_fraction=outlier_fraction)

    result: Result = fitter(model=model,
                            correction_method=correction_method,
                            output_method=output_method)

    if ARGS.save_signals:
        save_modeling_result(results_dir_path, result, f"gen_{ARGS.model_type}")

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(ground_truth, result, results_dir_path, f"gen_{ARGS.model_type}")

    logging.info("Application finished.")
