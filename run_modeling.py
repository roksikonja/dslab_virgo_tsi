import logging
import os

import numpy as np

from dslab_virgo_tsi.base import Result, FitResult, ModelFitter, BaseSignals, OutResult, FinalResult, Mode
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
--exposure_method = "measurements"
--output_method = "svgp
"""


def plot_results(result_: Result, results_dir, model_name):
    before_fit: FitResult = result_.history_mutual_nn[0]
    last_iter: FitResult = result_.history_mutual_nn[-1]

    base_sig: BaseSignals = result_.base_signals
    out_res: OutResult = result_.out
    final_res: FinalResult = result_.final

    logging.info("Plotting results ...")
    visualizer.plot_signals(
        [
            (base_sig.t_a_nn, final_res.degradation_a_nn, f"DEGRADATION_{Const.A}", False),
            (base_sig.t_b_nn, final_res.degradation_b_nn, f"DEGRADATION_{Const.B}", False)
        ],
        results_dir, f"DEGRADATION_{Const.A}_{Const.B}_{model_name}", x_ticker=1, legend="upper right",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.a_mutual_nn_corrected, f"{Const.A}_mutual_nn", False),
            (base_sig.t_mutual_nn, before_fit.b_mutual_nn_corrected, f"{Const.B}_mutual_nn", False),
            (base_sig.t_mutual_nn, last_iter.a_mutual_nn_corrected, f"{Const.A}_mutual_nn_corrected", False),
            (base_sig.t_mutual_nn, last_iter.b_mutual_nn_corrected, f"{Const.B}_mutual_nn_corrected", False)
        ],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_mutual_corrected", x_ticker=1, legend="upper right",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.ratio_a_b_mutual_nn_corrected, f"RATIO_{Const.A}_{Const.B}_raw", False),
            (base_sig.t_mutual_nn, last_iter.ratio_a_b_mutual_nn_corrected, f"RATIO_{Const.A}_not_{Const.B}_corrected",
             False),
            (base_sig.t_mutual_nn, np.divide(last_iter.a_mutual_nn_corrected, last_iter.b_mutual_nn_corrected),
             f"RATIO_{Const.A}_corrected_{Const.B}_corrected", False)
        ],
        results_dir, f"{model_name}_RATIO_DEGRADATION_{Const.A}_{Const.B}_raw_corrected", x_ticker=1,
        legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals(
        [
            (base_sig.t_a_nn, base_sig.a_nn, f"{Const.A}_raw", False),
            (base_sig.t_b_nn, base_sig.b_nn, f"{Const.B}_raw", False),
            (base_sig.t_a_nn, final_res.a_nn_corrected, f"{Const.A}_raw_corrected", False),
            (base_sig.t_b_nn, final_res.b_nn_corrected, f"{Const.B}_raw_corrected", False)
        ],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_raw_corrected_full", x_ticker=1, y_lim=[1357, 1369],
        legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"TSI_hourly_{model_name}")
        ],
        results_dir, f"TSI_hourly_{model_name}", x_ticker=1, legend="upper left", y_lim=[1362, 1369],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, max_points=1e7)

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"TSI_hourly_{model_name}")
        ],
        results_dir, f"TSI_hourly_{model_name}_points", x_ticker=1, legend="upper left", y_lim=[1362, 1369],
        data_points_triplets=[
            (base_sig.t_a_nn, final_res.a_nn_corrected, f"{Const.A}_raw_corrected"),
            (base_sig.t_b_nn, final_res.b_nn_corrected, f"{Const.B}_raw_corrected")
        ],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, max_points=1e7)

    visualizer.plot_signal_history(base_sig.t_mutual_nn, result_.history_mutual_nn,
                                   results_dir, f"{model_name}_history",
                                   ground_truth_triplet=None,
                                   legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    visualizer = Visualizer()
    visualizer.set_figsize()

    ARGS = parse_arguments()
    results_dir_path = create_results_dir(Const.RESULTS_DIR, ARGS.model_type)
    create_logger(results_dir_path)

    mode = Mode.VIRGO

    data, t_field_name, a_field_name, b_field_name, _ = load_data_run(ARGS, mode)
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
        save_modeling_result(results_dir_path, result, model_type)

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(result, results_dir_path, model_type)

    logging.info("Application finished.")
