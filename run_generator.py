import logging

import numpy as np

from dslab_virgo_tsi.base import Result, ModelFitter, FitResult, BaseSignals, OutResult, \
    FinalResult, Mode
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.run_utils import setup_run, create_results_dir, create_logger, save_modeling_result, \
    parse_arguments, load_data_run, ignore_warnings
from dslab_virgo_tsi.visualizer import Visualizer


def plot_results(ground_truth_, result_: Result, results_dir, model_name):
    before_fit: FitResult = result_.history_mutual_nn[0]
    last_iter: FitResult = result_.history_mutual_nn[-1]

    base_sig: BaseSignals = result_.base_signals
    final_res: FinalResult = result_.final
    out_res: OutResult = result_.out

    t_, x_ = ground_truth_

    logging.info("Plotting results ...")
    if out_res.params_out.svgp_iter_loglikelihood:
        visualizer.plot_iter_loglikelihood(out_res.params_out.svgp_iter_loglikelihood, results_dir,
                                           f"{model_name}_ITER_LOGLIKELIHOOD", legend="lower left",
                                           x_label=Const.ITERATION_UNIT, y_label=Const.LOG_LIKELIHOOD_UNIT)

    visualizer.plot_signals(
        [
            (base_sig.t_a_nn, final_res.degradation_a_nn, "DEGRADATION_A", False),
            (base_sig.t_b_nn, final_res.degradation_b_nn, "DEGRADATION_B", False)
        ],
        results_dir, f"{model_name}_DEGRADATION", legend="upper right",
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
        ],
        results_dir, f"{model_name}_RATIO_DEGRADATION_A_B_raw",
        legend="upper right", x_label="t", y_label="r(t)")

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

    visualizer.plot_signals(
        [
            (base_sig.t_a_nn, base_sig.a_nn, "A_raw", False),
            (base_sig.t_b_nn, base_sig.b_nn, "B_raw", False),
            (t_, x_, "ground_truth", False),
        ],
        results_dir, f"{model_name}_A_B_raw_full",
        legend="upper right", x_label="t", y_label="x(t)")

    visualizer.plot_signals(
        [
            (base_sig.t_a_nn, final_res.a_nn_corrected, "A_raw_corrected", False),
            (base_sig.t_b_nn, final_res.b_nn_corrected, "B_raw_corrected", False),
            (t_, x_, "ground_truth", False),
        ],
        results_dir, f"{model_name}_A_B_corrected_full",
        legend="upper right", x_label="t", y_label="x(t)")

    visualizer.plot_signal_history(base_sig.t_mutual_nn, result_.history_mutual_nn,
                                   results_dir, f"{model_name}_history",
                                   ground_truth_triplet=(t_, x_, "ground_truth"),
                                   legend="upper right", x_label="t", y_label="x(t)")

    visualizer.plot_signal_history_report(base_sig.t_mutual_nn, result_.history_mutual_nn,
                                          results_dir, f"{model_name}_history_report",
                                          ground_truth_triplet=(t_, x_, "ground_truth"), y_lim=[2, 21],
                                          legend="upper left", x_label="t", y_label="x(t)")

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out,
             f"gen_hourly_{model_name}_gt")
        ],
        results_dir, f"{model_name}_hourly_gt", ground_truth_triplet=(t_, x_, "ground_truth"),
        legend="upper left", x_label="t", y_label="x(t)", inducing_points=out_res.params_out.svgp_inducing_points)

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"gen_hourly_{model_name}")
        ],
        results_dir, f"{model_name}_hourly", legend="upper left", x_label="t", y_label="x(t)",
        inducing_points=out_res.params_out.svgp_inducing_points)

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"gen_hourly_{model_name}")
        ],
        results_dir, f"{model_name}_hourly_points",
        data_points_triplets=[
            (base_sig.t_a_nn, final_res.a_nn_corrected, "A_raw_corrected"),
            (base_sig.t_b_nn, final_res.b_nn_corrected, "B_raw_corrected")
        ],
        legend="upper left", x_label="t", y_label="x(t)")

    visualizer.plot_signals(
        [
            (out_res.t_hourly_out, out_res.signal_std_hourly_out, f"gen_hourly_{model_name}_CI", False)
        ],
        results_dir, f"{model_name}_CI",
        legend="upper left", x_label="t", y_label="sigma(t)")

    if isinstance(out_res.params_out.svgp_t_prior, np.ndarray):
        visualizer.plot_signals_mean_std_precompute(
            [
                (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out,
                 f"gen_hourly_{model_name}")
            ],
            results_dir, f"{model_name}_hourly_prior", legend="upper left", x_label="t", y_label="x(t)",
            f_sample_triplets=[(out_res.params_out.svgp_t_prior, out_res.params_out.svgp_prior_samples, "PRIOR")])

        visualizer.plot_signals_mean_std_precompute(
            [
                (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out,
                 f"gen_hourly_{model_name}")
            ],
            results_dir, f"{model_name}_hourly_posterior", legend="upper left", x_label="t", y_label="x(t)",
            f_sample_triplets=[
                (out_res.params_out.svgp_t_posterior, out_res.params_out.svgp_posterior_samples, "POSTERIOR")])


if __name__ == "__main__":
    ignore_warnings()

    visualizer = Visualizer()
    visualizer.set_figsize()

    ARGS = parse_arguments()
    results_dir_path = create_results_dir(Const.RESULTS_DIR, f"gen_{ARGS.model_type}_{ARGS.output_method}")
    create_logger(results_dir_path)

    mode = Mode.GENERATOR

    data, t_field_name, a_field_name, b_field_name, ground_truth = load_data_run(ARGS, mode)
    model, model_type, correction_method, exposure_method, output_model, output_method, outlier_fraction \
        = setup_run(ARGS, mode, results_dir_path)

    fitter = ModelFitter(data=data,
                         t_field_name=t_field_name,
                         a_field_name=a_field_name,
                         b_field_name=b_field_name,
                         exposure_method=exposure_method,
                         outlier_fraction=outlier_fraction)

    result: Result = fitter(mode=mode,
                            model=model,
                            correction_method=correction_method,
                            output_model=output_model)

    if ARGS.save_signals:
        save_modeling_result(results_dir_path, result, f"gen_{model_type}")

    result.out.params_out.svgp_inducing_points = None

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(ground_truth, result, results_dir_path, f"gen_{model_type}_{output_method}")

    logging.info("Application finished.")
