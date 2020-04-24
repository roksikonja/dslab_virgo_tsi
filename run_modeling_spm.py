import logging

import numpy as np

from dslab_virgo_tsi.base import Result, FitResult, ModelFitter, BaseSignals, OutResult, FinalResult, Mode
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.run_utils import setup_run, create_results_dir, create_logger, save_modeling_result, \
    parse_arguments, load_data_run, ignore_warnings
from dslab_virgo_tsi.visualizer import Visualizer


def plot_results(visualizer_: Visualizer, result_: Result, results_dir, model_name):
    before_fit: FitResult = result_.history_mutual_nn[0]
    last_iter: FitResult = result_.history_mutual_nn[-1]

    base_sig: BaseSignals = result_.base_signals
    out_res: OutResult = result_.out
    final_res: FinalResult = result_.final

    logging.info("Plotting results ...")

    if out_res.params_out.svgp_iter_loglikelihood:
        visualizer_.plot_iter_loglikelihood(out_res.params_out.svgp_iter_loglikelihood, results_dir,
                                            f"{model_name}_ITER_LOGLIKELIHOOD", legend="lower left",
                                            x_label=Const.ITERATION_UNIT, y_label=Const.LOG_LIKELIHOOD_UNIT)

    visualizer_.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"TSI_hourly_{model_name}")
        ],
        results_dir, f"{model_name}_TSI_hourly", legend="upper left",
        x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT, max_points=1e7,
        inducing_points=out_res.params_out.svgp_inducing_points)

    if isinstance(out_res.params_out.svgp_t_prior, np.ndarray):
        visualizer_.plot_signals_mean_std_precompute(
            [
                (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out,
                 f"TSI_hourly_{model_name}")
            ],
            results_dir, f"{model_name}_TSI_hourly_prior", legend="upper left",
            x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT, max_points=1e7,
            f_sample_triplets=[(out_res.params_out.svgp_t_prior, out_res.params_out.svgp_prior_samples, "PRIOR")])

        visualizer_.plot_signals_mean_std_precompute(
            [
                (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out,
                 f"TSI_hourly_{model_name}")
            ],
            results_dir, f"{model_name}_TSI_hourly_posterior", legend="upper left",
            x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT, max_points=1e7,
            f_sample_triplets=[
                (out_res.params_out.svgp_t_posterior, out_res.params_out.svgp_posterior_samples, "POSTERIOR")])

    visualizer_.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"TSI_hourly_{model_name}")
        ],
        results_dir, f"{model_name}_TSI_hourly_points", legend="upper left",
        data_points_triplets=[
            (base_sig.t_a_nn, final_res.a_nn_corrected, f"{Const.A}_corrected"),
            (base_sig.t_b_nn, final_res.b_nn_corrected, f"{Const.B}_corrected")
        ],
        x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT, max_points=1e7)

    visualizer_.plot_signals(
        [
            (base_sig.t_a_nn, final_res.degradation_a_nn, f"DEGRADATION_{Const.A}", False),
            (base_sig.t_b_nn, final_res.degradation_b_nn, f"DEGRADATION_{Const.B}", False)
        ],
        results_dir, f"{model_name}_DEGRADATION_{Const.A}_{Const.B}", legend="upper right",
        x_label=Const.DAY_UNIT, y_label=Const.DEGRADATION_UNIT)

    visualizer_.plot_signals(
        [
            (out_res.t_hourly_out, out_res.signal_std_hourly_out, f"CI_95%", False)
        ],
        results_dir, f"{model_name}_CI",
        legend="upper left", x_label="t", y_label="sigma(t)")

    visualizer_.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.a_mutual_nn_corrected, f"{Const.A}_mutual_nn", False),
            (base_sig.t_mutual_nn, before_fit.b_mutual_nn_corrected, f"{Const.B}_mutual_nn", False),
            (base_sig.t_mutual_nn, last_iter.a_mutual_nn_corrected, f"{Const.A}_mutual_nn_corrected", False),
            (base_sig.t_mutual_nn, last_iter.b_mutual_nn_corrected, f"{Const.B}_mutual_nn_corrected", False)
        ],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_mutual_raw_corrected", legend="upper right",
        x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT)

    visualizer_.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.a_mutual_nn_corrected, f"{Const.A}_mutual_nn", False),
            (base_sig.t_mutual_nn, before_fit.b_mutual_nn_corrected, f"{Const.B}_mutual_nn", False),
        ],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_mutual_raw", legend="upper right",
        x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT)

    visualizer_.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.ratio_a_b_mutual_nn_corrected, f"RATIO_{Const.A}_{Const.B}_raw", False),
            (base_sig.t_mutual_nn, last_iter.ratio_a_b_mutual_nn_corrected, f"RATIO_{Const.A}_not_{Const.B}_corrected",
             False),
            (base_sig.t_mutual_nn, np.divide(last_iter.a_mutual_nn_corrected, last_iter.b_mutual_nn_corrected),
             f"RATIO_{Const.A}_corrected_{Const.B}_corrected", False)
        ],
        results_dir, f"{model_name}_RATIO_DEGRADATION_{Const.A}_{Const.B}_raw_corrected",
        legend="upper right", x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT)

    visualizer_.plot_signals(
        [
            (base_sig.t_a_nn, base_sig.a_nn, f"{Const.A}_raw", False),
            (base_sig.t_b_nn, base_sig.b_nn, f"{Const.B}_raw", False),
            (base_sig.t_a_nn, final_res.a_nn_corrected, f"{Const.A}_corrected", False),
            (base_sig.t_b_nn, final_res.b_nn_corrected, f"{Const.B}_corrected", False),
        ],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_raw_corrected_full",
        legend="upper right", x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT)

    visualizer_.plot_signals(
        [
            (base_sig.t_a_nn, base_sig.a_nn, f"{Const.A}_raw", False),
            (base_sig.t_b_nn, base_sig.b_nn, f"{Const.B}_raw", False),
        ],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_raw_full",
        legend="upper right", x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT)

    visualizer_.plot_signals(
        [
            (base_sig.t_a_nn, final_res.a_nn_corrected, f"{Const.A}_corrected", False),
            (base_sig.t_b_nn, final_res.b_nn_corrected, f"{Const.B}_corrected", False),
        ],
        results_dir, f"{model_name}_{Const.A}_{Const.B}_raw_corrected_full",
        legend="upper right", x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT)

    visualizer_.plot_signal_history(base_sig.t_mutual_nn, result_.history_mutual_nn,
                                    results_dir, f"{model_name}_history",
                                    legend="upper right", x_label=Const.DAY_UNIT, y_label=Const.SPM_UNIT)


if __name__ == "__main__":
    ignore_warnings()

    visualizer = Visualizer()
    visualizer.set_figsize()

    ARGS = parse_arguments()
    results_dir_path = create_results_dir(Const.RESULTS_DIR, f"{ARGS.model_type}_{ARGS.output_method}")
    create_logger(results_dir_path)

    mode = Mode.SPM

    data, t_field_name, a_field_name, b_field_name, _ = load_data_run(ARGS, mode)

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
        save_modeling_result(results_dir_path, result, model_type)

    result.out.params_out.svgp_inducing_points = None

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(visualizer, result, results_dir_path, f"{model_type}_{output_method}")

    logging.info("Application finished.")
