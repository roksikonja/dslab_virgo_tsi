import logging

import numpy as np

from dslab_virgo_tsi.base import Result, ModelFitter, FitResult, BaseSignals, FinalResult, Mode
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.model_constants import GeneratorConstants as GenConsts
from dslab_virgo_tsi.run_utils import setup_run, create_results_dir, create_logger, parse_arguments, load_data_run, \
    ignore_warnings
from dslab_virgo_tsi.visualizer import Visualizer


def plot_results(ground_truths_, results_, results_dir, model_name_base):
    for idx in range(len(results_)):
        model_name = model_name_base + "_" + str(idx)

        ground_truth_, (result_, std_a, std_b) = ground_truths_[idx], results_[idx]

        before_fit: FitResult = result_.history_mutual_nn[0]
        last_iter: FitResult = result_.history_mutual_nn[-1]

        base_sig: BaseSignals = result_.base_signals
        final_res: FinalResult = result_.final

        t_, x_ = ground_truth_

        visualizer.plot_signals(
            [
                (base_sig.t_a_nn, final_res.degradation_a_nn, "DEGRADATION_A", False),
                (base_sig.t_b_nn, final_res.degradation_b_nn, "DEGRADATION_B", False)
            ],
            results_dir, f"{model_name}_DEGRADATION", legend="upper right",
            x_label="t", y_label="d(t)")

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

    degradations_a, degradations_b = [], []
    for idx in range(len(results_)):
        (result_, std_a, std_b) = results_[idx]

        base_sig: BaseSignals = result_.base_signals
        final_res: FinalResult = result_.final

        degradations_a.append((base_sig.t_a_nn, final_res.degradation_a_nn, f"DEGRADATION_A_{idx}", False))
        degradations_b.append((base_sig.t_b_nn, final_res.degradation_b_nn, f"DEGRADATION_B_{idx}", False))

    visualizer.plot_signals(degradations_a,
                            results_dir, f"{model_name_base}_DEGRADATIONS_A", legend="upper right",
                            x_label="t", y_label="d(t)")

    visualizer.plot_signals(degradations_b,
                            results_dir, f"{model_name_base}_DEGRADATIONS_B", legend="upper right",
                            x_label="t", y_label="d(t)")


if __name__ == "__main__":
    ignore_warnings()

    visualizer = Visualizer()
    visualizer.set_figsize()

    ARGS = parse_arguments()
    results_dir_path = create_results_dir(Const.RESULTS_DIR, f"gen_{ARGS.model_type}_{ARGS.output_method}")
    create_logger(results_dir_path)

    mode = Mode.GENERATOR
    model_type = None

    ground_truths, results = [], []
    for std_noise in np.arange(0.00, 0.05, 0.01):
        std_noise_a, std_noise_b = std_noise * 2.5, std_noise * 1.5

        data, t_field_name, a_field_name, b_field_name, ground_truth = load_data_run(ARGS, mode,
                                                                                     (std_noise_a, std_noise_b))
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
                                output_model=output_model,
                                compute_output=False)

        ground_truths.append(ground_truth)
        results.append((result, std_noise_a, std_noise_b))

    plot_results(ground_truths, results, results_dir_path, f"gen_{model_type}")

    logging.info("Application finished.")
