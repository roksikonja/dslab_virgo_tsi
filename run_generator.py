import argparse
import logging

import numpy as np
import pandas as pd

from dslab_virgo_tsi.base import ExposureMode, Result, ModelFitter, CorrectionMethod
from dslab_virgo_tsi.base import FitResult, BaseSignals, OutResult, FinalResult
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import create_results_dir, save_config, create_logger
from dslab_virgo_tsi.generator import SignalGenerator
from dslab_virgo_tsi.model_constants import EnsembleConstants as EnsConsts
from dslab_virgo_tsi.model_constants import ExpConstants as ExpConsts
from dslab_virgo_tsi.model_constants import ExpLinConstants as ExpLinConsts
from dslab_virgo_tsi.model_constants import IsotonicConstants as IsoConsts
from dslab_virgo_tsi.model_constants import SmoothMonotoneRegressionConstants as SMRConsts
from dslab_virgo_tsi.model_constants import SplineConstants as SplConsts
from dslab_virgo_tsi.models import ExpModel, ExpLinModel, SplineModel, IsotonicModel, EnsembleModel, \
    SmoothMonotoneRegression
from dslab_virgo_tsi.visualizer import Visualizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_length", type=int, default=100000, help="Generated signal length.")
    parser.add_argument("--degradation_model", type=str, default="exp", help="Model to train.")
    parser.add_argument("--degradation_rate", type=float, default=1.0, help="Tuning parameter for degradation.")
    parser.add_argument("--random_seed", type=int, default=0, help="Set random seed.")

    parser.add_argument("--window", type=int, default=81, help="Moving average window size for plotting.")

    parser.add_argument("--model_type", type=str, default="smooth_monotonic", help="Model to train.")
    parser.add_argument("--correction_method", type=str, default="one", help="Iterative correction method.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")
    parser.add_argument("--exposure_mode", type=str, default="measurements", help="Exposure computing method.")

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


if __name__ == "__main__":
    ARGS = parse_arguments()

    # Constants
    data_dir = Const.DATA_DIR
    results_dir_path = create_results_dir(Const.RESULTS_DIR, f"gen_{ARGS.model_type}")
    create_logger(results_dir_path)

    visualizer = Visualizer()
    visualizer.set_figsize()

    # Generator
    Generator = SignalGenerator(ARGS.signal_length, ARGS.random_seed)
    t = Generator.time
    x = Generator.x
    x_a_raw, x_b_raw, _ = Generator.generate_raw_signal(x, 5, rate=ARGS.degradation_rate)

    T, X_A, X_B = "t", "x_a", "x_b"
    data_gen = pd.DataFrame()
    data_gen[T] = t
    data_gen[X_A] = x_a_raw
    data_gen[X_B] = x_b_raw
    logging.info(f"Data generator loaded.")

    # Perform modeling
    model = None
    config = None
    if ARGS.model_type == "exp_lin":
        model = ExpLinModel()
        config = ExpLinConsts.return_config()
    elif ARGS.model_type == "exp":
        model = ExpModel()
        config = ExpConsts.return_config()
    elif ARGS.model_type == "spline":
        model = SplineModel()
        config = SplConsts.return_config()
    elif ARGS.model_type == "isotonic":
        model = IsotonicModel()
        config = IsoConsts.return_config()
    elif ARGS.model_type == "ensemble":
        models = [ExpLinModel(), ExpModel(), SplineModel(), IsotonicModel()]
        model = EnsembleModel(models=models)
        config = EnsConsts.return_config()
        config["models"] = str(models)
    elif ARGS.model_type == "smooth_monotonic":
        model = SmoothMonotoneRegression()
        config = SMRConsts.return_config()

    # Get correction method
    if ARGS.correction_method == "both":
        correction_method = CorrectionMethod.CORRECT_BOTH
    else:
        correction_method = CorrectionMethod.CORRECT_ONE

    if ARGS.exposure_mode == "measurements":
        exposure_mode = ExposureMode.NUM_MEASUREMENTS
    else:
        exposure_mode = ExposureMode.EXPOSURE_SUM

    config["correction_method"] = ARGS.correction_method
    config["model_type"] = ARGS.model_type
    config["outlier_fraction"] = ARGS.outlier_fraction
    config["exposure_mode"] = ARGS.exposure_mode
    config["mode"] = "generator"

    logging.info("Running in {} mode.".format(config["mode"]))
    logging.info(f"Model {ARGS.model_type} selected.")
    logging.info(f"Correction method {ARGS.correction_method} selected.")
    logging.info(f"Exposure mode {ARGS.exposure_mode} selected.")
    logging.info(f"Outlier fraction {ARGS.outlier_fraction} selected.")

    save_config(results_dir_path, config)

    fitter = ModelFitter(mode=config["mode"],
                         data=data_gen,
                         t_field_name=T,
                         a_field_name=X_A,
                         b_field_name=X_B,
                         exposure_mode=exposure_mode,
                         outlier_fraction=ARGS.outlier_fraction)

    result: Result = fitter(model=model,
                            correction_method=correction_method)

    plot_results(t, x, result, results_dir_path, f"gen_{ARGS.model_type}")
    logging.info("Application finished.")
