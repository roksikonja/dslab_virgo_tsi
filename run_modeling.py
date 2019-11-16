import argparse
import logging

import numpy as np

from dslab_virgo_tsi.base import ExposureMode, Result, FitResult, ModelFitter, BaseSignals, OutResult, FinalResult, \
    CorrectionMethod
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import load_data, create_results_dir, create_logger, save_modeling_result, \
    save_config, add_output_config
from dslab_virgo_tsi.model_constants import EnsembleConstants as EnsConsts
from dslab_virgo_tsi.model_constants import ExpConstants as ExpConsts
from dslab_virgo_tsi.model_constants import ExpLinConstants as ExpLinConsts
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts
from dslab_virgo_tsi.model_constants import IsotonicConstants as IsoConsts
from dslab_virgo_tsi.model_constants import SmoothMonotoneRegressionConstants as SMRConsts
from dslab_virgo_tsi.model_constants import SplineConstants as SplConsts
from dslab_virgo_tsi.models import ExpModel, ExpLinModel, SplineModel, IsotonicModel, EnsembleModel, \
    SmoothMonotoneRegression
from dslab_virgo_tsi.visualizer import Visualizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_plots", action="store_true", help="Flag for saving plots.")
    parser.add_argument("--save_signals", action="store_true", help="Flag for saving computed signals.")
    parser.add_argument("--window", type=int, default=81, help="Moving average window size for plotting.")

    parser.add_argument("--model_type", type=str, default="smooth_monotonic", help="Model to train.")
    parser.add_argument("--correction_method", type=str, default="one", help="Iterative correction method.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")
    parser.add_argument("--exposure_mode", type=str, default="measurements", help="Exposure computing method.")

    return parser.parse_args()


def plot_results(result_: Result, results_dir, model_name, window_size):
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
            (out_res.t_daily_out, out_res.signal_daily_out, out_res.signal_std_daily_out, f"TSI_daily_{model_name}")
        ],
        results_dir, f"TSI_daily_{model_name}", x_ticker=1, legend="upper left", y_lim=[1362, 1369],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, max_points=1e7)

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_hourly_out, out_res.signal_hourly_out, out_res.signal_std_hourly_out, f"TSI_hourly_{model_name}")
        ],
        results_dir, f"TSI_hourly_{model_name}", x_ticker=1, legend="upper left", y_lim=[1362, 1369],
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, max_points=1e7)

    visualizer.plot_signals_mean_std_precompute(
        [
            (out_res.t_daily_out, out_res.signal_daily_out, out_res.signal_std_daily_out, f"TSI_daily_{model_name}")
        ],
        results_dir, f"TSI_daily_{model_name}_points", x_ticker=1, legend="upper left", y_lim=[1362, 1369],
        data_points_triplets=[
            (base_sig.t_a_nn, final_res.a_nn_corrected, f"{Const.A}_raw_corrected"),
            (base_sig.t_b_nn, final_res.b_nn_corrected, f"{Const.B}_raw_corrected")
        ],
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

    visualizer.plot_signals_mean_std(
        [
            (base_sig.t_a_nn, final_res.a_nn_corrected, f"{Const.A}_corrected_conf_int", window_size)
        ],
        results_dir, f"{model_name}_{Const.A}_raw_corrected_full_conf_int", x_ticker=1, legend="lower left",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, y_lim=[1362, 1369])

    visualizer.plot_signals_mean_std(
        [
            (base_sig.t_b_nn, final_res.b_nn_corrected, f"{Const.B}_corrected_conf_int", window_size)
        ],
        results_dir, f"{model_name}_{Const.B}_raw_corrected_full_conf_int", x_ticker=1, legend="lower left",
        x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT, y_lim=[1362, 1369])

    visualizer.plot_signal_history(base_sig.t_mutual_nn, result_.history_mutual_nn,
                                   results_dir, f"{model_name}_history",
                                   ground_truth_triplet=None,
                                   legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)


if __name__ == "__main__":
    ARGS = parse_arguments()
    results_dir_path = create_results_dir(Const.RESULTS_DIR, ARGS.model_type)
    create_logger(results_dir_path)

    # Load data
    data_pmo6v = load_data(Const.DATA_DIR, Const.VIRGO_FILE)
    logging.info(f"Data {Const.VIRGO_FILE} loaded.")

    visualizer = Visualizer()
    visualizer.set_figsize()

    # Perform modeling
    model = None
    config = None
    if ARGS.model_type == "exp_lin":
        model = ExpLinModel()
        config = ExpLinConsts.return_config(ExpLinConsts)
    elif ARGS.model_type == "exp":
        model = ExpModel()
        config = ExpConsts.return_config(ExpConsts)
    elif ARGS.model_type == "spline":
        model = SplineModel()
        config = SplConsts.return_config(SplConsts)
    elif ARGS.model_type == "isotonic":
        model = IsotonicModel()
        config = IsoConsts.return_config(IsoConsts)
    elif ARGS.model_type == "ensemble":
        models = [ExpLinModel(), ExpModel(), SplineModel(), IsotonicModel()]
        model = EnsembleModel(models=models)
        config = EnsConsts.return_config(EnsConsts)
        config["models"] = str(models)
    elif ARGS.model_type == "smooth_monotonic":
        model = SmoothMonotoneRegression()
        config = SMRConsts.return_config(SMRConsts)

    # Get correction method
    if ARGS.correction_method == "both":
        correction_method = CorrectionMethod.CORRECT_BOTH
    else:
        correction_method = CorrectionMethod.CORRECT_ONE

    if ARGS.exposure_mode == "measurements":
        exposure_mode = ExposureMode.NUM_MEASUREMENTS
    else:
        exposure_mode = ExposureMode.EXPOSURE_SUM

    # Compute output config
    add_output_config(config, GPConsts.return_config(GPConsts, "OUTPUT"))

    config["correction_method"] = ARGS.correction_method
    config["model_type"] = ARGS.model_type
    config["outlier_fraction"] = ARGS.outlier_fraction
    config["exposure_mode"] = ARGS.exposure_mode
    config["mode"] = "virgo"

    logging.info("Running in {} mode.".format(config["mode"]))
    logging.info(f"Model {ARGS.model_type} selected.")
    logging.info(f"Correction method {ARGS.correction_method} selected.")
    logging.info(f"Exposure mode {ARGS.exposure_mode} selected.")
    logging.info(f"Outlier fraction {ARGS.outlier_fraction} selected.")

    save_config(results_dir_path, config)

    fitter = ModelFitter(mode=config["mode"],
                         data=data_pmo6v,
                         t_field_name=Const.T,
                         a_field_name=Const.A,
                         b_field_name=Const.B,
                         exposure_mode=ExposureMode.NUM_MEASUREMENTS,
                         outlier_fraction=ARGS.outlier_fraction)

    result: Result = fitter(model=model,
                            correction_method=correction_method)

    if ARGS.save_signals:
        save_modeling_result(results_dir_path, result, ARGS.model_type)

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(result, results_dir_path, ARGS.model_type, ARGS.window)

    logging.info("Application finished.")
