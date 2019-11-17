import logging
import os

from dslab_virgo_tsi.base import Result, ModelFitter, Mode
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.run_utils import setup_run, create_results_dir, create_logger, save_modeling_result, \
    parse_arguments, load_data_run
from dslab_virgo_tsi.visualizer import Visualizer

"""
--random_seed = 0
--save_plots = store_true
--save_signals = store_true

--correction_method = "one"
--outlier_fraction = 0.0
--exposure_method = "measurements"
--output_method = "svgp"
"""


def plot_results(results_, results_dir, model_name):
    signal_fourplets = []
    for result_ in results_:
        model_type_ = result_[0]
        result_ = result_[1]
        signal_fourplets.extend([
            (result_.base_signals.t_a_nn, result_.final.degradation_a_nn,
             f"DEGRADATION_{Const.A}_{model_type_}", False),
            (result_.base_signals.t_b_nn, result_.final.degradation_b_nn,
             f"DEGRADATION_{Const.B}_{model_type_}", False)
        ])

    logging.info("Plotting results ...")
    visualizer.plot_signals(signal_fourplets, results_dir, f"DEGRADATION_{Const.A}_{Const.B}_{model_name}",
                            x_ticker=1, legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    signal_fourplets = []
    for result_ in results_:
        model_type_ = result_[0]
        result_ = result_[1]
        signal_fourplets.extend([
            (result_.base_signals.t_mutual_nn, result_.history_mutual_nn[-1].a_mutual_nn_corrected,
             f"{Const.A}_mutual_nn_corrected_{model_type_}", False),
            (result_.base_signals.t_mutual_nn, result_.history_mutual_nn[-1].b_mutual_nn_corrected,
             f"{Const.B}_mutual_nn_corrected_{model_type_}", False)
        ])

    visualizer.plot_signals(
        signal_fourplets, results_dir, f"{model_name}_{Const.A}_{Const.B}_mutual_corrected",
        x_ticker=1, legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    visualizer = Visualizer()
    visualizer.set_figsize()

    run_type = "comparison"
    ARGS = parse_arguments()
    if ARGS.mode == "gen":
        mode = Mode.GENERATOR
        run_type = "gen_" + run_type
    else:
        mode = Mode.VIRGO

    results_dir_path = create_results_dir(Const.RESULTS_DIR, run_type)
    create_logger(results_dir_path)

    data, t_field_name, a_field_name, b_field_name, _ = load_data_run(ARGS, mode)

    results = []
    for model_type in ["exp", "exp_lin", "spline", "isotonic", "smooth_monotonic"]:
        ARGS.model_type = model_type
        model, _, correction_method, exposure_method, output_method, outlier_fraction \
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
                                output_method=output_method,
                                compute_output=False)

        results.append((model_type, result))

    if ARGS.save_signals:
        save_modeling_result(results_dir_path, results, run_type)

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(results, results_dir_path, run_type)

    logging.info("Application finished.")
