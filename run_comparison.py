import logging

from dslab_virgo_tsi.base import Result, ModelFitter, Mode
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.run_utils import setup_run, create_results_dir, create_logger, save_modeling_result, \
    parse_arguments, load_data_run, ignore_warnings
from dslab_virgo_tsi.visualizer import Visualizer


def plot_results(ground_truth_, results_, results_dir, model_name):
    t_, x_ = None, None
    if ground_truth_:
        t_, x_ = ground_truth_

    signal_fourplets = []
    signal_fourplets_a = []
    signal_fourplets_b = []
    for result_ in results_:
        model_type_ = result_[0]
        result_ = result_[1]

        fourplet_a = (result_.base_signals.t_a_nn, result_.final.degradation_a_nn,
                      f"DEGRADATION_{Const.A}_{model_type_}", False)
        fourplet_b = (result_.base_signals.t_b_nn, result_.final.degradation_b_nn,
                      f"DEGRADATION_{Const.B}_{model_type_}", False)

        signal_fourplets.extend([fourplet_a, fourplet_b])
        signal_fourplets_a.append(fourplet_a)
        signal_fourplets_b.append(fourplet_b)

    logging.info("Plotting results ...")
    visualizer.plot_signals(signal_fourplets, results_dir, f"DEGRADATION_{Const.A}_{Const.B}_{model_name}",
                            x_ticker=Const.XTICKER, legend="upper right", x_label=Const.YEAR_UNIT,
                            y_label=Const.TSI_UNIT)

    visualizer.plot_signals(signal_fourplets_a, results_dir, f"DEGRADATION_{Const.A}_{model_name}",
                            x_ticker=Const.XTICKER, legend="upper right", x_label=Const.YEAR_UNIT,
                            y_label=Const.TSI_UNIT)

    visualizer.plot_signals(signal_fourplets_b, results_dir, f"DEGRADATION_{Const.B}_{model_name}",
                            x_ticker=Const.XTICKER, legend="upper right", x_label=Const.YEAR_UNIT,
                            y_label=Const.TSI_UNIT)

    signal_fourplets_a = []
    signal_fourplets_b = []
    for result_ in results_:
        model_type_ = result_[0]
        result_ = result_[1]
        signal_fourplets_a.append(
            (result_.base_signals.t_mutual_nn, result_.history_mutual_nn[-1].a_mutual_nn_corrected,
             f"{Const.A}_mutual_nn_corrected_{model_type_}", False))

        signal_fourplets_b.append(
            (result_.base_signals.t_mutual_nn, result_.history_mutual_nn[-1].b_mutual_nn_corrected,
             f"{Const.B}_mutual_nn_corrected_{model_type_}", False))

    if ground_truth_:
        signal_fourplets_a.append((t_, x_, "ground_truth", False))
        signal_fourplets_b.append((t_, x_, "ground_truth", False), )

    visualizer.plot_signals(
        signal_fourplets_a, results_dir, f"{model_name}_{Const.A}_mutual_corrected",
        x_ticker=Const.XTICKER, legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)

    visualizer.plot_signals(
        signal_fourplets_b, results_dir, f"{model_name}_{Const.B}_mutual_corrected",
        x_ticker=Const.XTICKER, legend="upper right", x_label=Const.YEAR_UNIT, y_label=Const.TSI_UNIT)


if __name__ == "__main__":
    ignore_warnings()

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

    data, t_field_name, a_field_name, b_field_name, ground_truth = load_data_run(ARGS, mode)

    results = []
    for model_type in ["exp", "exp_lin", "spline", "isotonic", "smooth_monotonic", "ensemble"]:
        ARGS.model_type = model_type
        model, _, correction_method, exposure_method, output_model, output_method, outlier_fraction \
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

        results.append((model_type, result))

    if ARGS.save_signals:
        save_modeling_result(results_dir_path, results, run_type)

    if ARGS.save_plots or not ARGS.save_signals:
        plot_results(ground_truth, results, results_dir_path, run_type)

    logging.info("Application finished.")
