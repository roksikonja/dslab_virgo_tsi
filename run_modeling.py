import argparse
import datetime
import os

import matplotlib.pyplot as plt
from matplotlib import style

import constants
from data_utils import load_data, make_dir
from dslab_virgo_tsi.models import ExposureMode, ModelType, ModelFitter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="exp_lin", help="Model to train.")
    parser.add_argument("--visualize", action="store_true", help="Flag for visualizing results.")
    parser.add_argument("--window", type=int, default=81, help="Moving average window size.")

    return parser.parse_args()


def create_results_dir():
    results_dir = os.path.join(constants.RESULTS_DIR_PATH, datetime.datetime.now().strftime("modeling_%Y-%m-%d"))
    make_dir(results_dir)
    return results_dir


def model_type_from_arg(arg_model_type):
    conversion = {"exp_lin": ModelType.EXP_LIN, "exp": ModelType.EXP}
    try:
        return conversion[arg_model_type]
    except KeyError:
        raise NotImplementedError("Provided model_type parameter is not valid.")


def plot_results(results: ModelFitter, data_name, results_dir_path_):
    style.use(constants.MATPLOTLIB_STYLE)

    plt.figure(1, figsize=constants.FIG_SIZE)
    plt.plot(results.t_not_nan, results.ratio_a_b, results.t_not_nan, results.ratio_a_b_initial)
    plt.title(data_name + "-a to " + data_name + "-b ratio - raw, initial fit")
    plt.savefig(os.path.join(results_dir_path_, ARGS.model_type + "_ratio_a_b_raw_initial.pdf"),
                bbox_inches="tight", quality=100, dpi=200)

    plt.figure(2, figsize=constants.FIG_SIZE)
    plt.plot(results.t_not_nan, results.signal_b_not_nan, results.t_not_nan, results.signal_a_not_nan_corrected,
             results.t_not_nan, results.signal_b_not_nan_corrected)
    plt.legend(["b", "a_c", "b_c"])
    plt.title(data_name + "-a to " + data_name + "-b ratio - raw, degradation corrected")
    plt.savefig(os.path.join(results_dir_path_, ARGS.model_type + "_a_b_c.pdf"),
                bbox_inches="tight", quality=100, dpi=200)

    plt.figure(3, figsize=constants.FIG_SIZE)
    plt.plot(results.t_not_nan, results.ratio_a_b, results.t_not_nan, results.ratio_a_b_corrected, results.t_not_nan,
             results.degradation_a, results.t_not_nan, results.degradation_b)
    plt.title(data_name + "-a to " + data_name + "-b ratio - raw, degradation corrected")
    plt.legend(["ratio_a_b_raw", "ratio_a_b_c", "deg_a_opt", "deg_b_opt"])
    plt.savefig(os.path.join(results_dir_path_, ARGS.model_type + "_ratio_a_b_raw_opt.pdf"),
                bbox_inches="tight", quality=100, dpi=200)

    plt.figure(4, figsize=constants.FIG_SIZE)
    plt.scatter(results.t_a_downsample, results.signal_a_downsample, marker="x", c="b")
    plt.scatter(results.t_b_downsample, results.signal_b_downsample, marker="x", c="r")
    plt.plot(results.t_a_downsample, results.signal_a_downsample_corrected)
    plt.plot(results.t_b_downsample, results.signal_b_downsample_corrected)
    plt.title(data_name + "-a to " + data_name + "-b ratio - raw, degradation corrected")
    plt.legend(["a", "b", "a_c", "b_c"], loc="lower left")
    plt.savefig(os.path.join(results_dir_path_, ARGS.model_type + "_a_b_c_full.pdf"),
                bbox_inches="tight", quality=100, dpi=200)

    plt.figure(5, figsize=constants.FIG_SIZE)
    plt.plot(results.t_a_downsample, results.signal_a_downsample_moving_average, color="tab:blue")
    plt.fill_between(results.t_a_downsample,
                     results.signal_a_downsample_moving_average - 1.96 * results.signal_a_downsample_std,
                     results.signal_a_downsample_moving_average + 1.96 * results.signal_a_downsample_std,
                     facecolor='tab:blue', alpha=0.5, label='95% confidence interval')

    plt.plot(results.t_b_downsample, results.signal_b_downsample_moving_average, color="tab:orange")
    plt.fill_between(results.t_b_downsample,
                     results.signal_b_downsample_moving_average - 1.96 * results.signal_b_downsample_std,
                     results.signal_b_downsample_moving_average + 1.96 * results.signal_b_downsample_std,
                     facecolor='tab:orange', alpha=0.5, label='95% confidence interval')

    plt.plot(results.t_a_downsample, results.signal_a_corrected_moving_average, color="tab:green")
    plt.fill_between(results.t_a_downsample,
                     results.signal_a_corrected_moving_average - 1.96 * results.signal_a_corrected_std,
                     results.signal_a_corrected_moving_average + 1.96 * results.signal_a_corrected_std,
                     facecolor='tab:green', alpha=0.5, label='95% confidence interval')

    plt.plot(results.t_b_downsample, results.signal_b_corrected_moving_average, color="tab:red")
    plt.fill_between(results.t_b_downsample,
                     results.signal_b_corrected_moving_average - 1.96 * results.signal_b_corrected_std,
                     results.signal_b_corrected_moving_average + 1.96 * results.signal_b_corrected_std,
                     facecolor='tab:red', alpha=0.5, label='95% confidence interval')

    plt.title(data_name + "-a to " + data_name + "-b ratio - raw, degradation corrected, moving average")
    plt.legend(["a_ma", "b_ma", "a_c_ma", "b_c_ma"], loc="lower left")
    plt.savefig(os.path.join(results_dir_path_, ARGS.model_type + "_a_b_c_full_ma.pdf"),
                bbox_inches="tight", quality=100, dpi=200)


if __name__ == "__main__":
    ARGS = parse_arguments()
    results_dir_path = create_results_dir()

    # Load data
    data_pmo6v = load_data(os.path.join(constants.DATA_DIR_PATH, constants.VIRGO_FILE_PATH))

    # Perform modeling
    model = ModelFitter(data=data_pmo6v,
                        timestamp_field_name="timestamp",
                        signal_a_field_name="pmo6v_a",
                        signal_b_field_name="pmo6v_b",
                        temperature_field_name="temperature",
                        exposure_mode=ExposureMode.EXPOSURE_SUM,
                        model_type=model_type_from_arg(ARGS.model_type))

    plot_results(model, "pmo6v", results_dir_path)

    if ARGS.visualize:
        plt.show()
