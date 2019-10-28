import argparse

import pandas as pd

from dslab_virgo_tsi.base import ExposureMode, Result, ModelFitter, CorrectionMethod
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.generator import SignalGenerator, create_results_dir, plot_results
from dslab_virgo_tsi.models import ExpModel, ExpLinModel, SplineModel, IsotonicModel, EnsembleModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_length", type=int, default=100000, help="Generated signal length.")
    parser.add_argument("--degradation_model", type=str, default="exp", help="Model to train.")
    parser.add_argument("--degradation_rate", type=float, default=1.0, help="Tuning parameter for degradation.")
    parser.add_argument("--random_seed", type=int, default=0, help="Set random seed.")

    parser.add_argument("--model_type", type=str, default="isotonic", help="Model to train.")
    parser.add_argument("--model_smoothing", action="store_true", help="Only for isotonic model.")

    parser.add_argument("--correction_method", type=int, default=2, help="Iterative correction method.")
    parser.add_argument("--window", type=int, default=81, help="Moving average window size.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_arguments()

    # Constants
    data_dir = Const.DATA_DIR
    results_dir_path = create_results_dir(f"gen_{ARGS.model_type}")

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

    model = None
    if ARGS.model_type == "exp_lin":
        model = ExpLinModel()
    elif ARGS.model_type == "exp":
        model = ExpModel()
    elif ARGS.model_type == "spline":
        model = SplineModel()
    elif ARGS.model_type == "isotonic":
        model = IsotonicModel(smoothing=ARGS.model_smoothing)
    elif ARGS.model_type == "ensemble":
        model1, model2, model3, model4 = ExpLinModel(), ExpModel(), SplineModel(), IsotonicModel()
        model = EnsembleModel([model1, model2, model3, model4], [0.1, 0.3, 0.3, 0.3])

    # Get correction method
    if ARGS.correction_method == 1:
        correction_method = CorrectionMethod.CORRECT_BOTH
    else:
        correction_method = CorrectionMethod.CORRECT_ONE

    fitter = ModelFitter(data=data_gen,
                         t_field_name=T,
                         a_field_name=X_A,
                         b_field_name=X_B,
                         exposure_mode=ExposureMode.NUM_MEASUREMENTS,
                         outlier_fraction=ARGS.outlier_fraction)

    result: Result = fitter(model=model,
                            correction_method=correction_method,
                            moving_average_window=ARGS.window)

    plot_results(t, x, result, results_dir_path, f"gen_{ARGS.model_type}")
