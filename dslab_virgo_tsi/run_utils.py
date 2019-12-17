import argparse
import collections
import datetime
import logging
import os
import pickle
import warnings

import pandas as pd

from dslab_virgo_tsi.base import ExposureMethod, CorrectionMethod, Mode
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import make_dir, load_data
from dslab_virgo_tsi.generator import SignalGenerator
from dslab_virgo_tsi.model_constants import EnsembleConstants as EnsConsts
from dslab_virgo_tsi.model_constants import ExpConstants as ExpConsts
from dslab_virgo_tsi.model_constants import ExpLinConstants as ExpLinConsts
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts
from dslab_virgo_tsi.model_constants import GeneratorConstants as GenConsts
from dslab_virgo_tsi.model_constants import IsotonicConstants as IsoConsts
from dslab_virgo_tsi.model_constants import OutputTimeConstants as OutTimeConsts
from dslab_virgo_tsi.model_constants import SmoothMonotoneRegressionConstants as SMRConsts
from dslab_virgo_tsi.model_constants import SplineConstants as SplConsts
from dslab_virgo_tsi.models import ExpModel, ExpLinModel, SplineModel, IsotonicModel, EnsembleModel, \
    SmoothMonotonicModel, SVGPModel, GPModel, LocalGPModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="virgo", help="Set data mode.")

    parser.add_argument("--random_seed", type=int, default=5, help="Set generator random seed.")

    parser.add_argument("--virgo_days_start", type=int, default=-1, help="Use data from this day on. -1 for all.")
    parser.add_argument("--virgo_days_end", type=int, default=-1, help="Use data up to this day. -1 for all.")

    parser.add_argument("--save_plots", action="store_true", help="Flag for saving plots.")
    parser.add_argument("--save_signals", action="store_true", help="Flag for saving computed signals.")

    parser.add_argument("--model_type", type=str, default="smooth_monotonic", help="Model to train.")
    parser.add_argument("--correction_method", type=str, default="one", help="Iterative correction method.")
    parser.add_argument("--exposure_method", type=str, default="measurements", help="Exposure computing method.")
    parser.add_argument("--output_method", type=str, default="svgp", help="Exposure computing method.")
    parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")

    return parser.parse_args()


def setup_run(args, mode: Mode, results_dir_path):
    model_type = args.model_type

    # Perform modeling
    if model_type == "exp_lin":
        model_type = Const.EXP_LIN
        model = ExpLinModel()
        config = ExpLinConsts.return_config(ExpLinConsts)
    elif model_type == "exp":
        model_type = Const.EXP
        model = ExpModel()
        config = ExpConsts.return_config(ExpConsts)
    elif model_type == "spline":
        model_type = Const.SPLINE
        model = SplineModel()
        config = SplConsts.return_config(SplConsts)
    elif model_type == "isotonic":
        model_type = Const.ISOTONIC
        model = IsotonicModel()
        config = IsoConsts.return_config(IsoConsts)
    elif model_type == "ensemble":
        model_type = Const.ENSEMBLE
        models = [ExpLinModel(), ExpModel(), SplineModel(), IsotonicModel()]
        model = EnsembleModel(models=models)
        config = EnsConsts.return_config(EnsConsts)
        config["models"] = str(models)
    else:
        model_type = Const.SMOOTH_MONOTONIC
        model = SmoothMonotonicModel()
        config = SMRConsts.return_config(SMRConsts)

    # Correction method
    if args.correction_method == "both":
        correction_method = CorrectionMethod.CORRECT_BOTH
    else:
        correction_method = CorrectionMethod.CORRECT_ONE

    # Exposure method
    if args.exposure_method == "measurements":
        exposure_method = ExposureMethod.NUM_MEASUREMENTS
    else:
        exposure_method = ExposureMethod.EXPOSURE_SUM

    # Output method
    if args.output_method == "gp":
        output_method = Const.GP
        output_model = GPModel()
    elif args.output_method == "localgp":
        output_method = Const.LOCALGP
        output_model = LocalGPModel()
    else:
        output_method = Const.SVGP
        output_model = SVGPModel()

    # Compute output config
    add_output_config(config, GPConsts.return_config(GPConsts, "OUTPUT_GP"))
    add_output_config(config, OutTimeConsts.return_config(OutTimeConsts, "OUTPUT_TIME"))

    config["model_type"] = model_type
    config["mode"] = mode
    config["correction_method"] = correction_method
    config["outlier_fraction"] = args.outlier_fraction
    config["exposure_method"] = exposure_method
    config["output_method"] = output_method

    logging.info("Running in {} mode.".format(mode))
    logging.info(f"Model {model_type} selected.")
    logging.info(f"Correction method {correction_method} selected.")
    logging.info(f"Exposure method {exposure_method} selected.")
    logging.info(f"Output method {output_method} selected.")
    logging.info(f"Outlier fraction {args.outlier_fraction} selected.")

    if mode == Mode.VIRGO:
        config["virgo_days_start"] = args.virgo_days_start
        config["virgo_days_end"] = args.virgo_days_end
        if args.virgo_days_start > 0:
            logging.info(f"Data from VIRGO day {args.virgo_days_start} on used.")

        if args.virgo_days_end > 0:
            logging.info(f"Data up to VIRGO day {args.virgo_days_end}.")

    save_config(results_dir_path, config)

    return model, model_type, correction_method, exposure_method, output_model, output_method, args.outlier_fraction


def load_data_run(args, mode: Mode):
    if mode == Mode.GENERATOR:
        generator = SignalGenerator(length=GenConsts.SIGNAL_LENGTH,
                                    random_seed=args.random_seed,
                                    exposure_method=args.exposure_method)
        t = generator.time
        x = generator.x

        x_a_raw, x_b_raw, _ = generator.generate_raw_signal(x_=x,
                                                            random_seed=5,
                                                            rate=GenConsts.DEGRADATION_RATE,
                                                            degradation_model=GenConsts.DEGRADATION_MODEL)

        t_field_name, a_field_name, b_field_name = "t", "x_a", "x_b"
        data = pd.DataFrame()
        data[t_field_name] = t
        data[a_field_name] = x_a_raw
        data[b_field_name] = x_b_raw

        ground_truth = (t, x)
        logging.info(f"Dataset Generator of size {data.shape} loaded.")
    else:
        data = load_data(Const.DATA_DIR, Const.VIRGO_FILE)

        if args.virgo_days_end > 0:
            data = data[data[Const.T] <= args.virgo_days_end]

        if args.virgo_days_start > 0:
            data = data[data[Const.T] >= args.virgo_days_start]

        t_field_name, a_field_name, b_field_name = Const.T, Const.A, Const.B
        ground_truth = None
        logging.info(f"Dataset {Const.VIRGO_FILE} of size {data.shape} loaded.")

    return data, t_field_name, a_field_name, b_field_name, ground_truth


def create_results_dir(results_dir_path, model_type):
    results_dir = make_dir(os.path.join(results_dir_path,
                                        datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_{model_type}")))
    return results_dir


def save_modeling_result(results_dir, model_results, model_name):
    name = f"{model_name}_modeling_result.pkl"
    with open(os.path.join(results_dir, name), 'wb') as f:
        pickle.dump(model_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Modeling result saved to {name}.")


def save_config(results_dir, config):
    config_out = dict()

    for key in config:
        if str(key).islower():
            config_out[f"BASE_{key}"] = config[key]
        else:
            config_out[key] = config[key]

    config_out = collections.OrderedDict(sorted(config_out.items()))

    name = "config.txt"
    prev_const_type = "BASE"
    with open(os.path.join(results_dir, name), "a+") as f:
        for key in config_out:
            const_type = str(key).split("_")[0]
            if const_type != prev_const_type:
                prev_const_type = const_type
                f.write("\n")

            f.write("{:<50}{}\n".format(key + ":", config_out[key]))
        f.write("\n\n")
    logging.info(f"Config saved to {name}.")


def add_output_config(config, output_config):
    config.update(output_config)


def create_logger(results_dir):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)-5s %(name)-20s %(levelname)-15s %(message)s',
                        datefmt='[%m-%d %H:%M]',
                        handlers=[logging.FileHandler(os.path.join(results_dir, 'log.log')),
                                  logging.StreamHandler()])
    logging.info("Application started.")
    logging.info("Logging started.")


def ignore_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", category=RuntimeWarning)
