import logging
import collections
import pickle
import os
import datetime

from dslab_virgo_tsi.data_utils import make_dir
from dslab_virgo_tsi.base import ExposureMode, CorrectionMethod, OutputMethod
from dslab_virgo_tsi.model_constants import EnsembleConstants as EnsConsts
from dslab_virgo_tsi.model_constants import ExpConstants as ExpConsts
from dslab_virgo_tsi.model_constants import ExpLinConstants as ExpLinConsts
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts
from dslab_virgo_tsi.model_constants import IsotonicConstants as IsoConsts
from dslab_virgo_tsi.model_constants import SmoothMonotoneRegressionConstants as SMRConsts
from dslab_virgo_tsi.model_constants import SplineConstants as SplConsts
from dslab_virgo_tsi.models import ExpModel, ExpLinModel, SplineModel, IsotonicModel, EnsembleModel, \
    SmoothMonotoneRegression


def setup_run(args, mode, results_dir_path):
    model_type = args.model_type

    # Perform modeling
    if model_type == "exp_lin":
        model = ExpLinModel()
        config = ExpLinConsts.return_config(ExpLinConsts)
    elif model_type == "exp":
        model = ExpModel()
        config = ExpConsts.return_config(ExpConsts)
    elif model_type == "spline":
        model = SplineModel()
        config = SplConsts.return_config(SplConsts)
    elif model_type == "isotonic":
        model = IsotonicModel()
        config = IsoConsts.return_config(IsoConsts)
    elif model_type == "ensemble":
        models = [ExpLinModel(), ExpModel(), SplineModel(), IsotonicModel()]
        model = EnsembleModel(models=models)
        config = EnsConsts.return_config(EnsConsts)
        config["models"] = str(models)
    else:
        model = SmoothMonotoneRegression()
        config = SMRConsts.return_config(SMRConsts)

    # Correction method
    if args.correction_method == "both":
        correction_method = CorrectionMethod.CORRECT_BOTH
    else:
        correction_method = CorrectionMethod.CORRECT_ONE

    # Exposure mode
    if args.exposure_mode == "measurements":
        exposure_mode = ExposureMode.NUM_MEASUREMENTS
    else:
        exposure_mode = ExposureMode.EXPOSURE_SUM

    # Output method
    if args.output_method == "svgp":
        output_method = OutputMethod.SVGP
    elif args.output_method == "kf":
        output_method = OutputMethod.KF
    else:
        output_method = OutputMethod.GP

    # Compute output config
    add_output_config(config, GPConsts.return_config(GPConsts, "OUTPUT"))

    config["model_type"] = model_type
    config["mode"] = mode
    config["correction_method"] = correction_method
    config["outlier_fraction"] = args.outlier_fraction
    config["exposure_mode"] = exposure_mode
    config["output_method"] = output_method

    logging.info("Running in {} mode.".format(mode))
    logging.info(f"Model {model_type} selected.")
    logging.info(f"Correction method {correction_method} selected.")
    logging.info(f"Exposure mode {exposure_mode} selected.")
    logging.info(f"Output method {output_method} selected.")
    logging.info(f"Outlier fraction {args.outlier_fraction} selected.")

    save_config(results_dir_path, config)

    return model, mode, model_type, correction_method, exposure_mode, output_method, args.outlier_fraction


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
    with open(os.path.join(results_dir, name), "w") as f:
        for key in config_out:
            const_type = str(key).split("_")[0]
            if const_type != prev_const_type:
                prev_const_type = const_type
                f.write("\n")

            f.write("{:<50}{}\n".format(key + ":", config_out[key]))
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