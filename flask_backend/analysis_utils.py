import os
import pickle

from flask import current_app

from dslab_virgo_tsi.base import ModelFitter, CorrectionMethod, Mode
from dslab_virgo_tsi.model_constants import GaussianProcessConstants
from dslab_virgo_tsi.models import ExpLinModel, ExpModel, SplineModel, IsotonicModel, SmoothMonotonicModel, \
    LocalGPModel, SVGPModel
from dslab_virgo_tsi.run_utils import create_results_dir, save_modeling_result
from dslab_virgo_tsi.status_utils import status
from dslab_virgo_tsi.visualizer import Visualizer
from flask_backend import app
from flask_backend.models import Dataset
from run_modeling import plot_results


def get_models(model_type, output_model_type):
    if model_type == "EXP_LIN":
        model = ExpLinModel()
    elif model_type == "EXP":
        model = ExpModel()
    elif model_type == "SPLINE":
        model = SplineModel()
    elif model_type == "ISOTONIC":
        model = IsotonicModel()
    else:
        model = SmoothMonotonicModel()

    if output_model_type == "LOCALGP":
        output_model = LocalGPModel()
    else:
        output_model = SVGPModel()

    return model, output_model


def analysis_job(dataset: Dataset, model_type: str, output_model_type: str, model_params: str, correction_method: str):

    # try:
    # Load pickle
    status.update_progress("Loading dataset", 10)
    pickle_location = dataset.pickle_location
    with open(pickle_location, "rb") as f:
        fitter: ModelFitter = pickle.load(f)

    # Get models
    model, output_model = get_models(model_type, output_model_type)

    # Enforce optional params
    model_params_dict = eval(model_params)
    for key in model_params_dict:
        setattr(GaussianProcessConstants, key, model_params_dict[key])

    # Run Fitter
    result = fitter(model, output_model, CorrectionMethod(correction_method), Mode.VIRGO)

    # Create result folder
    with app.app_context():
        results_dir_path = create_results_dir(os.path.join(current_app.root_path, "static", "results"), model_type)

    # Store signals
    save_modeling_result(results_dir_path, result, model_type)
    result.out.params_out.svgp_inducing_points = None

    # Plot results
    status.update_progress("Plotting results", 90)
    visualizer = Visualizer()
    visualizer.set_figsize()
    plot_results(visualizer, result, results_dir_path, f"{model_type}_{output_model_type}")

    # except Exception as e:
    #     print(e)
    #
    # finally:
    #     status.update_progress("Done", 100)
    #     status.release()
