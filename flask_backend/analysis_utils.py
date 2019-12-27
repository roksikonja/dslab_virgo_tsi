import os
import pickle
from os.path import basename

from flask import current_app
from matplotlib import pyplot as plt

from dslab_virgo_tsi.base import ModelFitter, CorrectionMethod, Mode
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as Gpc
from dslab_virgo_tsi.models import ExpLinModel, ExpModel, SplineModel, IsotonicModel, SmoothMonotonicModel, \
    LocalGPModel, SVGPModel
from dslab_virgo_tsi.run_utils import create_results_dir, save_modeling_result
from dslab_virgo_tsi.status_utils import status
from dslab_virgo_tsi.visualizer import Visualizer
from flask_backend import app
from flask_backend.models import Dataset
from run_modeling import plot_results


def get_models(model_type, output_model_type, num_inducing_points, points_in_window):
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
        output_model = LocalGPModel(points_in_window=points_in_window)
    else:
        output_model = SVGPModel(num_inducing_points=num_inducing_points)

    return model, output_model


def analysis_job(dataset: Dataset, model_type: str, output_model_type: str, model_params: str, correction_method: str):

    try:
        # Enforce optional params
        # TODO: improve this code
        # TODO: this code sets constants that are used within model and passes constants that are passed via init
        # TODO: this way all relevant constants are taken into account
        num_inducing_points = Gpc.NUM_INDUCING_POINTS
        points_in_window = Gpc.POINTS_IN_WINDOW
        model_params_dict = eval(model_params)

        for key in model_params_dict:
            if key == "NUM_INDUCING_POINTS":
                num_inducing_points = model_params_dict[key]
            elif key == "POINTS_IN_WINDOW":
                points_in_window = model_params_dict[key]

            else:
                setattr(Gpc, key, model_params_dict[key])

        # Load pickle
        status.update_progress("Loading dataset", 10)
        pickle_location = dataset.pickle_location
        with open(pickle_location, "rb") as f:
            fitter: ModelFitter = pickle.load(f)

        # Get models
        model, output_model = get_models(model_type, output_model_type, num_inducing_points, points_in_window)

        # Run Fitter
        result = fitter(model, output_model, CorrectionMethod(correction_method), Mode.VIRGO)

        # Create result folder
        with app.app_context():
            results_dir_path = create_results_dir(os.path.join(current_app.root_path, "static", "results"),
                                                  f"{model_type}_{output_model_type}")

        # Store signals
        save_modeling_result(results_dir_path, result, model_type)
        result.out.params_out.svgp_inducing_points = None

        # Plot results
        status.update_progress("Plotting results", 90)
        visualizer = Visualizer()
        visualizer.set_figsize()
        plot_results(visualizer, result, results_dir_path, f"{model_type}_{output_model_type}")

        # Store folder location to status
        status.set_folder(basename(results_dir_path))

        # Close all figures from matplotlib
        plt.close("all")

        # Finish job
        status.set_percentage(100)

    except Exception as e:
        print(e)

    finally:
        status.release()
