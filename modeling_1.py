from data_utils import load_data, make_dir, downsample_signal, detect_outliers, notnan_indices
from constants import Constants as C
from modeling import compute_exposure, initial_fit, em_estimate_exp_lin, em_estimate_exp, DegradationModels
from visualizer import Visualizer

import os
import datetime
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="exp_lin", help="Model to train.")
parser.add_argument("--visualize", action="store_true", help="Flag for visualizing results.")
parser.add_argument("--window", type=int, default=81, help="Moving average window size.")
parser.add_argument("--outlier_fraction", type=float, default=0, help="Outlier fraction.")

ARGS = parser.parse_args()

visualizer = Visualizer()
visualizer.set_figsize()

# Parameters
data_dir = C.DATA_DIR
results_dir = C.RESULTS_DIR
virgo_file = C.VIRGO_FILE


results_dir = os.path.join(results_dir, datetime.datetime.now().strftime("modeling_%Y-%m-%d"))
make_dir(results_dir)

# Load data
data = load_data(os.path.join(data_dir, virgo_file))

t = data[C.T].values
pmo_a = data[C.A].values
pmo_b = data[C.B].values

# Filter outliers
outliers_a = notnan_indices(pmo_a)
outliers_a[outliers_a] = detect_outliers(pmo_a[notnan_indices(pmo_a)], None,
                                         outlier_fraction=ARGS.outlier_fraction)
pmo_a[outliers_a] = np.nan

outliers_b = notnan_indices(pmo_b)
outliers_b[outliers_b] = detect_outliers(pmo_b[notnan_indices(pmo_b)], None,
                                         outlier_fraction=ARGS.outlier_fraction)
pmo_b[outliers_b] = np.nan

pmo_outliers_a = pmo_a
pmo_outliers_a[~outliers_a] = np.nan

pmo_outliers_b = pmo_b
pmo_outliers_b[~outliers_b] = np.nan
data[C.A] = pmo_a
data[C.B] = pmo_b

# Calculate exposure
data[C.EA] = compute_exposure(pmo_a, "exposure_sum", pmo_b[notnan_indices(pmo_b)].mean())
data[C.EB] = compute_exposure(pmo_b, "exposure_sum", pmo_b[notnan_indices(pmo_b)].mean())
e_a = data[C.EA].values
e_b = data[C.EB].values

# Extract not nan rows
data_nn = data[[C.T, C.A, C.B, C.EA, C.EB]].dropna()
t_nn = data_nn[C.T].values
e_a_nn = data_nn[C.EA].values
e_b_nn = data_nn[C.EB].values
x_a_nn = data_nn[C.A].values
x_b_nn = data_nn[C.B].values

# Relative degradation ratio PMO6V-A to PMO6V-B
ratio_a_b = np.divide(x_a_nn, x_b_nn)

ratio_a_b_initial = None
ratio_a_b_c = None
x_a_nn_c = None
x_b_nn_c = None
parameters_opt = None
MODEL = None
WINDOW = ARGS.window

if ARGS.model_type.upper() == C.EXP_LIN:
    MODEL = C.EXP_LIN
    # Initial parameter estimation
    gamma_initial, lambda_initial, e_0_initial = initial_fit(ratio_a_b, e_a_nn)
    ratio_a_b_initial = DegradationModels.exp_unc_model(e_a_nn, gamma_initial, lambda_initial, e_0_initial)

    parameters_initial = [gamma_initial, lambda_initial, e_0_initial, 0]
    parameters_opt, x_a_nn_c, x_b_nn_c = em_estimate_exp_lin(x_a_nn, x_b_nn, e_a_nn, e_b_nn, parameters_initial)
    ratio_a_b_c = np.divide(x_a_nn_c, x_b_nn_c)

elif ARGS.model_type.upper() == C.EXP:
    MODEL = C.EXP
    gamma_initial, lambda_initial, e_0_initial = initial_fit(ratio_a_b, e_a_nn)
    ratio_a_b_initial = DegradationModels.exp_unc_model(e_a_nn, gamma_initial, lambda_initial, e_0_initial)

    parameters_initial = [gamma_initial, lambda_initial, e_0_initial, 0]
    parameters_opt, x_a_nn_c, x_b_nn_c = em_estimate_exp(x_a_nn, x_b_nn, e_a_nn, e_b_nn, parameters_initial)
    ratio_a_b_c = np.divide(x_a_nn_c, x_b_nn_c)


# Results
k_a = 10000
x_a = pmo_a
idx_a = notnan_indices(x_a)
x_a = downsample_signal(x_a[idx_a], k_a)
t_a = downsample_signal(t[idx_a], k_a)
e_a = downsample_signal(e_a[idx_a], k_a)

k_b = 10
x_b = pmo_b
idx_b = notnan_indices(x_b)
x_b = downsample_signal(x_b[idx_b], k_b)
t_b = downsample_signal(t[idx_b], k_b)
e_b = downsample_signal(e_b[idx_b], k_b)

x_a_c = None
x_b_c = None
deg_a = None
deg_b = None

if ARGS.model_type.upper() == C.EXP_LIN:
    x_a_c = x_a / DegradationModels.exp_lin_model(e_a, *parameters_opt)
    x_b_c = x_b / DegradationModels.exp_lin_model(e_b, *parameters_opt)
    deg_a = DegradationModels.exp_lin_model(e_a_nn, *parameters_opt)
    deg_b = DegradationModels.exp_lin_model(e_b_nn, *parameters_opt)

elif ARGS.model_type.upper() == C.EXP:
    x_a_c = x_a / DegradationModels.exp_model(e_a, *parameters_opt)
    x_b_c = x_b / DegradationModels.exp_model(e_b, *parameters_opt)
    deg_a = DegradationModels.exp_model(e_a_nn, *parameters_opt)
    deg_b = DegradationModels.exp_model(e_b_nn, *parameters_opt)


figs = []
fig = visualizer.plot_signals([(t_nn, ratio_a_b, "RATIO_{}_{}_raw".format(C.A, C.B), False),
                               (t_nn, ratio_a_b_initial, "RATIO_{}_{}_initial_fit".format(C.A, C.B), False)],
                              results_dir, "RATIO_{}_{}_raw_initial_fit".format(C.A, C.B), x_ticker=1,
                              legend="upper right", x_label=C.YEAR_UNIT, y_label=C.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_nn, x_a_nn, "{}_raw_nn".format(C.A)),
                               (t_nn, x_b_nn, "{}_raw_nn".format(C.B)),
                               (t_nn, x_a_nn_c, "{}_raw_nn_corrected".format(C.A)),
                               (t_nn, x_b_nn_c, "{}_raw_nn_corrected".format(C.B))],
                              results_dir, "{}_{}_{}_raw_corrected".format(MODEL, C.A, C.B), x_ticker=1,
                              legend="upper right", x_label=C.YEAR_UNIT, y_label=C.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_nn, ratio_a_b, "RATIO_{}_{}_raw".format(C.A, C.B)),
                               (t_nn, ratio_a_b_c, "RATIO_{}_{}_corrected".format(C.A, C.B)),
                               (t_nn, deg_a, "DEGRADATION_{}".format(C.A)),
                               (t_nn, deg_b, "DEGRADATION_{}".format(C.B))],
                              results_dir, "{}_RATIO_DEGRADATION_{}_{}_raw_corrected".format(MODEL, C.A, C.B),
                              x_ticker=1, legend="upper right", x_label=C.YEAR_UNIT, y_label=C.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals([(t_a, x_a, "{}_raw".format(C.A)),
                               (t_b, x_b, "{}_raw".format(C.B)),
                               (t_a, x_a_c, "{}_raw_corrected".format(C.A)),
                               (t_b, x_b_c, "{}_raw_corrected".format(C.B))],
                              results_dir, "{}_{}_{}_raw_corrected_full".format(MODEL, C.A, C.B), x_ticker=1,
                              legend="upper right", x_label=C.YEAR_UNIT, y_label=C.TSI_UNIT)
figs.append(fig)

fig = visualizer.plot_signals_mean_std([(t_a, x_a, "{}_conf_int".format(C.A), WINDOW),
                                        (t_b, x_b, "{}_conf_int".format(C.B), WINDOW),
                                        (t_a, x_a_c, "{}_corrected_conf_int".format(C.A), WINDOW),
                                        (t_b, x_b_c, "{}_corrected_conf_int".format(C.B), WINDOW)],
                                       results_dir, "{}_{}_{}_raw_corrected_full_conf_int".format(MODEL, C.A, C.B),
                                       x_ticker=1, legend="lower left", x_label=C.YEAR_UNIT, y_label=C.TSI_UNIT)
figs.append(fig)


if ARGS.visualize:
    for fig in figs:
        fig.show()
