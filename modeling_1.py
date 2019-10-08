from data_utils import load_data, make_dir, downsample_signal, moving_average_std, notnan_indices, mission_day_to_year
from constants import Constants as C
from modeling import compute_exposure, initial_fit, em_estimate_exp_lin, em_estimate_exp, DegradationModels

import os
import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="exp_lin", help="Model to train.")
parser.add_argument("--visualize", action="store_true", help="Flag for visualizing results.")
parser.add_argument("--window", type=int, default=81, help="Moving average window size.")

ARGS = parser.parse_args()

# Parameters
data_dir = C.DATA_DIR
results_dir = C.RESULTS_DIR
virgo_file = C.VIRGO_FILE


results_dir = os.path.join(results_dir, datetime.datetime.now().strftime("modeling_%Y-%m-%d"))
make_dir(results_dir)

# Load data
data = load_data(os.path.join(data_dir, virgo_file))

t = data["timestamp"]
pmo_a = data["pmo6v_a"]
pmo_b = data["pmo6v_b"]

# Calculate exposure
data["exposure_a"] = compute_exposure(pmo_a, "exposure_sum", pmo_b.mean())
data["exposure_b"] = compute_exposure(pmo_b, "exposure_sum", pmo_b.mean())
e_a = data["exposure_a"].values
e_b = data["exposure_b"].values

# Extract not nan rows
data_nn = data[["timestamp", "pmo6v_a", "pmo6v_b", "exposure_a", "exposure_b"]].dropna()
t_nn = data_nn["timestamp"].values
e_a_nn = data_nn["exposure_a"].values
e_b_nn = data_nn["exposure_b"].values
x_a_nn = data_nn["pmo6v_a"].values
x_b_nn = data_nn["pmo6v_b"].values

# Relative degradation ratio PMO6V-A to PMO6V-B
ratio_a_b = x_a_nn/x_b_nn

ratio_a_b_initial = None
ratio_a_b_c = None
x_a_nn_c = None
x_b_nn_c = None
parameters_opt = None


if ARGS.model_type == "exp_lin":
    # Initial parameter estimation
    gamma_initial, lambda_initial, e_0_initial = initial_fit(ratio_a_b, e_a_nn)
    ratio_a_b_initial = DegradationModels.exp_unc_model(e_a_nn, gamma_initial, lambda_initial, e_0_initial)

    parameters_initial = [gamma_initial, lambda_initial, e_0_initial, 0]
    parameters_opt, x_a_nn_c, x_b_nn_c = em_estimate_exp_lin(x_a_nn, x_b_nn, e_a_nn, e_b_nn, parameters_initial)
    ratio_a_b_c = x_a_nn_c / x_b_nn_c

elif ARGS.model_type == "exp":
    gamma_initial, lambda_initial, e_0_initial = initial_fit(ratio_a_b, e_a_nn)
    ratio_a_b_initial = DegradationModels.exp_unc_model(e_a_nn, gamma_initial, lambda_initial, e_0_initial)

    parameters_initial = [gamma_initial, lambda_initial, e_0_initial, 0]
    parameters_opt, x_a_nn_c, x_b_nn_c = em_estimate_exp(x_a_nn, x_b_nn, e_a_nn, e_b_nn, parameters_initial)
    ratio_a_b_c = x_a_nn_c / x_b_nn_c


# Results
k_a = 10000
x_a = pmo_a.values
idx_a = notnan_indices(x_a)
x_a = downsample_signal(x_a[idx_a], k_a)
t_a = downsample_signal(t[idx_a], k_a)
e_a = downsample_signal(e_a[idx_a], k_a)

k_b = 10
x_b = pmo_b.values
idx_b = notnan_indices(x_b)
x_b = downsample_signal(x_b[idx_b], k_b)
t_b = downsample_signal(t[idx_b], k_b)
e_b = downsample_signal(e_b[idx_b], k_b)

# Mission day to year
t_nn = np.array(list(map(mission_day_to_year, t_nn)))
t_a = np.array(list(map(mission_day_to_year, t_a)))
t_b = np.array(list(map(mission_day_to_year, t_b)))

x_a_c = None
x_b_c = None
deg_a = None
deg_b = None

if ARGS.model_type == "exp_lin":
    x_a_c = x_a / DegradationModels.exp_lin_model(e_a, *parameters_opt)
    x_b_c = x_b / DegradationModels.exp_lin_model(e_b, *parameters_opt)
    deg_a = DegradationModels.exp_lin_model(e_a_nn, *parameters_opt)
    deg_b = DegradationModels.exp_lin_model(e_b_nn, *parameters_opt)

elif ARGS.model_type == "exp":
    x_a_c = x_a / DegradationModels.exp_model(e_a, *parameters_opt)
    x_b_c = x_b / DegradationModels.exp_model(e_b, *parameters_opt)
    deg_a = DegradationModels.exp_model(e_a_nn, *parameters_opt)
    deg_b = DegradationModels.exp_model(e_b_nn, *parameters_opt)

x_a_ma, x_a_std = moving_average_std(x_a, w=ARGS.window, center=True)
x_a_c_ma, x_a_c_std = moving_average_std(x_a_c, w=ARGS.window, center=True)
x_b_ma, x_b_std = moving_average_std(x_b, w=ARGS.window, center=True)
x_b_c_ma, x_b_c_std = moving_average_std(x_b_c, w=ARGS.window, center=True)

style.use(C.MATPLOTLIB_STYLE)

plt.figure(1, figsize=C.FIG_SIZE)
plt.plot(t_nn, ratio_a_b, t_nn, ratio_a_b_initial)
plt.title("PMO6V-a to PMO6V-b ratio - raw, initial fit")
plt.savefig(os.path.join(results_dir, ARGS.model_type + "_ratio_a_b_raw_initial.pdf"),
            bbox_inches="tight", quality=100, dpi=200)

plt.figure(2, figsize=C.FIG_SIZE)
plt.plot(t_nn, x_b_nn, t_nn, x_a_nn_c, t_nn, x_b_nn_c)
plt.legend(["pmo_b", "pmo_a_c", "pmo_b_c"])
plt.title("PMO6V-a and PMO6V-b - raw, degradation corrected")
plt.savefig(os.path.join(results_dir, ARGS.model_type + "_pmo_a_b_c.pdf"),
            bbox_inches="tight", quality=100, dpi=200)

plt.figure(3, figsize=C.FIG_SIZE)
plt.plot(t_nn, ratio_a_b, t_nn, ratio_a_b_c, t_nn, deg_a, t_nn, deg_b)
plt.title("PMO6V-a to PMO6V-b ratio - raw, degradation corrected")
plt.legend(["ratio_a_b_raw", "ratio_a_b_c", "deg_a_opt", "deg_b_opt"])
plt.savefig(os.path.join(results_dir,  ARGS.model_type + "_ratio_a_b_raw_opt.pdf"),
            bbox_inches="tight", quality=100, dpi=200)

plt.figure(4, figsize=C.FIG_SIZE)
plt.scatter(t_a, x_a, marker="x", c="b")
plt.scatter(t_b, x_b, marker="x", c="r")
plt.plot(t_a, x_a_c)
plt.plot(t_b, x_b_c)
plt.title("PMO6V-a and PMO6V-b - raw, degradation corrected")
plt.legend(["pmo_a", "pmo_b", "pmo_a_c", "pmo_b_c"], loc="lower left")
plt.savefig(os.path.join(results_dir,  ARGS.model_type + "_pmo_a_b_c_full.pdf"),
            bbox_inches="tight", quality=100, dpi=200)

plt.figure(5, figsize=C.FIG_SIZE)
plt.plot(t_a, x_a_ma, color="tab:blue")
plt.fill_between(t_a, x_a_ma - 1.96 * x_a_std, x_a_ma + 1.96 * x_a_std,
                 facecolor='tab:blue', alpha=0.5, label='95% confidence interval')

plt.plot(t_b, x_b_ma, color="tab:orange")
plt.fill_between(t_b, x_b_ma - 1.96 * x_b_std, x_b_ma + 1.96 * x_b_std,
                 facecolor='tab:orange', alpha=0.5, label='95% confidence interval')

plt.plot(t_a, x_a_c_ma, color="tab:green")
plt.fill_between(t_a, x_a_c_ma - 1.96 * x_a_c_std, x_a_c_ma + 1.96 * x_a_c_std,
                 facecolor='tab:green', alpha=0.5, label='95% confidence interval')

plt.plot(t_b, x_b_c_ma, color="tab:red")
plt.fill_between(t_b, x_b_c_ma - 1.96 * x_b_c_std, x_b_c_ma + 1.96 * x_b_c_std,
                 facecolor='tab:red', alpha=0.5, label='95% confidence interval')

plt.title("PMO6V-a and PMO6V-b - raw, degradation corrected, moving average")
plt.legend(["pmo_a_ma", "pmo_b_ma", "pmo_a_c_ma", "pmo_b_c_ma"], loc="lower left")
plt.savefig(os.path.join(results_dir,  ARGS.model_type + "_pmo_a_b_c_full_ma.pdf"),
            bbox_inches="tight", quality=100, dpi=200)

if ARGS.visualize:
    plt.show()
