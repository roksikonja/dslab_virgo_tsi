from data_utils import load_data, make_dir
from constants import Constants as C
from modeling import compute_exposure, initial_fit, em_estimate, DegradationModels

import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

# Parameters
data_dir = C.DATA_DIR
results_dir = C.RESULTS_DIR

virgo_file = C.VIRGO_FILE

modeling_dir = os.path.join(results_dir, datetime.datetime.now().strftime("modeling_%Y-%m-%d"))
make_dir(modeling_dir)

# Load data
data = load_data(os.path.join(data_dir, virgo_file))

t = data["timestamp"]
pmo_a = data["pmo6v_a"]
pmo_b = data["pmo6v_b"]

# Calculate exposure
data["exposure_a"] = compute_exposure(pmo_a, "exposure_sum", pmo_b.mean())
data["exposure_b"] = compute_exposure(pmo_b, "exposure_sum", pmo_b.mean())

# Extract not nan rows
data_nn = data[["timestamp", "pmo6v_a", "pmo6v_b", "exposure_a", "exposure_b"]].dropna()
t_nn = data_nn["timestamp"].values
e_a_nn = data_nn["exposure_a"].values
e_b_nn = data_nn["exposure_b"].values
x_a_nn = data_nn["pmo6v_a"].values
x_b_nn = data_nn["pmo6v_b"].values

# Relative degradation ratio PMO6V-A to PMO6V-B
ratio_a_b = x_a_nn/x_b_nn

# Initial parameter estimation
gamma_initial, lambda_initial, e_0_initial = initial_fit(ratio_a_b, e_a_nn)
ratio_a_b_initial = DegradationModels.exp_unc_model(e_a_nn, gamma_initial, lambda_initial, e_0_initial)

# plt.figure(1, figsize=(16, 8))
# plt.plot(t_nn, ratio_a_b, t_nn, ratio_a_b_initial)
# plt.savefig(os.path.join(results_dir, "ratio_a_b_raw.pdf"), bbox_inches="tight", quality=100, dpi=200)
# plt.show()

parameters_initial = [gamma_initial, lambda_initial, e_0_initial, 0]
parameters_opt = em_estimate(x_a_nn, x_b_nn, e_a_nn, e_b_nn, t_nn, parameters_initial)


