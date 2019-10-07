import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


def initial_fit(ratio_a_b, e_a):
    """
        y(t) = gamma + exp(-lambda_ * (e_a - e_0))
    """
    # TODO: Auto?? epsilon = 1e-5
    epsilon = 1e-5
    # epsilon = (ratio_a_b.max() - ratio_a_b.min()) / 100
    gamma = ratio_a_b.min()

    y = np.log(ratio_a_b - gamma + epsilon)
    x = e_a.reshape(-1, 1)

    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(x, y)

    lambda_ = -regressor.coef_[0]
    e_0 = regressor.intercept_ / lambda_

    return gamma, lambda_, e_0


def compute_exposure(x, mode="num_measurements", mean=1):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.values

    if mode == "num_measurements":
        x = np.nan_to_num(x) > 0
        return np.cumsum(x)
    elif mode == "exposure_sum" and mean:
        x = np.nan_to_num(x)
        x = x / mean
        return np.cumsum(x)


def em_estimate_exp(x_a_raw, x_b_raw, e_a, e_b, parameters_initial, epsilon=1e-7):
    x_a_c = None
    x_b_c = x_b_raw

    parameters_prev = np.zeros(shape=(2,))
    parameters_opt = [parameters_initial[1], parameters_initial[2]]

    convergence = True
    while convergence:
        ratio_a_b_c = x_a_raw / x_b_c

        parameters_opt, _ = curve_fit(DegradationModels.exp_model, e_a, ratio_a_b_c,
                                      p0=(parameters_initial[1], parameters_initial[2]))

        x_a_c = x_a_raw / DegradationModels.exp_model(e_a, *parameters_opt)
        x_b_c = x_b_raw / DegradationModels.exp_model(e_b, *parameters_opt)

        delta_norm = np.linalg.norm(parameters_prev - parameters_opt) / np.linalg.norm(parameters_prev)

        print("norm\t", delta_norm)
        parameters_prev = parameters_opt

        convergence = delta_norm > epsilon

    return parameters_opt, x_a_c, x_b_c


def em_estimate_exp_lin(x_a_raw, x_b_raw, e_a, e_b, parameters_initial, epsilon=1e-7):
    x_a_c = None
    x_b_c = x_b_raw

    parameters_prev = np.zeros(shape=(3,))
    parameters_opt = [parameters_initial[1], parameters_initial[2], parameters_initial[3]]

    convergence = True
    while convergence:
        ratio_a_b_c = x_a_raw / x_b_c

        parameters_opt, _ = curve_fit(DegradationModels.exp_lin_model, e_a, ratio_a_b_c,
                                      p0=(parameters_initial[1], parameters_initial[2], parameters_initial[3]))

        x_a_c = x_a_raw / DegradationModels.exp_lin_model(e_a, *parameters_opt)
        x_b_c = x_b_raw / DegradationModels.exp_lin_model(e_b, *parameters_opt)

        delta_norm = np.linalg.norm(parameters_prev - parameters_opt) / np.linalg.norm(parameters_prev)

        print("norm\t", delta_norm)
        parameters_prev = parameters_opt

        convergence = delta_norm > epsilon

    return parameters_opt, x_a_c, x_b_c


class DegradationModels(object):

    @staticmethod
    def exp_lin_unc_model(x, gamma, lambda_, e_0, linear):
        """
            Unconstrained exponential-linear degradation model.
        """
        return np.exp(-lambda_ * (x - e_0)) + gamma + linear * x

    @staticmethod
    def exp_unc_model(x, gamma, lambda_, e_0):
        """
            Unconstrained exponential-linear degradation model.
        """
        return np.exp(-lambda_ * (x - e_0)) + gamma

    @staticmethod
    def exp_model(x, lambda_, e_0):
        """
            Constrained exponential degradation model: y(0) = 1.
        """
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0))
        return y

    @staticmethod
    def exp_lin_model(x, lambda_, e_0, linear):
        """
            Constrained exponential-linear degradation model: y(0) = 1.
        """
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0)) + linear * x
        return y
