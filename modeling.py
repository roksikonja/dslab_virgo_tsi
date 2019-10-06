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


def em_estimate(x_a_raw, x_b_raw, e_a, e_b, t, parameters_initial, visualize=True):
    x_a_c = None
    ratio_a_b_c = None

    x_b_c = x_b_raw
    ratio_a_b = x_a_raw / x_b_raw

    parameters_prev = np.zeros(shape=(3,))
    parameters_opt = parameters_prev

    num_iter = 50
    for i in range(1, num_iter + 1):
        ratio_a_b_c = x_a_raw / x_b_c

        parameters_opt, _ = curve_fit(DegradationModels.exp_lin_model, e_a, ratio_a_b_c,
                                      p0=(parameters_initial[1], parameters_initial[2], parameters_initial[3]))

        x_a_c = x_a_raw / DegradationModels.exp_lin_model(e_a, *parameters_opt)
        x_b_c = x_b_raw / DegradationModels.exp_lin_model(e_b, *parameters_opt)

        print("norm\t", np.linalg.norm(parameters_prev - parameters_opt), parameters_opt)
        parameters_prev = parameters_opt

        # if visualize:
        #     plt.figure(4, figsize=(16, N * 8))
        #     plt.subplot(2*N, 1, 2*(i-1)+1)
        #     plt.plot(t_nn, x_b_nn, t_nn, x_a_nn_c_lin, t_nn, x_b_nn_c_lin)
        #     plt.legend(["pmo_b", "pmo_a_c_lin", "pmo_b_c_lin"])
        #     plt.subplot(2*N, 1, 2*(i-1)+2)
        #     plt.plot(t_nn, ratio_a_b, t_nn, ratio, t_nn, exponential_linear_model(e_a_nn, *popt))
        #     plt.legend(["ratio_a_b", "ratio"])

    if visualize:
        plt.figure(4, figsize=(16, 8))
        plt.plot(t, x_b_raw, t, x_a_c, t, x_b_c)
        plt.legend(["pmo_b", "pmo_a_c", "pmo_b_c"])
        plt.show()

        plt.figure(5, figsize=(16, 8))
        plt.plot(t, ratio_a_b, t, ratio_a_b_c, t, DegradationModels.exp_lin_model(e_a, *parameters_opt))
        plt.legend(["ratio_a_b_raw", "ratio_a_b_c", "exp_lin_opt"])
        plt.show()

    return parameters_opt


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
