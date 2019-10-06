import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def initial_fit(ratio_a_b, e_a):
    """
        y(t) = gamma + exp(-lambda_ * (e_a - e_0))
    """
    # TODO: Auto?? epsilon = 1e-5

    epsilon = (ratio_a_b.max() - ratio_a_b.min()) / 100
    gamma = ratio_a_b.min()

    y = np.log(ratio_a_b - gamma + epsilon)
    X = e_a.reshape(-1, 1)

    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(X, y)

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
    def exponential_linear_model(x, lambda_, e_0, linear):
        """
            Constrained exponential-linear degradation model: y(0) = 1.
        """
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0)) + linear * x
        return y


