from enum import Enum, auto

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from data_utils import downsample_signal, not_nan_indices, mission_day_to_year, moving_average_std


class ExposureMode(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


class ModelType(Enum):
    EXP_LIN_UNC = auto()
    EXP_LIN = auto()
    EXP_UNC = auto()
    EXP = auto()


class ModelFitter:
    def __init__(self, data, timestamp_field_name, signal_a_field_name, signal_b_field_name, temperature_field_name,
                 exposure_mode, model_type):
        self.t = data[timestamp_field_name]
        self.signal_a = data[signal_a_field_name].values
        self.signal_b = data[signal_b_field_name].values

        # Calculate exposure
        signal_b_mean = float(np.mean(self.signal_b[~np.isnan(self.signal_b)]))
        self.exposure_a = self._compute_exposure(self.signal_a, exposure_mode, signal_b_mean)
        self.exposure_b = self._compute_exposure(self.signal_b, exposure_mode, signal_b_mean)
        data["exposure_a"] = self.exposure_a
        data["exposure_b"] = self.exposure_b

        # Extract not nan rows
        data_not_nan = data[
            [timestamp_field_name, signal_a_field_name, signal_b_field_name, "exposure_a", "exposure_b"]].dropna()
        self.t_not_nan = mission_day_to_year(data_not_nan[timestamp_field_name].values)
        self.exposure_a_not_nan = data_not_nan["exposure_a"].values
        self.exposure_b_not_nan = data_not_nan["exposure_b"].values
        self.signal_a_not_nan = data_not_nan[signal_a_field_name].values
        self.signal_b_not_nan = data_not_nan[signal_b_field_name].values

        # Relative degradation ratio
        self.ratio_a_b = self.signal_a_not_nan / self.signal_b_not_nan

        # Get model
        self.model_function = self._get_function_from_model_type(model_type)

        # TODO: remove unneeded unpacking
        gamma_initial, lambda_initial, e_0_initial = self._initial_fit(self.ratio_a_b, self.exposure_a_not_nan)
        self.ratio_a_b_initial = self._exp_unc(self.exposure_a_not_nan, gamma_initial, lambda_initial, e_0_initial)

        parameters_initial = self._get_initial_parameters_from_model_type(model_type, lambda_initial, e_0_initial)
        self.parameters_opt, self.signal_a_not_nan_corrected, self.signal_b_not_nan_corrected = self._em_estimate(
            self.model_function,
            self.signal_a_not_nan,
            self.signal_b_not_nan,
            self.exposure_a_not_nan,
            self.exposure_b_not_nan,
            parameters_initial)
        self.ratio_a_b_corrected = self.signal_a_not_nan_corrected / self.signal_b_not_nan_corrected

        self.degradation_a = None
        self.degradation_b = None
        self.t_a_downsample = None
        self.t_b_downsample = None
        self.signal_a_downsample = None
        self.signal_b_downsample = None
        self.signal_a_downsample_corrected = None
        self.signal_b_downsample_corrected = None
        self.signal_a_downsample_moving_average = None
        self.signal_a_downsample_std = None
        self.signal_a_corrected_moving_average = None
        self.signal_a_corrected_std = None
        self.signal_b_downsample_moving_average = None
        self.signal_b_downsample_std = None
        self.signal_b_corrected_moving_average = None
        self.signal_b_corrected_std = None
        self.get_downsampled_result()

    def get_downsampled_result(self, downsample_factor_a=10000, downsample_factor_b=10, window_size=81):
        # Downsample a
        not_nan_indices_a = not_nan_indices(self.signal_a)
        self.signal_a_downsample = downsample_signal(self.signal_a[not_nan_indices_a], downsample_factor_a)
        t_a_downsample = downsample_signal(self.t[not_nan_indices_a], downsample_factor_a)
        exposure_a_downsample = downsample_signal(self.exposure_a[not_nan_indices_a], downsample_factor_a)

        # Downsample b
        not_nan_indices_b = not_nan_indices(self.signal_b)
        self.signal_b_downsample = downsample_signal(self.signal_b[not_nan_indices_b], downsample_factor_b)
        t_b_downsample = downsample_signal(self.t[not_nan_indices_b], downsample_factor_b)
        exposure_b_downsample = downsample_signal(self.exposure_b[not_nan_indices_b], downsample_factor_b)

        # Mission day to year
        self.t_a_downsample = mission_day_to_year(t_a_downsample)
        self.t_b_downsample = mission_day_to_year(t_b_downsample)

        self.signal_a_downsample_corrected = self.signal_a_downsample / self.model_function(exposure_a_downsample,
                                                                                            *self.parameters_opt)
        self.signal_b_downsample_corrected = self.signal_b_downsample / self.model_function(exposure_b_downsample,
                                                                                            *self.parameters_opt)
        self.degradation_a = self.model_function(self.exposure_a_not_nan, *self.parameters_opt)
        self.degradation_b = self.model_function(self.exposure_b_not_nan, *self.parameters_opt)

        self.signal_a_downsample_moving_average, self.signal_a_downsample_std = moving_average_std(
            self.signal_a_downsample,
            w=window_size)
        self.signal_a_corrected_moving_average, self.signal_a_corrected_std = moving_average_std(
            self.signal_a_downsample_corrected,
            w=window_size)
        self.signal_b_downsample_moving_average, self.signal_b_downsample_std = moving_average_std(
            self.signal_b_downsample,
            w=window_size)
        self.signal_b_corrected_moving_average, self.signal_b_corrected_std = moving_average_std(
            self.signal_b_downsample_corrected,
            w=window_size)

    @staticmethod
    def _initial_fit(ratio_a_b, exposure_a):
        """y(t) = gamma + exp(-lambda_ * (exposure_a - e_0))"""
        # TODO: Auto?? epsilon = 1e-5
        epsilon = 1e-5
        # epsilon = (ratio_a_b.max() - ratio_a_b.min()) / 100
        gamma = ratio_a_b.min()

        y = np.log(ratio_a_b - gamma + epsilon)
        x = exposure_a.reshape(-1, 1)

        regression = LinearRegression(fit_intercept=True)
        regression.fit(x, y)

        lambda_ = -regression.coef_[0]
        e_0 = regression.intercept_ / lambda_

        return gamma, lambda_, e_0

    @staticmethod
    def _compute_exposure(x, mode=ExposureMode.NUM_MEASUREMENTS, mean=1.0):
        if mode == ExposureMode.NUM_MEASUREMENTS:
            x = np.nan_to_num(x) > 0
            return np.cumsum(x)
        elif mode == ExposureMode.EXPOSURE_SUM:
            x = np.nan_to_num(x)
            x = x / mean
            return np.cumsum(x)

    @staticmethod
    def _exp_lin_unc(x, gamma, lambda_, e_0, linear):
        """Unconstrained exponential-linear degradation model."""
        return np.exp(-lambda_ * (x - e_0)) + gamma + linear * x

    @staticmethod
    def _exp_lin(x, lambda_, e_0, linear):
        """Constrained exponential-linear degradation model: y(0) = 1."""
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0)) + linear * x
        return y

    @staticmethod
    def _exp_unc(x, gamma, lambda_, e_0):
        """Unconstrained exponential-linear degradation model."""
        return np.exp(-lambda_ * (x - e_0)) + gamma

    @staticmethod
    def _exp(x, lambda_, e_0):
        """Constrained exponential degradation model: y(0) = 1."""
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0))
        return y

    def _get_function_from_model_type(self, model_type):
        if model_type == ModelType.EXP_LIN:
            return self._exp_lin
        return self._exp

    @staticmethod
    def _get_initial_parameters_from_model_type(model_type, lambda_initial, e_0_initial):
        if model_type == ModelType.EXP_LIN:
            return lambda_initial, e_0_initial, 0
        return lambda_initial, e_0_initial

    @staticmethod
    def _em_estimate(model_function, x_a_raw, x_b_raw, exposure_a, exposure_b, parameters_initial,
                     epsilon=1e-7):
        x_a_corrected = None
        x_b_corrected = x_b_raw

        parameters_prev = np.zeros_like(parameters_initial)
        parameters_opt = parameters_initial

        convergence = True
        while convergence:
            ratio_a_b_corrected = x_a_raw / x_b_corrected

            parameters_opt, _ = curve_fit(model_function, exposure_a, ratio_a_b_corrected, p0=parameters_initial)

            x_a_corrected = x_a_raw / model_function(exposure_a, *parameters_opt)
            x_b_corrected = x_b_raw / model_function(exposure_b, *parameters_opt)

            delta_norm = np.linalg.norm(parameters_prev - parameters_opt) / np.linalg.norm(parameters_prev)

            print("norm\t", delta_norm)
            parameters_prev = parameters_opt

            convergence = delta_norm > epsilon

        return parameters_opt, x_a_corrected, x_b_corrected
