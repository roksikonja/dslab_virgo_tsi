from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from dslab_virgo_tsi.data_utils import resampling_average_std, downsample_signal, notnan_indices, detect_outliers


class ExposureMode(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


# TODO: ratio_a_b = history[0].iteration_ratio_a_b
# TODO: signal_a_mutual_nn_corrected = history[-1].iteration_signal_a
# TODO: signal_a_mutual_nn = history[0].iteration_signal_a
# TODO: filter outliers
class ModelingResult:
    def __init__(self):
        self.t_mutual_nn = None
        self.history_mutual_nn = None
        self.t_a_nn = None
        self.t_b_nn = None
        self.signal_a_nn = None
        self.signal_b_nn = None
        self.signal_a_nn_corrected = None
        self.signal_b_nn_corrected = None
        self.t_hourly_out = None
        self.signal_hourly_out = None
        self.t_daily_out = None
        self.signal_daily_out = None

    def downsample_signals(self, k_a, k_b):
        self.t_a_nn = downsample_signal(self.t_a_nn, k_a)
        self.t_b_nn = downsample_signal(self.t_b_nn, k_b)
        self.signal_a_nn = downsample_signal(self.signal_a_nn, k_a)
        self.signal_b_nn = downsample_signal(self.signal_b_nn, k_b)
        self.signal_a_nn_corrected = downsample_signal(self.signal_a_nn_corrected, k_a)
        self.signal_b_nn_corrected = downsample_signal(self.signal_b_nn_corrected, k_b)


class IterationResult:
    def __init__(self, iteration_signal_a, iteration_signal_b, iteration_ratio_a_b):
        self.iteration_signal_a = iteration_signal_a
        self.iteration_signal_b = iteration_signal_b
        self.iteration_ratio_a_b = iteration_ratio_a_b


class BaseModel(ABC):
    """
    mutual_nn -> values taken at times when all signals are not nan
    nn -> values taken at times when specific signal is not nan
    """

    def __init__(self, data, timestamp_field_name, signal_a_field_name, signal_b_field_name, exposure_mode,
                 moving_average_window, outlier_fraction):
        self.result = ModelingResult()

        signal_a = data[signal_a_field_name].values
        signal_b = data[signal_b_field_name].values
        t = data[timestamp_field_name].values

        # Parameters
        self.moving_average_window = moving_average_window
        self.outlier_fraction = outlier_fraction

        if self.outlier_fraction > 0:
            data = self._filter_outliers(data, signal_a_field_name, signal_b_field_name)

        # Calculate exposure
        signal_b_mean = float(np.mean(signal_b[~np.isnan(signal_b)]))
        exposure_a = self._compute_exposure(signal_a, exposure_mode, signal_b_mean)
        exposure_b = self._compute_exposure(signal_b, exposure_mode, signal_b_mean)
        data["e_a"] = exposure_a
        data["e_b"] = exposure_b

        # Not nan rows (non-mutual)
        index_a_nn = ~np.isnan(signal_a)
        index_b_nn = ~np.isnan(signal_b)
        signal_a_nn = signal_a[index_a_nn]
        signal_b_nn = signal_b[index_b_nn]
        t_a_nn = t[index_a_nn]
        t_b_nn = t[index_b_nn]
        exposure_a_nn = exposure_a[index_a_nn]
        exposure_b_nn = exposure_b[index_b_nn]

        # Extract mutual not nan rows
        data_mutual_nn = data[[timestamp_field_name, signal_a_field_name, signal_b_field_name, "e_a", "e_b"]].dropna()
        t_mutual_nn = data_mutual_nn[timestamp_field_name].values
        exposure_a_mutual_nn = data_mutual_nn["e_a"].values
        exposure_b_mutual_nn = data_mutual_nn["e_b"].values
        signal_a_mutual_nn = data_mutual_nn[signal_a_field_name].values
        signal_b_mutual_nn = data_mutual_nn[signal_b_field_name].values

        # Variable needed for initial fit
        self.ratio_a_b_mutual_nn = np.divide(exposure_a_mutual_nn, exposure_b_mutual_nn)

        # Variables needed in subclasses
        self.t_mutual_nn = t_mutual_nn
        self.signal_a_nn = signal_a_nn
        self.signal_b_nn = signal_b_nn
        self.t_a_nn = t_a_nn
        self.t_b_nn = t_b_nn
        self.signal_a_mutual_nn = signal_a_mutual_nn
        self.signal_b_mutual_nn = signal_b_mutual_nn
        self.exposure_a_mutual_nn = exposure_a_mutual_nn
        self.exposure_b_mutual_nn = exposure_b_mutual_nn
        self.exposure_a_nn = exposure_a_nn
        self.exposure_b_nn = exposure_b_nn
        self.degradation_a = None
        self.degradation_b = None
        self.t_hourly_out = None
        self.signal_hourly_out = None
        self.t_daily_out = None
        self.signal_daily_out = None

    def _compute_corrections(self):
        print("Compute corrections")
        # Compute corrected signals (whole history)
        self.history_mutual_nn, self.parameters_opt = self._iterative_correction(self.signal_a_mutual_nn,
                                                                                 self.signal_b_mutual_nn,
                                                                                 self.exposure_a_mutual_nn,
                                                                                 self.exposure_b_mutual_nn)
        print(self.parameters_opt)

    def _compute_output(self):
        min_time = np.floor(self.t_a_nn.min())
        max_time = np.ceil(self.t_a_nn.max())

        self.t_hourly_out = np.arange(min_time, max_time, 1.0 / 24.0)
        self.t_daily_out = np.arange(min_time, max_time, 1.0)
        self.signal_hourly_out = np.zeros(shape=self.t_hourly_out.shape)
        self.signal_daily_out = np.zeros(shape=self.t_daily_out.shape)

        x_a_hourly_mean, x_a_hourly_std = resampling_average_std(self.signal_a_nn, self.t_a_nn, self.t_hourly_out,
                                                                 w=self.moving_average_window)
        x_b_hourly_mean, x_b_hourly_std = resampling_average_std(self.signal_b_nn, self.t_b_nn, self.t_hourly_out,
                                                                 w=self.moving_average_window)

        x_a_hourly_gain = np.multiply(x_a_hourly_mean, np.divide(np.square(x_b_hourly_std),
                                                                 np.square(x_a_hourly_std) + np.square(x_b_hourly_std)))
        x_b_hourly_gain = np.multiply(x_b_hourly_mean, np.divide(np.square(x_a_hourly_std),
                                                                 np.square(x_a_hourly_std) + np.square(x_b_hourly_std)))
        self.signal_hourly_out = x_a_hourly_gain + x_b_hourly_gain

        x_a_daily_mean, x_a_daily_std = resampling_average_std(self.signal_a_nn, self.t_a_nn, self.t_daily_out,
                                                               w=self.moving_average_window)
        x_b_daily_mean, x_b_daily_std = resampling_average_std(self.signal_b_nn, self.t_b_nn, self.t_daily_out,
                                                               w=self.moving_average_window)

        x_a_daily_gain = np.multiply(x_a_daily_mean, np.divide(np.square(x_b_daily_std),
                                                               np.square(x_a_daily_std) + np.square(x_b_daily_std)))
        x_b_daily_gain = np.multiply(x_b_daily_mean, np.divide(np.square(x_a_daily_std),
                                                               np.square(x_a_daily_std) + np.square(x_b_daily_std)))
        self.signal_daily_out = x_a_daily_gain + x_b_daily_gain

    def get_result(self):
        signal_a_nn_corrected = self.signal_a_nn / self.degradation_a
        signal_b_nn_corrected = self.signal_b_nn / self.degradation_b

        # Store all values in ModelingResult object
        result = ModelingResult()
        result.t_mutual_nn = self.t_mutual_nn
        result.history_mutual_nn = self.history_mutual_nn
        result.t_a_nn = self.t_a_nn
        result.t_b_nn = self.t_b_nn
        result.signal_a_nn = self.signal_a_nn
        result.signal_b_nn = self.signal_b_nn
        result.signal_a_nn_corrected = signal_a_nn_corrected
        result.signal_b_nn_corrected = signal_b_nn_corrected
        result.t_hourly_out = self.t_hourly_out
        result.signal_hourly_out = self.signal_hourly_out
        result.t_daily_out = self.t_daily_out
        result.signal_daily_out = self.signal_daily_out

        return result

    def _filter_outliers(self, data, signal_a_field_name, signal_b_field_name):
        data = data.copy()

        x_a = data[signal_a_field_name].values
        x_b = data[signal_b_field_name].values

        outliers_a = notnan_indices(x_a)
        outliers_a[outliers_a] = detect_outliers(x_a[notnan_indices(x_a)], None,
                                                 outlier_fraction=self.outlier_fraction)

        outliers_b = notnan_indices(x_b)
        outliers_b[outliers_b] = detect_outliers(x_b[notnan_indices(x_b)], None,
                                                 outlier_fraction=self.outlier_fraction)

        print("{}: {} outliers".format(signal_a_field_name, outliers_a.sum()))
        print("{}: {} outliers".format(signal_b_field_name, outliers_b.sum()))

        x_a_outliers = x_a.copy()
        x_b_outliers = x_b.copy()

        x_a[outliers_a] = np.nan
        x_b[outliers_b] = np.nan
        x_a_outliers[~outliers_a] = np.nan
        x_b_outliers[~outliers_b] = np.nan

        data[signal_a_field_name] = x_a
        data[signal_b_field_name] = x_b

        return data

    @staticmethod
    def _compute_exposure(x, mode=ExposureMode.NUM_MEASUREMENTS, mean=1.0):
        if mode == ExposureMode.NUM_MEASUREMENTS:
            x = np.nan_to_num(x) > 0
            x = x / x.shape[0]
            return np.cumsum(x)
        elif mode == ExposureMode.EXPOSURE_SUM:
            x = np.nan_to_num(x)
            x = x / mean
            x = x / x.shape[0]
            return np.cumsum(x)

    def _iterative_correction(self, signal_a, signal_b, exposure_a, exposure_b, eps=1e-5, max_iter=100):
        x_a_corrected = signal_a
        x_b_corrected = signal_b

        delta_norm = None
        parameters_opt = None
        history = [IterationResult(x_a_corrected, x_b_corrected, np.divide(x_a_corrected, x_b_corrected))]

        step = 1
        while (not delta_norm or delta_norm > eps) and step <= max_iter:
            step += 1
            ratio_a_b_corrected = signal_a / x_b_corrected

            x_a_previous = x_a_corrected
            x_b_previous = x_b_corrected
            x_a_corrected, x_b_corrected, parameters_opt = self._fit_and_correct(signal_a, signal_b, exposure_a,
                                                                                 exposure_b,
                                                                                 ratio_a_b_corrected)

            history.append(IterationResult(x_a_corrected, x_b_corrected, np.divide(x_a_corrected, x_b_corrected)))

            delta_norm_a = np.linalg.norm(x_a_corrected - x_a_previous) / np.linalg.norm(x_a_previous)
            delta_norm_b = np.linalg.norm(x_b_corrected - x_b_previous) / np.linalg.norm(x_b_previous)
            delta_norm = delta_norm_a + delta_norm_b

            print("norm\t", delta_norm)

        return history, parameters_opt

    @abstractmethod
    def _fit_and_correct(self, x_a, x_b, exposure_a, exposure_b, ratio_a_b):
        """

        :param x_a:
        :param x_b:
        :param exposure_a:
        :param exposure_b:
        :param ratio_a_b:
        :return: x_a_corrected, x_b_corrected, parameters_opt
        """
        pass


class ExpFamilyModel(BaseModel):
    def __init__(self, data, timestamp_field_name, signal_a_field_name, signal_b_field_name, exposure_mode,
                 moving_average_window, outlier_fraction):
        # Prepare needed data
        super().__init__(data, timestamp_field_name, signal_a_field_name, signal_b_field_name, exposure_mode,
                         moving_average_window, outlier_fraction)

    @staticmethod
    def _exp_unc(x, gamma, lambda_, e_0):
        """Unconstrained exponential-linear degradation model."""
        return np.exp(-lambda_ * (x - e_0)) + gamma

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

        return lambda_, e_0

    @abstractmethod
    def _fit_and_correct(self, x_a, x_b, exposure_a, exposure_b, ratio_a_b):
        pass


class ExpModel(ExpFamilyModel):
    def __init__(self, data, timestamp_field_name, signal_a_field_name, signal_b_field_name, exposure_mode,
                 moving_average_window, outlier_fraction):
        # Prepare needed data
        super().__init__(data, timestamp_field_name, signal_a_field_name, signal_b_field_name, exposure_mode,
                         moving_average_window, outlier_fraction)

        # Prepare initial parameters
        lambda_initial, e_0_initial = self._initial_fit(self.ratio_a_b_mutual_nn, self.exposure_a_mutual_nn)
        self.parameters_initial = [lambda_initial, e_0_initial]

        # Run optimization
        self._compute_corrections()
        self._compute_output()

        # Compute final signal result for all not-nan values (non-mutual)
        self.degradation_a = self._exp(self.exposure_a_nn, *self.parameters_opt)
        self.degradation_b = self._exp(self.exposure_b_nn, *self.parameters_opt)

    @staticmethod
    def _exp(x, lambda_, e_0):
        """Constrained exponential degradation model: y(0) = 1."""
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0))
        return y

    def _fit_and_correct(self, x_a, x_b, exposure_a, exposure_b, ratio_a_b):
        parameters_opt, _ = curve_fit(self._exp, exposure_a, ratio_a_b, p0=self.parameters_initial, maxfev=100000)
        x_a_corrected = x_a / self._exp(exposure_a, *parameters_opt)
        x_b_corrected = x_b / self._exp(exposure_b, *parameters_opt)

        return x_a_corrected, x_b_corrected, parameters_opt


class ExpLinModel(ExpFamilyModel):
    def __init__(self, data, timestamp_field_name, signal_a_field_name, signal_b_field_name, exposure_mode,
                 moving_average_window, outlier_fraction):
        # Prepare needed data
        super().__init__(data, timestamp_field_name, signal_a_field_name, signal_b_field_name, exposure_mode,
                         moving_average_window, outlier_fraction)

        # Prepare initial parameters
        lambda_initial, e_0_initial = self._initial_fit(self.ratio_a_b_mutual_nn, self.exposure_a_mutual_nn)
        self.parameters_initial = [lambda_initial, e_0_initial, 0]

        # Run optimization
        self._compute_corrections()
        self._compute_output()

        # Compute final signal result for all not-nan values (non-mutual)
        self.degradation_a = self._exp_lin(self.exposure_a_nn, *self.parameters_opt)
        self.degradation_b = self._exp_lin(self.exposure_b_nn, *self.parameters_opt)

    @staticmethod
    def _exp_lin(x, lambda_, e_0, linear):
        """Constrained exponential-linear degradation model: y(0) = 1."""
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0)) + linear * x
        return y

    def _fit_and_correct(self, x_a, x_b, exposure_a, exposure_b, ratio_a_b):
        print(self.parameters_initial)
        parameters_opt, _ = curve_fit(self._exp_lin, exposure_a, ratio_a_b, p0=self.parameters_initial, maxfev=100000)
        x_a_corrected = x_a / self._exp_lin(exposure_a, *parameters_opt)
        x_b_corrected = x_b / self._exp_lin(exposure_b, *parameters_opt)

        return x_a_corrected, x_b_corrected, parameters_opt
