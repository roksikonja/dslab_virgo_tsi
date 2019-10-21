from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from dslab_virgo_tsi.data_utils import resampling_average_std, downsample_signal, notnan_indices, detect_outliers


class ExposureMode(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


# TODO: filter outliers
class ModelingResult:
    def __init__(self):
        self.t_mutual_nn = None
        self.history_mutual_nn = None
        self.t_a_nn = None
        self.t_b_nn = None
        self.a_nn = None
        self.b_nn = None
        self.a_nn_corrected = None
        self.b_nn_corrected = None
        self.t_hourly_out = None
        self.signal_hourly_out = None
        self.t_daily_out = None
        self.signal_daily_out = None

    def downsample_signals(self, k_a, k_b):
        self.t_a_nn = downsample_signal(self.t_a_nn, k_a)
        self.t_b_nn = downsample_signal(self.t_b_nn, k_b)
        self.a_nn = downsample_signal(self.a_nn, k_a)
        self.b_nn = downsample_signal(self.b_nn, k_b)
        self.a_nn_corrected = downsample_signal(self.a_nn_corrected, k_a)
        self.b_nn_corrected = downsample_signal(self.b_nn_corrected, k_b)


class IterationResult:
    def __init__(self, a, b, ratio_a_b):
        self.a = a
        self.b = b
        self.ratio_a_b = ratio_a_b


class BaseModel(ABC):
    """
    mutual_nn -> values taken at times when all signals are not nan
    nn -> values taken at times when specific signal is not nan
    """

    def __init__(self, data, t_field_name, a_field_name, b_field_name, exposure_mode, moving_average_window,
                 outlier_fraction):
        self.result = ModelingResult()

        a = data[a_field_name].values
        b = data[b_field_name].values
        t = data[t_field_name].values

        # Parameters
        self.moving_average_window = moving_average_window

        # Filter outliers
        if outlier_fraction > 0:
            data = self._filter_outliers(data, a_field_name, b_field_name, outlier_fraction)

        # Calculate exposure
        b_mean = float(np.mean(b[~np.isnan(b)]))
        exposure_a = self._compute_exposure(a, exposure_mode, b_mean)
        exposure_b = self._compute_exposure(b, exposure_mode, b_mean)
        data["e_a"] = exposure_a
        data["e_b"] = exposure_b

        # Not nan rows (non-mutual)
        index_a_nn = notnan_indices(a)
        index_b_nn = notnan_indices(b)
        a_nn = a[index_a_nn]
        b_nn = b[index_b_nn]
        t_a_nn = t[index_a_nn]
        t_b_nn = t[index_b_nn]
        exposure_a_nn = exposure_a[index_a_nn]
        exposure_b_nn = exposure_b[index_b_nn]

        # Extract mutual not nan rows
        data_mutual_nn = data[[t_field_name, a_field_name, b_field_name, "e_a", "e_b"]].dropna()
        t_mutual_nn = data_mutual_nn[t_field_name].values
        exposure_a_mutual_nn = data_mutual_nn["e_a"].values
        exposure_b_mutual_nn = data_mutual_nn["e_b"].values
        a_mutual_nn = data_mutual_nn[a_field_name].values
        b_mutual_nn = data_mutual_nn[b_field_name].values

        # Variable needed for initial fit
        self.ratio_a_b_mutual_nn = np.divide(exposure_a_mutual_nn, exposure_b_mutual_nn)

        # Variables needed later
        self.t_mutual_nn = t_mutual_nn
        self.a_nn = a_nn
        self.b_nn = b_nn
        self.t_a_nn = t_a_nn
        self.t_b_nn = t_b_nn
        self.a_mutual_nn = a_mutual_nn
        self.b_mutual_nn = b_mutual_nn
        self.exposure_a_mutual_nn = exposure_a_mutual_nn
        self.exposure_b_mutual_nn = exposure_b_mutual_nn
        self.exposure_a_nn = exposure_a_nn
        self.exposure_b_nn = exposure_b_nn
        self.degradation_a = None
        self.degradation_b = None

    def _compute_corrections(self):
        print("Compute corrections")
        # Compute corrected signals (whole history)
        self.history_mutual_nn, self.parameters_opt = self._iterative_correction(self.a_mutual_nn,
                                                                                 self.b_mutual_nn,
                                                                                 self.exposure_a_mutual_nn,
                                                                                 self.exposure_b_mutual_nn)

    def _compute_output(self):
        min_time = np.floor(self.t_a_nn.min())
        max_time = np.ceil(self.t_a_nn.max())

        self.t_hourly_out = np.arange(min_time, max_time, 1.0 / 24.0)
        self.t_daily_out = np.arange(min_time, max_time, 1.0)

        self.signal_hourly_out = self._compute_gain(self.a_nn, self.b_nn, self.t_a_nn, self.t_b_nn,
                                                    self.t_hourly_out, self.moving_average_window)
        self.signal_daily_out = self._compute_gain(self.a_nn, self.b_nn, self.t_a_nn, self.t_b_nn,
                                                   self.t_daily_out, self.moving_average_window)

    @staticmethod
    def _compute_gain(a, b, t_a, t_b, t_out, moving_average_window):
        a_out_mean, a_out_std = resampling_average_std(a, t_a, t_out, w=moving_average_window)
        b_out_mean, b_out_std = resampling_average_std(b, t_b, t_out, w=moving_average_window)

        a_out_std_squared = np.square(a_out_std)
        b_out_std_squared = np.square(b_out_std)

        a_out_gain = np.multiply(a_out_mean, np.divide(b_out_std_squared, a_out_std_squared + b_out_std_squared))
        b_out_gain = np.multiply(b_out_mean, np.divide(a_out_std_squared, a_out_std_squared + b_out_std_squared))

        return a_out_gain + b_out_gain

    def get_result(self):
        a_nn_corrected = self.a_nn / self.degradation_a
        b_nn_corrected = self.b_nn / self.degradation_b

        # Store all values in ModelingResult object
        result = ModelingResult()
        result.t_mutual_nn = self.t_mutual_nn
        result.history_mutual_nn = self.history_mutual_nn
        result.t_a_nn = self.t_a_nn
        result.t_b_nn = self.t_b_nn
        result.a_nn = self.a_nn
        result.b_nn = self.b_nn
        result.a_nn_corrected = a_nn_corrected
        result.b_nn_corrected = b_nn_corrected
        # result.t_hourly_out = self.t_hourly_out
        # result.signal_hourly_out = self.signal_hourly_out
        # result.t_daily_out = self.t_daily_out
        # result.signal_daily_out = self.signal_daily_out

        return result

    @staticmethod
    def _filter_outliers(data, a_field_name, b_field_name, outlier_fraction):
        data = data.copy()

        a = data[a_field_name].values
        b = data[b_field_name].values

        a_outlier_indices = notnan_indices(a)
        a_outlier_indices[a_outlier_indices] = detect_outliers(a[notnan_indices(a)], None, outlier_fraction)

        b_outlier_indices = notnan_indices(b)
        b_outlier_indices[b_outlier_indices] = detect_outliers(b[notnan_indices(b)], None, outlier_fraction)

        print("{}: {} outliers".format(a_field_name, a_outlier_indices.sum()))
        print("{}: {} outliers".format(b_field_name, b_outlier_indices.sum()))

        # a_outliers = a.copy()
        # b_outliers = b.copy()

        a[a_outlier_indices] = np.nan
        b[b_outlier_indices] = np.nan

        data[a_field_name] = a
        data[b_field_name] = b

        # a_outliers[~a_outlier_indices] = np.nan
        # b_outliers[~b_outlier_indices] = np.nan

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

    def _iterative_correction(self, a, b, exposure_a, exposure_b, eps=1e-2, max_iter=100):
        a_corrected = a
        b_corrected = b

        delta_norm = None
        parameters_opt = None
        history = [IterationResult(a_corrected, b_corrected, np.divide(a_corrected, b_corrected))]

        step = 0
        while (not delta_norm or delta_norm > eps) and step < max_iter:
            step += 1
            ratio_a_b_corrected = a / b_corrected

            a_previous = a_corrected
            b_previous = b_corrected
            a_corrected, b_corrected, parameters_opt = self._fit_and_correct(a, b, exposure_a, exposure_b,
                                                                             ratio_a_b_corrected)

            history.append(IterationResult(a_corrected, b_corrected, np.divide(a_corrected, b_corrected)))

            delta_norm_a = np.linalg.norm(a_corrected - a_previous) / np.linalg.norm(a_previous)
            delta_norm_b = np.linalg.norm(b_corrected - b_previous) / np.linalg.norm(b_previous)
            delta_norm = delta_norm_a + delta_norm_b

            print("\nstep:\t" + str(step) + "\nnorm:\t", delta_norm, "\nparameters:\t", parameters_opt)

        return history, parameters_opt

    @abstractmethod
    def _fit_and_correct(self, a, b, exposure_a, exposure_b, ratio_a_b):
        """

        :param a:
        :param b:
        :param exposure_a:
        :param exposure_b:
        :param ratio_a_b:
        :return: a_corrected, b_corrected, parameters_opt
        """
        pass


class ExpFamilyModel(BaseModel):
    def __init__(self, data, t_field_name, a_field_name, b_field_name, exposure_mode,
                 moving_average_window, outlier_fraction):
        # Prepare needed data
        super().__init__(data, t_field_name, a_field_name, b_field_name, exposure_mode,
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
    def _fit_and_correct(self, a, b, exposure_a, exposure_b, ratio_a_b):
        pass


class ExpModel(ExpFamilyModel):
    def __init__(self, data, t_field_name, a_field_name, b_field_name, exposure_mode,
                 moving_average_window, outlier_fraction):
        # Prepare needed data
        super().__init__(data, t_field_name, a_field_name, b_field_name, exposure_mode,
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

    def _fit_and_correct(self, a, b, exposure_a, exposure_b, ratio_a_b):
        parameters_opt, _ = curve_fit(self._exp, exposure_a, ratio_a_b, p0=self.parameters_initial, maxfev=100000)
        a_corrected = a / self._exp(exposure_a, *parameters_opt)
        b_corrected = b / self._exp(exposure_b, *parameters_opt)

        return a_corrected, b_corrected, parameters_opt


class ExpLinModel(ExpFamilyModel):
    def __init__(self, data, t_field_name, a_field_name, b_field_name, exposure_mode,
                 moving_average_window, outlier_fraction):
        # Prepare needed data
        super().__init__(data, t_field_name, a_field_name, b_field_name, exposure_mode,
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

    def _fit_and_correct(self, a, b, exposure_a, exposure_b, ratio_a_b):
        parameters_opt, _ = curve_fit(self._exp_lin, exposure_a, ratio_a_b, p0=self.parameters_initial, maxfev=100000)
        a_corrected = a / self._exp_lin(exposure_a, *parameters_opt)
        b_corrected = b / self._exp_lin(exposure_b, *parameters_opt)

        return a_corrected, b_corrected, parameters_opt
