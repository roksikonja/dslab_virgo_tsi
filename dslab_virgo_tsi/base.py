from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from dslab_virgo_tsi.data_utils import resampling_average_std, notnan_indices, detect_outliers, downsample_signal


class ExposureMode(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


class BaseSignals:
    def __init__(self, a_nn, b_nn, t_a_nn, t_b_nn, exposure_a_nn, exposure_b_nn, a_mutual_nn, b_mutual_nn, t_mutual_nn,
                 exposure_a_mutual_nn, exposure_b_mutual_nn):
        self.a_nn, self.b_nn = a_nn, b_nn
        self.t_a_nn, self.t_b_nn = t_a_nn, t_b_nn
        self.exposure_a_nn, self.exposure_b_nn = exposure_a_nn, exposure_b_nn
        self.a_mutual_nn, self.b_mutual_nn = a_mutual_nn, b_mutual_nn
        self.t_mutual_nn = t_mutual_nn
        self.exposure_a_mutual_nn, self.exposure_b_mutual_nn = exposure_a_mutual_nn, exposure_b_mutual_nn

    def downsample_nn_signals(self, k_a, k_b):
        self.t_a_nn, self.t_b_nn = downsample_signal(self.t_a_nn, k_a), downsample_signal(self.t_b_nn, k_b)
        self.a_nn, self.b_nn = downsample_signal(self.a_nn, k_a), downsample_signal(self.b_nn, k_b)
        self.exposure_a_nn = downsample_signal(self.exposure_a_nn, k_a)
        self.exposure_b_nn = downsample_signal(self.exposure_b_nn, k_b)


class OutResult:
    def __init__(self, t_hourly_out, signal_hourly_out, signal_std_hourly_out, t_daily_out, signal_daily_out,
                 signal_std_daily_out):
        self.t_hourly_out = t_hourly_out
        self.signal_hourly_out = signal_hourly_out
        self.signal_std_hourly_out = signal_std_hourly_out
        self.t_daily_out = t_daily_out
        self.signal_daily_out = signal_daily_out
        self.signal_std_daily_out = signal_std_daily_out


class Params:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.args) + "\t" + str(self.kwargs)


class FitResult:
    def __init__(self, a_mutual_nn_corrected, b_mutual_nn_corrected, current_params: Params = None):
        self.a_mutual_nn_corrected, self.b_mutual_nn_corrected = a_mutual_nn_corrected, b_mutual_nn_corrected
        self.ratio_a_b_mutual_nn_corrected = np.divide(a_mutual_nn_corrected, b_mutual_nn_corrected)
        self.current_params = current_params


class FinalResult:
    def __init__(self, base_signals: BaseSignals, degradation_a_nn, degradation_b_nn):
        self.degradation_a_nn, self.degradation_b_nn = degradation_a_nn, degradation_b_nn
        self.a_nn_corrected = np.divide(base_signals.a_nn, degradation_a_nn)
        self.b_nn_corrected = np.divide(base_signals.b_nn, degradation_b_nn)

    def downsample_nn_signals(self, k_a, k_b):
        self.degradation_a_nn = downsample_signal(self.degradation_a_nn, k_a)
        self.degradation_b_nn = downsample_signal(self.degradation_b_nn, k_b)
        self.a_nn_corrected = downsample_signal(self.a_nn_corrected, k_a)
        self.b_nn_corrected = downsample_signal(self.b_nn_corrected, k_b)


class Result:
    def __init__(self, base_signals: BaseSignals, history_mutual_nn: List[FitResult], final: FinalResult,
                 out: OutResult):
        self.base_signals = base_signals
        self.history_mutual_nn = history_mutual_nn
        self.final = final
        self.out = out

    def downsample_nn_signals(self, k_a, k_b):
        self.base_signals.downsample_nn_signals(k_a, k_b)
        self.final.downsample_nn_signals(k_a, k_b)


class BaseModel(ABC):
    @abstractmethod
    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        pass

    @abstractmethod
    def fit_and_correct(self, base_signals: BaseSignals, initial_params: Params, ratio) -> FitResult:
        """

        Parameters
        ----------
        base_signals
        initial_params
        ratio
            Interpretation depends on the chosen iterative algorithm

        Returns
        -------

        """
        pass

    @abstractmethod
    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        pass


class ModelFitter:
    """
    mutual_nn -> values taken at times when all signals are not nan
    nn -> values taken at times when specific signal is not nan
    """

    def __init__(self, data, a_field_name, b_field_name, t_field_name, exposure_mode, outlier_fraction=0):
        # Compute all signals and store all relevant to BaseSignals object
        a, b, t = data[a_field_name].values, data[b_field_name].values, data[t_field_name].values

        # Filter outliers
        if outlier_fraction > 0:
            data = self._filter_outliers(data, a_field_name, b_field_name, outlier_fraction)

        # Calculate exposure
        b_mean = float(np.mean(b[notnan_indices(b)]))
        a_length = a.shape[0]
        exposure_a = self._compute_exposure(a, exposure_mode, b_mean, a_length)
        exposure_b = self._compute_exposure(b, exposure_mode, b_mean, a_length)
        data["e_a"], data["e_b"] = exposure_a, exposure_b

        # Not nan rows (non-mutual)
        index_a_nn, index_b_nn = notnan_indices(a), notnan_indices(b)
        a_nn, b_nn = a[index_a_nn], b[index_b_nn]
        t_a_nn, t_b_nn = t[index_a_nn], t[index_b_nn]
        exposure_a_nn, exposure_b_nn = exposure_a[index_a_nn], exposure_b[index_b_nn]

        # Extract mutual not nan rows
        data_mutual_nn = data[[t_field_name, a_field_name, b_field_name, "e_a", "e_b"]].dropna()
        a_mutual_nn, b_mutual_nn = data_mutual_nn[a_field_name].values, data_mutual_nn[b_field_name].values
        t_mutual_nn = data_mutual_nn[t_field_name].values
        exposure_a_mutual_nn, exposure_b_mutual_nn = data_mutual_nn["e_a"].values, data_mutual_nn["e_b"].values

        # Create BaseSignals instance
        self.base_signals = BaseSignals(a_nn, b_nn, t_a_nn, t_b_nn, exposure_a_nn, exposure_b_nn, a_mutual_nn,
                                        b_mutual_nn, t_mutual_nn, exposure_a_mutual_nn, exposure_b_mutual_nn)

    def __call__(self, model: BaseModel, iterative_correction_model, moving_average_window=81) -> Result:
        # Perform initial fit if needed
        initial_params: Params = model.get_initial_params(self.base_signals)

        # Compute iterative corrections
        history_mutual_nn = self._iterative_correction(model, self.base_signals, initial_params,
                                                       iterative_correction_model)

        # Compute final result
        optimal_params = history_mutual_nn[-1].current_params
        final_result = model.compute_final_result(self.base_signals, optimal_params)

        # Compute output signals
        out_result = self._compute_output(self.base_signals, final_result, moving_average_window)

        # Return all together
        return Result(self.base_signals, history_mutual_nn, final_result, out_result)

    @staticmethod
    def _filter_outliers(data, a_field_name, b_field_name, outlier_fraction):
        data = data.copy()

        a, b = data[a_field_name].values, data[b_field_name].values

        a_outlier_indices = notnan_indices(a)
        a_outlier_indices[a_outlier_indices] = detect_outliers(a[notnan_indices(a)], None, outlier_fraction)

        b_outlier_indices = notnan_indices(b)
        b_outlier_indices[b_outlier_indices] = detect_outliers(b[notnan_indices(b)], None, outlier_fraction)

        print(f"{a_field_name}: {a_outlier_indices.sum()} outliers")
        print(f"{b_field_name}: {b_outlier_indices.sum()} outliers")

        # a_outliers = a.copy()
        # b_outliers = b.copy()

        a[a_outlier_indices] = np.nan
        b[b_outlier_indices] = np.nan

        data[a_field_name] = a
        data[b_field_name] = b

        # a_outliers[~a_outlier_indices] = np.nan
        # b_outliers[~b_outlier_indices] = np.nan

        return data

    def _compute_output(self, base_signals: BaseSignals, final_result: FinalResult, moving_average_window) -> OutResult:
        print("Compute output")
        min_time = np.maximum(np.ceil(base_signals.t_a_nn.min()), np.ceil(base_signals.t_b_nn.min()))
        max_time = np.minimum(np.floor(base_signals.t_a_nn.max()), np.floor(base_signals.t_b_nn.max()))

        t_hourly_out = np.arange(min_time, max_time, 1.0 / 24.0)
        t_daily_out = np.arange(min_time, max_time, 1.0)

        a_nn_interpolation_func = interp1d(base_signals.t_a_nn, final_result.a_nn_corrected, kind="linear")
        a_nn_hourly_resampled = a_nn_interpolation_func(t_hourly_out)
        a_nn_daily_resampled = a_nn_interpolation_func(t_daily_out)

        b_nn_interpolation_func = interp1d(base_signals.t_b_nn, final_result.b_nn_corrected, kind="linear")
        b_nn_hourly_resampled = b_nn_interpolation_func(t_hourly_out)
        b_nn_daily_resampled = b_nn_interpolation_func(t_daily_out)

        signal_hourly_out, signal_std_hourly_out = self._resample_and_compute_gain(
            a_nn_hourly_resampled, b_nn_hourly_resampled, moving_average_window * 24)
        signal_daily_out, signal_std_daily_out = self._resample_and_compute_gain(
            a_nn_daily_resampled, b_nn_daily_resampled, moving_average_window)

        return OutResult(t_hourly_out, signal_hourly_out, signal_std_hourly_out, t_daily_out, signal_daily_out,
                         signal_std_daily_out)

    @staticmethod
    def _resample_and_compute_gain(a, b, moving_average_window):
        a_out_mean, a_out_std = resampling_average_std(a, w=moving_average_window)
        b_out_mean, b_out_std = resampling_average_std(b, w=moving_average_window)

        a_out_std_squared, b_out_std_squared = np.square(a_out_std), np.square(b_out_std)

        a_out_gain = np.multiply(a_out_mean, np.divide(b_out_std_squared, a_out_std_squared + b_out_std_squared))
        b_out_gain = np.multiply(b_out_mean, np.divide(a_out_std_squared, a_out_std_squared + b_out_std_squared))

        return a_out_gain + b_out_gain, np.sqrt(np.divide(np.multiply(a_out_std_squared, b_out_std_squared),
                                                          a_out_std_squared + b_out_std_squared))

    @staticmethod
    def _compute_exposure(x, mode=ExposureMode.NUM_MEASUREMENTS, mean=1.0, length=None):
        if mode == ExposureMode.NUM_MEASUREMENTS:
            x = np.nan_to_num(x) > 0
        elif mode == ExposureMode.EXPOSURE_SUM:
            x = np.nan_to_num(x)
            x = x / mean

        if length:
            x = x / length
        return np.cumsum(x)

    @staticmethod
    def _iterative_correction(model: BaseModel, base_signals: BaseSignals, initial_params: Params,
                              iterative_correction_model, eps=1e-7, max_iter=100) -> List[FitResult]:
        """Note that we here deal only with mutual_nn data."""
        a, b = base_signals.a_mutual_nn, base_signals.b_mutual_nn
        a_corrected, b_corrected = np.copy(a), np.copy(b)

        delta_norm = None
        history = [FitResult(a_corrected, b_corrected, initial_params)]

        step = 0
        while (not delta_norm or delta_norm > eps) and step < max_iter:
            step += 1
            a_previous, b_previous = np.copy(a_corrected), np.copy(b_corrected)

            ratio = None
            if iterative_correction_model == 1:
                ratio = np.divide(a_corrected, b_corrected)
            elif iterative_correction_model == 2:
                ratio = np.divide(a, b_corrected)

            # Use model for fitting and extract results
            fit_result: FitResult = model.fit_and_correct(base_signals, initial_params, ratio)
            a_corrected = fit_result.a_mutual_nn_corrected
            b_corrected = fit_result.b_mutual_nn_corrected
            params = fit_result.current_params

            # Store current state to history
            history.append(fit_result)

            # Compute delta
            delta_norm_a = np.linalg.norm(a_corrected - a_previous) / np.linalg.norm(a_previous)
            delta_norm_b = np.linalg.norm(b_corrected - b_previous) / np.linalg.norm(b_previous)
            delta_norm = delta_norm_a + delta_norm_b

            print("\nstep:\t" + str(step) + "\nnorm:\t", delta_norm, "\nparams:\t", params)

        if iterative_correction_model == 1:
            print("final fit")
            ratio = np.divide(a, b_corrected)
            fit_result: FitResult = model.fit_and_correct(base_signals, initial_params, ratio)
            history.append(fit_result)

        return history
