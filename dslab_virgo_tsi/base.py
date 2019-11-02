from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Tuple

import numpy as np
from pykalman import KalmanFilter
from scipy.interpolate import interp1d

from dslab_virgo_tsi.data_utils import resampling_average_std, notnan_indices, detect_outliers


class ExposureMode(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


class CorrectionMethod(Enum):
    CORRECT_BOTH = auto()
    CORRECT_ONE = auto()


class BaseSignals:
    def __init__(self, a_nn, b_nn, t_a_nn, t_b_nn, exposure_a_nn, exposure_b_nn, a_mutual_nn, b_mutual_nn, t_mutual_nn,
                 exposure_a_mutual_nn, exposure_b_mutual_nn):
        """

        Parameters
        ----------
        a_nn : array_like
            Signal a with non nan values.
        b_nn : array_like
            Signal a with non nan values.
        t_a_nn : array_like
            Time at which sensor a has non nan value.
        t_b_nn : array_like
            Time at which sensor b has non nan value.
        exposure_a_nn : array_like
            Exposure at which sensor b has non nan value.
        exposure_b_nn : array_like
            Exposure at which sensor b has non nan value.
        a_mutual_nn : array_like
            Signal a where neither a nor b signal have nan values.
        b_mutual_nn : array_like
            Signal b where neither a nor b signal have nan values.
        t_mutual_nn : array_like
            Time where neither a nor b signal have nan values.
        exposure_a_mutual_nn : array_like
            Exposure a where neither a nor b signal have nan values.
        exposure_b_mutual_nn : array_like
            Exposure b where neither a nor b signal have nan values.
        """
        self.a_nn, self.b_nn = a_nn, b_nn
        self.t_a_nn, self.t_b_nn = t_a_nn, t_b_nn
        self.exposure_a_nn, self.exposure_b_nn = exposure_a_nn, exposure_b_nn
        self.a_mutual_nn, self.b_mutual_nn = a_mutual_nn, b_mutual_nn
        self.t_mutual_nn = t_mutual_nn
        self.exposure_a_mutual_nn, self.exposure_b_mutual_nn = exposure_a_mutual_nn, exposure_b_mutual_nn


class OutResult:
    def __init__(self, t_hourly_out, signal_hourly_out, signal_std_hourly_out, t_daily_out, signal_daily_out,
                 signal_std_daily_out):
        """

        Parameters
        ----------
        t_hourly_out : array_like
            Times by hour.
        signal_hourly_out : array_like
            Final signal sampled every hour.
        signal_std_hourly_out : array_like
            Standard deviation of final signal sampled every hour.
        t_daily_out : array_like
            Times by day.
        signal_daily_out : array_like
            Final signal sampled every day.
        signal_std_daily_out : array_like
            Standard deviation of final signal sampled every day.
        """
        self.t_hourly_out = t_hourly_out
        self.signal_hourly_out = signal_hourly_out
        self.signal_std_hourly_out = signal_std_hourly_out
        self.t_daily_out = t_daily_out
        self.signal_daily_out = signal_daily_out
        self.signal_std_daily_out = signal_std_daily_out


class Params:
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args : List
            List of parameters.
        kwargs : Dict
            Dictionary of parameters.
        """
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.args) + "\t" + str(self.kwargs)


class Corrections:
    def __init__(self, a_correction, b_correction):
        """

        Parameters
        ----------
        a_correction : array_like
            Corrected signal a.
        b_correction : array_like
            Corrected signal b.
        """
        self.a_correction = a_correction
        self.b_correction = b_correction


class FitResult:
    def __init__(self, a_mutual_nn_corrected, b_mutual_nn_corrected, ratio_a_b_mutual_nn_corrected):
        self.a_mutual_nn_corrected, self.b_mutual_nn_corrected = a_mutual_nn_corrected, b_mutual_nn_corrected
        self.ratio_a_b_mutual_nn_corrected = ratio_a_b_mutual_nn_corrected


class FinalResult:
    def __init__(self, base_signals: BaseSignals, degradation_a_nn, degradation_b_nn):
        self.degradation_a_nn, self.degradation_b_nn = degradation_a_nn, degradation_b_nn
        self.a_nn_corrected = np.divide(base_signals.a_nn, degradation_a_nn)
        self.b_nn_corrected = np.divide(base_signals.b_nn, degradation_b_nn)


class Result:
    def __init__(self, base_signals: BaseSignals, history_mutual_nn: List[FitResult], final: FinalResult,
                 out: OutResult):
        self.base_signals = base_signals
        self.history_mutual_nn = history_mutual_nn
        self.final = final
        self.out = out


class BaseModel(ABC):
    @abstractmethod
    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        pass

    @abstractmethod
    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        """

        Parameters
        ----------
        base_signals
        fit_result
        initial_params
        method

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

    def __init__(self, mode, data, a_field_name, b_field_name, t_field_name, exposure_mode, outlier_fraction=0):
        self.mode = mode

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

    def __call__(self, model: BaseModel, correction_method: CorrectionMethod, ratio_smoothing) -> Result:
        # Perform initial fit if needed
        initial_params: Params = model.get_initial_params(self.base_signals)

        # Compute iterative corrections
        history_mutual_nn, optimal_params = self._iterative_correction(model, self.base_signals, initial_params,
                                                                       correction_method, ratio_smoothing)

        # Compute final result
        final_result = model.compute_final_result(self.base_signals, optimal_params)

        # Compute output signals
        out_result = self._compute_output(self.base_signals, final_result)

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

        a[a_outlier_indices] = np.nan
        b[b_outlier_indices] = np.nan

        data[a_field_name] = a
        data[b_field_name] = b

        return data

    def _compute_output(self, base_signals: BaseSignals, final_result: FinalResult) -> OutResult:
        print("Compute output")

        num_hours, num_days = None, None
        min_time, max_time = None, None
        if self.mode == "virgo":
            min_time = np.maximum(np.ceil(base_signals.t_a_nn.min()), np.ceil(base_signals.t_b_nn.min()))
            max_time = np.minimum(np.floor(base_signals.t_a_nn.max()), np.floor(base_signals.t_b_nn.max()))

            num_days = int((max_time - min_time) + 1)
            num_hours = int(24 * (max_time - min_time) + 1)
        elif self.mode == "generator":
            min_time = np.maximum(base_signals.t_a_nn.min(), base_signals.t_b_nn.min())
            max_time = np.minimum(base_signals.t_a_nn.max(), base_signals.t_b_nn.max())

            num_days = int(10000)
            num_hours = int(10000)

        t_hourly_out = np.linspace(min_time, max_time, num_hours)
        t_daily_out = np.linspace(min_time, max_time, num_days)

        a_nn_interpolation_func = interp1d(base_signals.t_a_nn, final_result.a_nn_corrected, kind="nearest")
        a_nn_hourly_resampled = a_nn_interpolation_func(t_hourly_out)
        a_nn_daily_resampled = a_nn_interpolation_func(t_daily_out)

        b_nn_interpolation_func = interp1d(base_signals.t_b_nn, final_result.b_nn_corrected, kind="linear")
        b_nn_hourly_resampled = b_nn_interpolation_func(t_hourly_out)
        b_nn_daily_resampled = b_nn_interpolation_func(t_daily_out)

        observations_hourly = np.stack((a_nn_hourly_resampled, b_nn_hourly_resampled), axis=1)
        observations_daily = np.stack((a_nn_daily_resampled, b_nn_daily_resampled), axis=1)

        mean = np.mean(np.concatenate((final_result.a_nn_corrected, final_result.b_nn_corrected), axis=0))

        kf = KalmanFilter(n_dim_obs=2,
                          initial_state_mean=0,
                          transition_matrices=[[1]],
                          transition_offsets=0,
                          observation_matrices=[[1], [1]],
                          observation_offsets=[0, 0],
                          em_vars=["transition_covariance",
                                   "observation_covariance",
                                   "initial_state_covariance"])

        print("Running EM for Kalman Filter")
        kf.em(observations_daily - mean)

        signal_hourly_out, signal_std_hourly_out = kf.smooth(observations_hourly - mean)
        signal_daily_out, signal_std_daily_out = kf.smooth(observations_daily - mean)

        return OutResult(t_hourly_out, signal_hourly_out.ravel() + mean, np.sqrt(signal_std_hourly_out.ravel()),
                         t_daily_out, signal_daily_out.ravel() + mean, np.sqrt(signal_std_daily_out.ravel()))

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
                              method: CorrectionMethod, ratio_smoothing, eps=1e-7, max_iter=100) -> Tuple[
        List[FitResult], Params]:
        """Note that we here deal only with mutual_nn data. Variable ratio_a_b_mutual_nn_corrected has different
        definitions based on the correction method used."""
        fit_result = FitResult(np.copy(base_signals.a_mutual_nn), np.copy(base_signals.b_mutual_nn),
                               np.divide(base_signals.a_mutual_nn, base_signals.b_mutual_nn))

        history = [fit_result]

        step = 0
        has_converged = False
        current_params = initial_params
        while not has_converged and step < max_iter:
            step += 1
            fit_result_previous = fit_result

            # Use model for fitting
            corrections, current_params = model.fit_and_correct(base_signals, fit_result_previous, initial_params,
                                                                method)

            if method == CorrectionMethod.CORRECT_BOTH:
                a_mutual_nn_corrected = np.divide(fit_result_previous.a_mutual_nn_corrected, corrections.a_correction)
                b_mutual_nn_corrected = np.divide(fit_result_previous.b_mutual_nn_corrected, corrections.b_correction)
                ratio_a_b_mutual_nn_corrected = np.divide(a_mutual_nn_corrected, b_mutual_nn_corrected)
            else:
                a_mutual_nn_corrected = np.divide(base_signals.a_mutual_nn, corrections.a_correction)
                b_mutual_nn_corrected = np.divide(base_signals.b_mutual_nn, corrections.b_correction)
                ratio_a_b_mutual_nn_corrected = np.divide(base_signals.a_mutual_nn, b_mutual_nn_corrected)

            if ratio_smoothing:
                ratio_a_b_mutual_nn_corrected, _ = resampling_average_std(ratio_a_b_mutual_nn_corrected,
                                                                          w=ratio_a_b_mutual_nn_corrected.shape[
                                                                                0] / 100,
                                                                          std=False)

            # Store current state to history
            fit_result = FitResult(a_mutual_nn_corrected, b_mutual_nn_corrected, ratio_a_b_mutual_nn_corrected)
            history.append(fit_result)

            # Compute delta
            a_previous = fit_result_previous.a_mutual_nn_corrected
            b_previous = fit_result_previous.b_mutual_nn_corrected
            delta_norm_a = np.linalg.norm(a_mutual_nn_corrected - a_previous) / np.linalg.norm(a_previous)
            delta_norm_b = np.linalg.norm(b_mutual_nn_corrected - b_previous) / np.linalg.norm(b_previous)
            delta_norm = delta_norm_a + delta_norm_b

            has_converged = delta_norm < eps

            print("\nstep:\t" + str(step) + "\nnorm:\t", delta_norm, "\nparams:\t", current_params)

        if method == CorrectionMethod.CORRECT_BOTH:
            method = CorrectionMethod.CORRECT_ONE
            ratio_final = np.divide(base_signals.a_mutual_nn, fit_result.b_mutual_nn_corrected)
            fit_result_final = FitResult(base_signals.a_mutual_nn, fit_result.b_mutual_nn_corrected, ratio_final)
            _, current_params = model.fit_and_correct(base_signals, fit_result_final, initial_params, method)

        return history, current_params
