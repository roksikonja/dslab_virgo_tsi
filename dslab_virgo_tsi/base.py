import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Tuple

import numpy as np

from dslab_virgo_tsi.data_utils import notnan_indices, detect_outliers
from dslab_virgo_tsi.model_constants import OutputTimeConstants as OutTimeConsts
from dslab_virgo_tsi.status_utils import status


class Mode(Enum):
    VIRGO = auto()
    GENERATOR = auto()


class ExposureMethod(Enum):
    NUM_MEASUREMENTS = "Num. measurements"
    EXPOSURE_SUM = "Exposure sum"


class CorrectionMethod(Enum):
    CORRECT_BOTH = "Correct both"
    CORRECT_ONE = "Correct one"


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


class OutParams:
    def __init__(self, svgp_iter_loglikelihood=None, svgp_inducing_points=None, svgp_prior_samples=None,
                 svgp_t_prior=None, svgp_posterior_samples=None, svgp_t_posterior=None):
        self.svgp_iter_loglikelihood = svgp_iter_loglikelihood
        self.svgp_inducing_points = svgp_inducing_points
        self.svgp_prior_samples = svgp_prior_samples
        self.svgp_t_prior = svgp_t_prior
        self.svgp_posterior_samples = svgp_posterior_samples
        self.svgp_t_posterior = svgp_t_posterior


class OutResult:
    def __init__(self, t_hourly_out, signal_hourly_out, signal_std_hourly_out, t_daily_out, signal_daily_out,
                 signal_std_daily_out, params_out: OutParams):
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
        self.params_out = params_out


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


class BaseOutputModel(ABC):
    @abstractmethod
    def fit_and_predict(self, mode: Mode, base_signals: BaseSignals, final_result: FinalResult, t_hourly_out):
        pass


class ModelFitter:
    """
    mutual_nn -> values taken at times when all signals are not nan
    nn -> values taken at times when specific signal is not nan
    """

    def __init__(self, data, a_field_name, b_field_name, t_field_name, exposure_method, outlier_fraction=0):
        # Compute all signals and store all relevant to BaseSignals object
        a, b, t = data[a_field_name].values, data[b_field_name].values, data[t_field_name].values

        # Filter outliers
        if outlier_fraction > 0:
            data = self._filter_outliers(data, a_field_name, b_field_name, outlier_fraction)

        # Calculate exposure
        b_mean = float(np.mean(b[notnan_indices(b)]))
        a_length = a.shape[0]
        exposure_a = self._compute_exposure(a, exposure_method, b_mean, a_length)
        exposure_b = self._compute_exposure(b, exposure_method, b_mean, a_length)
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
        logging.info("Mutual signals extracted.")

        # Create BaseSignals instance
        self.base_signals = BaseSignals(a_nn, b_nn, t_a_nn, t_b_nn, exposure_a_nn, exposure_b_nn, a_mutual_nn,
                                        b_mutual_nn, t_mutual_nn, exposure_a_mutual_nn, exposure_b_mutual_nn)

    def __call__(self, model: BaseModel, output_model: BaseOutputModel, correction_method: CorrectionMethod,
                 mode: Mode, compute_output=True) -> Result:

        # Perform initial fit if needed
        initial_params: Params = model.get_initial_params(self.base_signals)

        # Compute iterative corrections
        status.update_progress("Performing iterative corrections", 15)
        history_mutual_nn, optimal_params = self._iterative_correction(model, self.base_signals, initial_params,
                                                                       correction_method)

        # Compute final result
        final_result = model.compute_final_result(self.base_signals, optimal_params)

        # Compute output signals
        if compute_output:
            out_result = self._compute_output(mode, output_model, self.base_signals, final_result)
        else:
            # Only in model comparison mode
            out_result = None

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

        logging.info(f"{a_field_name}: {a_outlier_indices.sum()} outliers")
        logging.info(f"{b_field_name}: {b_outlier_indices.sum()} outliers")
        logging.info("Outliers filtered.")

        a[a_outlier_indices] = np.nan
        b[b_outlier_indices] = np.nan

        data[a_field_name] = a
        data[b_field_name] = b

        return data

    def _compute_output(self, mode: Mode, output_model: BaseOutputModel, base_signals: BaseSignals,
                        final_result: FinalResult) -> OutResult:
        logging.info("Computing output ...")

        # Output resampling
        t_hourly_out, num_hours_in_day = self._output_resampling(mode, base_signals)

        logging.info(f"Data shapes are t_hourly {t_hourly_out.shape}.")

        # Training samples
        signal_hourly_out, signal_std_hourly_out, params_out = output_model.fit_and_predict(mode, base_signals,
                                                                                            final_result,
                                                                                            t_hourly_out)

        logging.info(f"Data shapes are t_hourly {t_hourly_out.shape}, signal_hourly_out {signal_hourly_out.shape} and "
                     f"signal_std_hourly_out {signal_std_hourly_out.shape}.")

        # Daily values
        t_daily_out = t_hourly_out[::num_hours_in_day, :]
        signal_daily_out = signal_hourly_out[::num_hours_in_day]
        signal_std_daily_out = signal_std_hourly_out[::num_hours_in_day]

        return OutResult(t_hourly_out.ravel(), signal_hourly_out.ravel(), signal_std_hourly_out.ravel(),
                         t_daily_out.ravel(), signal_daily_out.ravel(), signal_std_daily_out.ravel(),
                         params_out)

    @staticmethod
    def _compute_exposure(x, mode=ExposureMethod.NUM_MEASUREMENTS, mean=1.0, length=None):
        if mode == ExposureMethod.NUM_MEASUREMENTS:
            x = np.nan_to_num(x) > 0
        elif mode == ExposureMethod.EXPOSURE_SUM:
            x = np.nan_to_num(x)
            x = x / mean

        if length:
            x = x / length

        logging.info("Exposure computed.")
        return np.cumsum(x)

    @staticmethod
    def _iterative_correction(model: BaseModel, base_signals: BaseSignals, initial_params: Params,
                              method: CorrectionMethod, eps=1e-7, max_iter=100) -> Tuple[
         List[FitResult], Params]:
        """Note that we here deal only with mutual_nn data. Variable ratio_a_b_mutual_nn_corrected has different
        definitions based on the correction method used."""
        logging.info("Iterative correction started.")
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
            logging.info("Step:\t{:<30}Delta norm:\t{:>10}".format(step, delta_norm))

        if method == CorrectionMethod.CORRECT_BOTH:
            method = CorrectionMethod.CORRECT_ONE
            ratio_final = np.divide(base_signals.a_mutual_nn, fit_result.b_mutual_nn_corrected)
            fit_result_final = FitResult(base_signals.a_mutual_nn, fit_result.b_mutual_nn_corrected, ratio_final)
            _, current_params = model.fit_and_correct(base_signals, fit_result_final, initial_params, method)
            logging.info("Final fit.")

        return history, current_params

    @staticmethod
    def _output_resampling(mode, base_signals):
        if mode == Mode.GENERATOR:
            min_time = 0
            max_time = np.minimum(base_signals.t_a_nn.max(), base_signals.t_b_nn.max())
            num_hours = int(OutTimeConsts.GEN_NUM_HOURS + 1)
            num_hours_per_day = OutTimeConsts.GEN_NUM_HOURS_PER_DAY
        else:
            min_time = 0
            max_time = np.minimum(np.floor(base_signals.t_a_nn.max()), np.floor(base_signals.t_b_nn.max()))
            num_hours = int(OutTimeConsts.VIRGO_NUM_HOURS_PER_UNIT * (max_time - min_time) + 1)
            num_hours_per_day = OutTimeConsts.VIRGO_NUM_HOURS_PER_DAY

        t_hourly_out = np.linspace(min_time, max_time, num_hours).reshape(-1, 1)

        return t_hourly_out, num_hours_per_day
