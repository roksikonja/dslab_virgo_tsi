import logging
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Tuple

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import notnan_indices, detect_outliers, median_downsample_by_factor, get_summary, \
    normalize, unnormalize, find_nearest
from dslab_virgo_tsi.gpflow_utils import SVGaussianProcess, VirgoWhiteKernel, VirgoMatern12Kernel
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts
from dslab_virgo_tsi.model_constants import OutputTimeConstants as OutTimeConsts


class Mode(Enum):
    VIRGO = auto()
    GENERATOR = auto()


class ExposureMethod(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


class CorrectionMethod(Enum):
    CORRECT_BOTH = auto()
    CORRECT_ONE = auto()


class OutputMethod(Enum):
    SVGP = auto()
    GP = auto()


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


class Kernels:
    matern_kernel = Matern(length_scale=GPConsts.MATERN_LENGTH_SCALE,
                           length_scale_bounds=GPConsts.MATERN_LENGTH_SCALE_BOUNDS,
                           nu=GPConsts.MATERN_NU)

    white_kernel = WhiteKernel(noise_level=GPConsts.WHITE_NOISE_LEVEL,
                               noise_level_bounds=GPConsts.WHITE_NOISE_LEVEL_BOUNDS)

    gpf_matern32 = gpflow.kernels.Matern32()
    gpf_matern12 = gpflow.kernels.Matern12()
    gpf_linear = gpflow.kernels.Linear()
    gpf_white = gpflow.kernels.White()

    gpf_dual_matern12 = VirgoMatern12Kernel()
    gpf_dual_white = VirgoWhiteKernel(label_a=GPConsts.LABEL_A,
                                      label_b=GPConsts.LABEL_B)


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

    def __init__(self, mode: Mode, data, a_field_name, b_field_name, t_field_name, exposure_method, outlier_fraction=0):
        self.mode = mode

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

    def __call__(self, model: BaseModel, correction_method: CorrectionMethod, output_method: OutputMethod,
                 compute_output=True) -> Result:

        # Perform initial fit if needed
        initial_params: Params = model.get_initial_params(self.base_signals)

        # Compute iterative corrections
        history_mutual_nn, optimal_params = self._iterative_correction(model, self.base_signals, initial_params,
                                                                       correction_method)

        # Compute final result
        final_result = model.compute_final_result(self.base_signals, optimal_params)

        # Compute output signals
        if compute_output:
            out_result = self._compute_output(self.base_signals, final_result, output_method)
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

    def _compute_output(self, base_signals: BaseSignals, final_result: FinalResult,
                        output_method: OutputMethod) -> OutResult:
        logging.info("Computing output ...")

        # Data standardization parameters
        x_mean = np.mean(np.concatenate((final_result.a_nn_corrected, final_result.b_nn_corrected), axis=0))
        x_std = np.std(np.concatenate((final_result.a_nn_corrected, final_result.b_nn_corrected), axis=0))
        t_mean = np.mean(np.concatenate((base_signals.t_a_nn, base_signals.t_b_nn), axis=0))
        t_std = np.std(np.concatenate((base_signals.t_a_nn, base_signals.t_b_nn), axis=0))

        # Output resampling
        t_hourly_out, num_hours_in_day = self._output_resampling(self.mode, base_signals)

        # Initial fit samples
        # TODO: Add dual kernels for scikit-learn
        t_initial, x_initial = self._gp_samples(base_signals, final_result, GPConsts.DUAL_KERNEL)
        t_initial = t_initial[:, 0].reshape(-1, 1)  # Remove labels

        # Initial fit kernel
        # TODO: Add dual kernels for scikit-learn
        if False:
            kernel = Kernels.matern_kernel + Kernels.white_kernel
        else:
            kernel = Kernels.matern_kernel + Kernels.white_kernel

        # Training samples
        # TODO: RESAMPLING OF BOTH SIGNALS
        t_a_downsampled, a_downsampled = base_signals.t_a_nn, final_result.a_nn_corrected
        t_b_downsampled, b_downsampled = base_signals.t_b_nn, final_result.b_nn_corrected

        subsampling_rate_a = a_downsampled.shape[0] // b_downsampled.shape[0]
        subsampling_rate_a *= 1000

        t_a_downsampled, a_downsampled = t_a_downsampled[::subsampling_rate_a], a_downsampled[::subsampling_rate_a]

        x = np.concatenate([a_downsampled, b_downsampled], axis=0).reshape(-1, 1)
        t = np.concatenate([t_a_downsampled, t_b_downsampled], axis=0).reshape(-1, 1)

        # Add labels
        if GPConsts.DUAL_KERNEL:
            # Training
            labels = np.concatenate([GPConsts.LABEL_A * np.ones(shape=a_downsampled.shape),
                                     GPConsts.LABEL_B * np.ones(shape=b_downsampled.shape)])
            labels = labels.astype(np.int).reshape(-1, 1)

            t = np.concatenate([t, labels], axis=1).reshape(-1, 2)

            # Output
            labels_out = GPConsts.LABEL_OUT * np.ones(shape=t_hourly_out.shape)
            labels_out = labels_out.astype(np.int).reshape(-1, 1)
            t_hourly_out = np.concatenate([t_hourly_out, labels_out], axis=1).reshape(-1, 2)

            logging.info("GP using dual kernel.")

        # Normalize data
        if GPConsts.NORMALIZE:
            t, x = normalize(t, t_mean, t_std), normalize(x, x_mean, x_std)
            t_initial, x_initial = normalize(t_initial, t_mean, t_std), normalize(x_initial, x_mean, x_std)
            t_hourly_out = normalize(t_hourly_out, t_mean, t_std)
        else:
            x = x - x_mean

        logging.info(f"Dual kernel is set to {GPConsts.DUAL_KERNEL}. Data shapes are t {t.shape}, "
                     f"x {x.shape} and t_hourly {t_hourly_out.shape} with t_inital {t_initial.shape} and "
                     f"x_initial {x_initial.shape}")

        if output_method == OutputMethod.GP:
            logging.info(f"{output_method} started.")

            t, x = t_initial, x_initial
            gpr = self._gaussian_process(kernel, t, x)

            # Prediction
            # TODO: Add for dual kernels, remove [:, 0]
            signal_hourly_out, signal_std_hourly_out = gpr.predict(t_hourly_out, return_std=True)
        else:
            logging.info(f"{output_method} started.")

            length_scale_initial = None
            if self.mode == Mode.VIRGO and GPConsts.INITIAL_FIT:
                logging.info(f"Running GP initial fit on {t_initial.shape} samples.")
                gpr = self._gaussian_process(kernel, t_initial, x_initial)
                length_scale_initial = gpr.kernel_.get_params()["k1__length_scale"]
                noise_level_initial = gpr.kernel_.get_params()["k2__noise_level"]

            # Induction variables
            t_uniform = np.linspace(np.min(t[:, 0]), np.max(t[:, 0]), GPConsts.NUM_INDUCING_POINTS)
            t_uniform_indices = find_nearest(t[:, 0], t_uniform).astype(int)
            z = t[t_uniform_indices, :].copy()

            if self.mode == Mode.GENERATOR:
                kernel = gpflow.kernels.Sum([Kernels.gpf_matern32, Kernels.gpf_white, Kernels.gpf_linear])
            else:
                if GPConsts.DUAL_KERNEL:
                    kernel = gpflow.kernels.Sum([Kernels.gpf_dual_matern12, Kernels.gpf_dual_white])
                else:
                    kernel = gpflow.kernels.Sum([Kernels.gpf_matern12, Kernels.gpf_white])

            # Model
            m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), z, num_data=t.shape[0])

            # Initial guess
            if length_scale_initial:
                m.kernel.kernels[0].lengthscale.assign(length_scale_initial)
                if GPConsts.DUAL_KERNEL:
                    m.kernel.kernels[1].variance_a.assign(noise_level_initial)
                    m.kernel.kernels[1].variance_b.assign(noise_level_initial)
                else:
                    m.kernel.kernels[1].variance.assign(noise_level_initial)

            # Non-trainable parameters
            gpflow.utilities.set_trainable(m.inducing_variable, False)
            gpflow.utilities.set_trainable(m.kernel.kernels[0].variance, False)

            logging.info("Model created.\n\n" + str(get_summary(m)) + "\n")

            # Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((t, x)).repeat().shuffle(buffer_size=t.shape[0],
                                                                                        seed=Const.RANDOM_SEED)
            # Training
            start = time.time()
            maxiter = ci_niter(GPConsts.MAX_ITERATIONS)

            SVGaussianProcess.run_adam(model=m,
                                       iterations=maxiter,
                                       train_dataset=train_dataset,
                                       minibatch_size=GPConsts.MINIBATCH_SIZE)

            end = time.time()
            logging.info("Training finished after: {:>10} sec".format(end - start))
            logging.info("Trained model.\n\n" + str(get_summary(m)) + "\n")

            # Prediction
            signal_hourly_out, signal_var_hourly_out = m.predict_y(t_hourly_out)
            signal_std_hourly_out = tf.sqrt(signal_var_hourly_out)
            signal_hourly_out = tf.reshape(signal_hourly_out, [-1]).numpy()
            signal_std_hourly_out = tf.reshape(signal_std_hourly_out, [-1]).numpy()

        # Compute final output
        if GPConsts.NORMALIZE:
            signal_hourly_out = unnormalize(signal_hourly_out, x_mean, x_std)
            signal_std_hourly_out = signal_std_hourly_out * (x_std ** 2)
            t_hourly_out = unnormalize(t_hourly_out, t_mean, t_std)
        else:
            signal_hourly_out = signal_hourly_out + x_mean
            signal_hourly_out = signal_hourly_out

        signal_hourly_out = signal_hourly_out.reshape(-1, 1)
        signal_std_hourly_out = signal_std_hourly_out.reshape(-1, 1)

        if GPConsts.DUAL_KERNEL:
            t_hourly_out = t_hourly_out[:, 0].reshape(-1, 1)

        logging.info(f"Dual kernel is set to {GPConsts.DUAL_KERNEL}. Data shapes are t {t.shape}, "
                     f"x {x.shape} and t_hourly {t_hourly_out.shape} with t_inital {t_initial.shape} and "
                     f"x_initial {x_initial.shape}. signal_hourly_out {signal_hourly_out.shape} and "
                     f"signal_std_hourly_out {signal_std_hourly_out.shape}.")
        # Daily values
        t_daily_out = t_hourly_out[::num_hours_in_day, :]
        signal_daily_out = signal_hourly_out[::num_hours_in_day]
        signal_std_daily_out = signal_std_hourly_out[::num_hours_in_day]

        return OutResult(t_hourly_out.ravel(), signal_hourly_out.ravel(), signal_std_hourly_out.ravel(),
                         t_daily_out.ravel(), signal_daily_out.ravel(), signal_std_daily_out.ravel())

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
    def _gaussian_process(kernel, t, x):

        gpr = GaussianProcessRegressor(kernel=kernel,
                                       random_state=Const.RANDOM_SEED,
                                       n_restarts_optimizer=GPConsts.N_RESTARTS_OPTIMIZER)

        logging.info("GP data samples: {:>10}".format(t.shape[0]))
        logging.info("GP initial parameters:\t{:>10}".format(str(gpr.kernel)))
        gpr.fit(t, x)
        logging.info("GP log marginal likelihood:\t{:>10}".format(gpr.log_marginal_likelihood()))
        logging.info("GP parameters:\t{:>10}".format(str(gpr.kernel_)))

        return gpr

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

    @staticmethod
    def _gp_samples(base_signals, final_result, dual_kernel):
        t_a_downsampled = median_downsample_by_factor(base_signals.t_a_nn,
                                                      GPConsts.DOWNSAMPLING_FACTOR_A).reshape(-1, 1)
        t_b_downsampled = median_downsample_by_factor(base_signals.t_b_nn,
                                                      GPConsts.DOWNSAMPLING_FACTOR_B).reshape(-1, 1)
        a_downsampled = median_downsample_by_factor(final_result.a_nn_corrected,
                                                    GPConsts.DOWNSAMPLING_FACTOR_A).reshape(-1, 1)
        b_downsampled = median_downsample_by_factor(final_result.b_nn_corrected,
                                                    GPConsts.DOWNSAMPLING_FACTOR_B).reshape(-1, 1)

        x = np.concatenate([a_downsampled, b_downsampled], axis=0).reshape(-1, 1)
        t = np.concatenate([t_a_downsampled, t_b_downsampled], axis=0)

        if dual_kernel:
            labels = np.concatenate([GPConsts.LABEL_A * np.ones(shape=a_downsampled.shape),
                                     GPConsts.LABEL_B * np.ones(shape=b_downsampled.shape)])
            labels = labels.astype(np.int).reshape(-1, 1)

            t = np.concatenate([t, labels], axis=1).reshape(-1, 2)

        return t, x
