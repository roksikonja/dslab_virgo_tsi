import logging
import time
from typing import List, Tuple

import cvxpy as cp
import gpflow
import numpy as np
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from numba import njit
from scipy.interpolate import UnivariateSpline, splev, splrep, interp1d
from scipy.optimize import curve_fit, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from dslab_virgo_tsi.base import BaseModel, BaseOutputModel, BaseSignals, Params, FinalResult, FitResult, Corrections, \
    CorrectionMethod, Mode, OutParams
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import median_downsample_by_factor, get_summary, \
    normalize, unnormalize, find_nearest, extract_time_window
from dslab_virgo_tsi.gp_utils import SVGaussianProcess
from dslab_virgo_tsi.kernels import Kernels, DualMatern, DualWhiteKernel
from dslab_virgo_tsi.model_constants import EnsembleConstants as EnsConsts
from dslab_virgo_tsi.model_constants import ExpConstants as ExpConsts
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts
from dslab_virgo_tsi.model_constants import IsotonicConstants as IsoConsts
from dslab_virgo_tsi.model_constants import SmoothMonotoneRegressionConstants as SMRConsts
from dslab_virgo_tsi.model_constants import SplineConstants as SplConsts
from dslab_virgo_tsi.status_utils import status


class ExpFamilyMixin:
    @staticmethod
    def _initial_fit(ratio_a_b, exposure_a):
        """

        Parameters
        ----------
        ratio_a_b : array_like
            Ratio of signals a and b.
        exposure_a
            Amount of exposure to the time of ratio measurement.

        Returns
        -------
        lambda_ : float
            Initial guess for scaling parameter in exponential decay.
        e_0 : float
            Initial guess for shift parameter in exponential decay.
        """
        epsilon = 1e-5
        gamma = ratio_a_b.min()

        y = np.log(ratio_a_b - gamma + epsilon)
        x = exposure_a.reshape(-1, 1)

        regression = LinearRegression(fit_intercept=True)
        regression.fit(x, y)

        lambda_ = -regression.coef_[0]
        e_0 = regression.intercept_ / lambda_

        return lambda_, e_0


class DegradationSpline:
    def __init__(self, k, steps, s_max):
        """

        Parameters
        ----------
        k : int
            Degree of polynomials used in spline fitting.
        steps : int
            Number of steps for bisection search for right parameters.
        """
        self.k = k
        self.sp = None
        self.steps = steps
        self.s_max = s_max

    @staticmethod
    def _guess(x, y, k, s, w=None):
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        y : array_like
            Input y coordinate values.
        k : int
            Degree of Spline polynomial.
        s : float (between 0 and 1)
            Sensitivity parameter  (0 is for interpolating, 1 is for maximal smoothing).
        w : array_like, optional
            Predefined weights.

        Returns
        -------
        knots:
            A tuple (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.

        """
        return splrep(x, y, w, k=k, s=s)

    @staticmethod
    def _err(c, x, y, t, k, w=None):
        """

        Parameters
        ----------
        c : array_like
            B-spline coefficients.
        x : array_like
            Input x coordinate values.
        y : array_like
            Input y coordinate values.
        t : array_like
            Vector of knots.
        k : int
            Degree of Spline polynomial.
        w : array_like, optional
            Predefined weights.

        Returns
        -------
        absolute_value: float
            Returns absolute value of error vector.

        """
        """The error function to minimize"""
        diff = y - splev(x, (t, c, k))
        if w is None:
            diff = np.einsum('...i,...i', diff, diff)
        else:
            diff = np.dot(diff * diff, w)
        return np.abs(diff)

    def _spline_dirichlet(self, x, y, k, s, w=None):
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        y : array_like
            Input y coordinate values.
        k : int
            Degree of Spline polynomial.
        s : float (between 0 and 1)
            Sensitivity parameter  (0 is for interpolating, 1 is for maximal smoothing).
        w : array_like, optional
            Predefined weights.

        Returns
        -------
        spline: UnivariateSpline
            Returns Spline which best suits given minimization problem.

        """
        t, c0, k = self._guess(x, y, k, s, w=w)
        con = {'type': 'eq',
               'fun': lambda c: splev(0, (t, c, k), der=0) - 1,
               }
        opt = minimize(self._err, c0, (x, y, t, k, w), constraints=con)
        copt = opt.x
        return UnivariateSpline._from_tck((t, copt, k))

    @staticmethod
    def _is_decreasing(spline, x):
        """

        Parameters
        ----------
        spline: UnivariateSpline
            Univariate spline from SciPy package.
        x : array_like
            Input x coordinates.

        Returns
        -------
        is_decreasing : Bool
            Returns True if spline is decreasing, otherwise False.

        """
        spline_derivative = spline.derivative()
        return np.all(spline_derivative(x.ravel()) <= 0)

    @staticmethod
    def _is_convex(spline, x):
        """

        Parameters
        ----------
        spline: UnivariateSpline
            Univariate spline from SciPy package.
        x : array_like
            Input x coordinates.

        Returns
        -------
        is_decreasing : Bool
            Returns True if spline is convex, otherwise False.

        """
        spline_derivative_2 = spline.derivative().derivative()
        return np.all(spline_derivative_2(x.ravel()) > 0)

    def _find_convex_decreasing_spline_binary_search(self, x, y):
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        y : array_like
            Input y coordinate values.

        Returns
        -------
        spline : UnivariateSpline
            Returns spline with smallest sensitivity s, for which it is still decreasing.
            To find the optimal s we use bisection.

        """
        start = 0
        end = self.s_max
        mid = (end - start) / 2
        spline = self._spline_dirichlet(x.ravel(), y.ravel(), k=self.k, s=mid)
        step = 1
        while step <= self.steps:
            if self._is_decreasing(spline, x):
                end = mid
            else:
                start = mid

            mid = (end + start) / 2
            spline = self._spline_dirichlet(x.ravel(), y.ravel(), k=self.k, s=mid)
            step += 1
        spline = self._spline_dirichlet(x.ravel(), y.ravel(), k=self.k, s=end)
        if not self._is_convex(spline, x):
            logging.warning("Spline is decreasing but not convex.")
        return spline

    def fit(self, x, y):
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        y : array_like
            Input y coordinate values.

        Returns
        -------
        spline : UnivariateSpline
            Returns decreasing spline which approximates y = f(x) via method
            _find_convex_decreasing_spline_binary_search.

        """
        self.sp = self._find_convex_decreasing_spline_binary_search(x, y)
        return self.sp

    def predict(self, x):
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.

        Returns
        -------
        y : array_like
            Returns y = fitted_spline(x) as approximation to y = f(x).

        """
        return self.sp(x)


class ExpModel(BaseModel, ExpFamilyMixin):
    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        ratio_a_b_mutual_nn = np.divide(base_signals.a_mutual_nn, base_signals.b_mutual_nn)
        lambda_initial, e_0_initial = self._initial_fit(ratio_a_b_mutual_nn, base_signals.exposure_a_mutual_nn)
        logging.info("Initial parameters computed.")
        return Params(all=[lambda_initial, e_0_initial])

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        params, _ = curve_fit(self._exp, base_signals.exposure_a_mutual_nn,
                              fit_result.ratio_a_b_mutual_nn_corrected,
                              p0=initial_params.kwargs.get('all'), maxfev=ExpConsts.MAX_FEVAL)

        a_correction = self._exp(base_signals.exposure_a_mutual_nn, *params)
        b_correction = self._exp(base_signals.exposure_b_mutual_nn, *params)

        return Corrections(a_correction, b_correction), Params(all=params)

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        optimal_params_list = optimal_params.kwargs.get('all')

        degradation_a_nn = self._exp(base_signals.exposure_a_nn, *optimal_params_list)
        degradation_b_nn = self._exp(base_signals.exposure_b_nn, *optimal_params_list)

        return FinalResult(base_signals, degradation_a_nn, degradation_b_nn)

    @staticmethod
    def _exp(x, lambda_, e_0):
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        lambda_ : float
            Rate of exponential decay.
        e_0 : float
            Offset of exponential decay

        Returns
        -------
        y : array_like
            Returns 1 - exp(lambda_ * e_0) + exp(-lambda_ * (x - e_0)). It holds y(0) = 1.

        """
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0))
        return y

    def __repr__(self):
        return "ExpModel"


class ExpLinModel(BaseModel, ExpFamilyMixin):
    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        ratio_a_b_mutual_nn = np.divide(base_signals.a_mutual_nn, base_signals.b_mutual_nn)
        lambda_initial, e_0_initial = self._initial_fit(ratio_a_b_mutual_nn, base_signals.exposure_a_mutual_nn)
        logging.info("Initial parameters computed.")
        return Params(all=[lambda_initial, e_0_initial, 0])

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        params, _ = curve_fit(self._exp_lin, base_signals.exposure_a_mutual_nn,
                              fit_result.ratio_a_b_mutual_nn_corrected,
                              p0=initial_params.kwargs.get('all'), maxfev=ExpConsts.MAX_FEVAL)

        a_correction = self._exp_lin(base_signals.exposure_a_mutual_nn, *params)
        b_correction = self._exp_lin(base_signals.exposure_b_mutual_nn, *params)

        return Corrections(a_correction, b_correction), Params(all=params)

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        optimal_params_list = optimal_params.kwargs.get('all')

        degradation_a_nn = self._exp_lin(base_signals.exposure_a_nn, *optimal_params_list)
        degradation_b_nn = self._exp_lin(base_signals.exposure_b_nn, *optimal_params_list)

        return FinalResult(base_signals, degradation_a_nn, degradation_b_nn)

    @staticmethod
    def _exp_lin(x, lambda_, e_0, linear):
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        lambda_ : float
            Rate of exponential decay.
        e_0 : float
            Offset of exponential decay
        linear : float
            Coefficient of linear decay.

        Returns
        -------
        y : array_like
            Returns 1 - exp(lambda_ * e_0) + exp(-lambda_ * (x - e_0)) + linear * x. It holds y(0) = 1.

        """
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0)) + linear * x
        return y

    def __repr__(self):
        return "ExpLinModel"


class SplineModel(BaseModel):
    def __init__(self, k=SplConsts.K, steps=SplConsts.STEPS, thinning=SplConsts.THINNING, s_max=SplConsts.S_MAX):
        """

        Parameters
        ----------
        k : int
            Degree of polynomial used in spline fitting.
        steps : int
            Number of steps in search method for decreasing spline.
        thinning : int
            Take each thinning-th sample of signal when fitting spline to speed up the process.
        s_max : float
            Upper bound on bisection interval.
        """
        self.k = k
        self.steps = steps
        self.thinning = thinning
        self.s_max = s_max
        self.model = DegradationSpline(self.k, steps=self.steps, s_max=self.s_max)

    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        return Params()

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        self.model.fit(base_signals.exposure_a_mutual_nn[::self.thinning],
                       fit_result.ratio_a_b_mutual_nn_corrected[::self.thinning])
        a_correction = self.model.predict(base_signals.exposure_a_mutual_nn)
        b_correction = self.model.predict(base_signals.exposure_b_mutual_nn)

        return Corrections(a_correction, b_correction), Params(sp=self.model)

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        sp = optimal_params.kwargs.get('sp')
        return FinalResult(base_signals, sp.predict(base_signals.exposure_a_nn), sp.predict(base_signals.exposure_b_nn))

    def __repr__(self):
        return "SplineModel"


class IsotonicModel(BaseModel):
    def __init__(self, smoothing=IsoConsts.SMOOTHING, y_max=IsoConsts.Y_MAX, y_min=IsoConsts.Y_MIN,
                 increasing=IsoConsts.INCREASING, out_of_bounds=IsoConsts.OUT_OF_BOUNDS, k=IsoConsts.K,
                 steps=IsoConsts.STEPS, number_of_points=IsoConsts.NUMBER_OF_POINTS):
        """

        Parameters
        ----------
        smoothing : Boolean
            True if we want to smooth Isotonic regression with splines in the end, otherwise False.
        y_max : float
            Maximal value for IsotonicRegression model.
        y_min : float
            Minimal value for IsotonicRegression model.
        increasing : Boolean
            True if we want increasing IsotonicRegression, False if we want decreasing.
        out_of_bounds : string
            What should IsotonicRegression do outside the minimal and maximal x coordinate given while fitting.
            Possible inputs are 'nan', 'clip' or 'raise'.
        k : int
            Degree of polynomial used in spline fitting.
        steps : int
            Number of steps in search method for decreasing spline.
        number_of_points : int
            Number of points used for smoothing IsotonicRegression model.
        """
        self.k = k
        self.steps = steps
        self.smoothing = smoothing
        self.number_of_points = number_of_points
        self.model = IsotonicRegression(y_max=y_max, y_min=y_min, increasing=increasing, out_of_bounds=out_of_bounds)

    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        return Params()

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        self.model.fit(base_signals.exposure_a_mutual_nn, fit_result.ratio_a_b_mutual_nn_corrected)
        if self.smoothing:
            max_exposure = base_signals.exposure_a_mutual_nn[-1]
            exposure = np.linspace(0, max_exposure, self.number_of_points)
            ratio = self.model.predict(exposure)
            self.model = DegradationSpline(self.k, steps=self.steps, s_max=SplConsts.S_MAX)
            self.model.fit(exposure, ratio)

        a_correction = self.model.predict(base_signals.exposure_a_mutual_nn)
        b_correction = self.model.predict(base_signals.exposure_b_mutual_nn)

        return Corrections(a_correction, b_correction), Params(model=self.model)

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        model = optimal_params.kwargs.get('model')
        return FinalResult(base_signals, model.predict(base_signals.exposure_a_nn),
                           model.predict(base_signals.exposure_b_nn))

    def __repr__(self):
        return "IsotonicModel"


class SmoothMonotonicModel(BaseModel):
    def __init__(self, increasing=SMRConsts.INCREASING, number_of_points=SMRConsts.NUMBER_OF_POINTS,
                 y_max=SMRConsts.Y_MAX, y_min=SMRConsts.Y_MIN, out_of_bounds=SMRConsts.OUT_OF_BOUNDS,
                 solver=SMRConsts.SOLVER, lam=SMRConsts.LAM):
        """

        Parameters
        ----------
        increasing : bool
            True if we want to smooth Isotonic regression with splines in the end, otherwise False.
        number_of_points : int
            Number of points used for smoothing IsotonicRegression model.
        y_max : float
            Maximal value for IsotonicRegression model.
        y_min : float
            Minimal value for IsotonicRegression model.
        out_of_bounds : string
            What should IsotonicRegression do outside the minimal and maximal x coordinate given while fitting.
            Possible inputs are 'nan', 'clip' or 'raise'.
        solver : cp.SOLVER
            For SOLVER one can choose: ECOS, ECOS_BB, OSQP, SCS, others (better) are under
            license: GUROBI (best), CVXOPT.
        lam : float
            Parameter for smoothing greater lam greater smoothing.
        """
        self.increasing = increasing
        self.number_of_points = number_of_points
        self.model_for_help = IsotonicRegression(y_max=y_max, y_min=y_min,
                                                 increasing=increasing, out_of_bounds=out_of_bounds)
        self.solver = solver
        self.model = None
        self.lam = lam

    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        return Params()

    def _calculate_smooth_monotone_function(self, exposure, convex):
        y = self.model_for_help.predict(exposure)
        mu = cp.Variable(self.number_of_points)
        objective = cp.Minimize(cp.sum_squares(mu - y) + self.lam * cp.sum_squares(mu[:-1] - mu[1:]))
        constraints = [mu <= 1, mu[0] == 1]
        if not self.increasing:
            constraints.append(mu[1:] <= mu[:-1])

        if convex:
            constraints.append(mu[:-2] + mu[2:] >= 2 * mu[1:-1])

        model = cp.Problem(objective, constraints)
        model.solve(solver=self.solver)
        spline = interp1d(exposure, mu.value, fill_value="extrapolate")
        return spline

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        self.model_for_help.fit(base_signals.exposure_a_mutual_nn, fit_result.ratio_a_b_mutual_nn_corrected)
        max_exposure = base_signals.exposure_a_mutual_nn[-1]
        exposure = np.linspace(0, max_exposure, self.number_of_points)

        if method == CorrectionMethod.CORRECT_ONE:
            convex = True
        else:
            convex = False

        self.model = self._calculate_smooth_monotone_function(exposure, convex)
        a_correction = self.model(base_signals.exposure_a_mutual_nn)
        b_correction = self.model(base_signals.exposure_b_mutual_nn)

        return Corrections(a_correction, b_correction), Params(model=self.model)

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        model = optimal_params.kwargs.get('model')
        return FinalResult(base_signals, model(base_signals.exposure_a_nn),
                           model(base_signals.exposure_b_nn))

    def __repr__(self):
        return "SmoothMonotonicRegressionModel"


class EnsembleModel(BaseModel):
    def __init__(self, models: List[BaseModel] = EnsConsts.MODELS, weights=EnsConsts.WEIGHTS):
        """

        Parameters
        ----------
        models : List[BaseModel]
            List of models we would like to train in an Ensemble.
        weights : array_like
            List of probability contributions of each model in Ensemble.
        """
        self.models = models
        self.weights = weights

    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        params: List[Params] = []
        for model in self.models:
            params.append(model.get_initial_params(base_signals))
        return Params(all=params)

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        initial_params_list: List[Params] = initial_params.kwargs.get("all")
        # Placeholder for empty NumPy array of adequate size (size can be obtained 5 lines below)
        a_correction, b_correction = 0, 0
        params = []
        for index, (model, weight) in enumerate(zip(self.models, self.weights)):
            corrections, current_params = model.fit_and_correct(base_signals, fit_result, initial_params_list[index],
                                                                method)
            # a_correction += np.divide(weight, corrections.a_correction)
            # b_correction += np.divide(weight, corrections.b_correction)
            a_correction += np.multiply(weight, corrections.a_correction)
            b_correction += np.multiply(weight, corrections.b_correction)
            params.append(current_params)
        # return Corrections(np.divide(1, a_correction), np.divide(1, b_correction)), Params(params=params)
        return Corrections(a_correction, b_correction), Params(params=params)

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        optimal_params_list: List[Params] = optimal_params.kwargs.get('params')
        # Placeholder for empty NumPy array of adequate size (size can be obtained 4 lines below)
        degradation_a_nn, degradation_b_nn = 0, 0
        for index, (model, weight) in enumerate(zip(optimal_params_list, self.weights)):
            partial_results: FinalResult = self.models[index].compute_final_result(base_signals,
                                                                                   optimal_params_list[index])
            degradation_a_nn += weight * partial_results.degradation_a_nn
            degradation_b_nn += weight * partial_results.degradation_b_nn
        return FinalResult(base_signals, degradation_a_nn, degradation_b_nn)

    def __repr__(self):
        return "EnsembleModel"


class GPFamilyMixin:
    def __init__(self, normalization, dual_kernel, t_mean=0.0, t_std=1.0, x_mean=0.0, x_std=1.0):
        self.normalization = normalization
        self.dual_kernel = dual_kernel
        self.t_mean = t_mean
        self.t_std = t_std
        self.x_mean = x_mean
        self.x_std = x_std

    @staticmethod
    def _gaussian_process(kernel, t, x):
        logging.info(f"Running GP on data with t = {t.shape} and x = {x.shape}.")

        status.update_progress("Performing initial fit", 20)

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
    def _gp_samples(mode, base_signals, final_result, dual_kernel):
        if mode == Mode.GENERATOR:
            GPConsts.DOWNSAMPLING_FACTOR_A = int(GPConsts.DOWNSAMPLING_FACTOR_A / 10)
            GPConsts.DOWNSAMPLING_FACTOR_B = int(GPConsts.DOWNSAMPLING_FACTOR_B / 10)

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

    def _gp_initial_fit(self, mode, base_signals, final_result):

        # Initial fit samples
        t_initial, x_initial = self._gp_samples(mode, base_signals, final_result, self.dual_kernel)

        # Initial fit kernel
        if self.dual_kernel:
            kernel = Kernels.dual_matern_kernel + Kernels.dual_white_kernel
        else:
            kernel = Kernels.matern_kernel + Kernels.white_kernel

        if self.normalization:
            t_initial, x_initial = normalize(t_initial, self.t_mean, self.t_std), \
                                   normalize(x_initial, self.x_mean, self.x_std)
        else:
            x_initial = x_initial - self.x_mean

        gpr = self._gaussian_process(kernel, t_initial, x_initial)

        return gpr

    def _initial_fit(self, mode, base_signals, final_result):
        logging.info("Running initial fit.")
        length_scale_initial, noise_level_a_initial, noise_level_b_initial = None, None, None

        # Train
        gpr = self._gp_initial_fit(mode, base_signals, final_result)

        length_scale_initial = gpr.kernel_.get_params()["k1__length_scale"]

        if self.dual_kernel:
            noise_level_a_initial = gpr.kernel_.get_params()["k2__noise_level_a"]
            noise_level_b_initial = gpr.kernel_.get_params()["k2__noise_level_b"]
        else:
            noise_level_a_initial = gpr.kernel_.get_params()["k2__noise_level"]

        return length_scale_initial, noise_level_a_initial, noise_level_b_initial


class LocalGPModel(BaseOutputModel):
    def __init__(self,
                 dual_kernel=GPConsts.DUAL_KERNEL,
                 normalization=GPConsts.NORMALIZATION,
                 clipping=GPConsts.CLIPPING,
                 points_in_window=GPConsts.POINTS_IN_WINDOW):
        self.dual_kernel = dual_kernel
        self.normalization = normalization
        self.clipping = clipping
        self.point_in_window = points_in_window

    def fit_and_predict(self, mode: Mode, base_signals: BaseSignals, final_result: FinalResult, t_hourly_out):
        # Training samples
        t_a, a = base_signals.t_a_nn, final_result.a_nn_corrected
        t_b, b = base_signals.t_b_nn, final_result.b_nn_corrected

        t, x = self._interleave(self.dual_kernel, t_a, t_b, a, b)

        if mode == Mode.GENERATOR:
            window = GPConsts.GEN_WINDOW
        else:
            window = GPConsts.WINDOW

        if self.dual_kernel:
            labels_out = GPConsts.LABEL_OUT * np.ones(shape=t_hourly_out.shape)
            labels_out = labels_out.astype(np.int).reshape(-1, 1)
            t_hourly_out = np.concatenate([t_hourly_out, labels_out], axis=1).reshape(-1, 2)

        if self.clipping:
            x_mean, x_std = np.mean(x), np.std(x)
            non_clipping_indices = np.logical_and(np.greater_equal(x, x_mean - 5 * x_std),
                                                  np.less_equal(x, x_mean + 5 * x_std)).astype(np.bool).flatten()

            t, x = t[non_clipping_indices, :], x[non_clipping_indices, :]

        signal_hourly_out, signal_std_hourly_out = self._gp_target_time(mode, t, x, t_hourly_out, window,
                                                                        self.point_in_window)

        return signal_hourly_out.reshape(-1, 1), signal_std_hourly_out.reshape(-1, 1), OutParams()

    @staticmethod
    @njit(cache=True)
    def _interleave(dual_kernel, t_a, t_b, a, b, label_a=GPConsts.LABEL_A, label_b=GPConsts.LABEL_B):
        index_a = 0
        index_b = 0
        index_together = 0
        a_length = t_a.size
        b_length = t_b.size
        if dual_kernel:
            t_merged = np.empty((a_length + b_length, 2))
        else:
            t_merged = np.empty((a_length + b_length, 1))

        val_merged = np.empty((a_length + b_length, 1))

        while index_together < a_length + b_length:
            while index_a < a_length and (t_a[index_a] <= t_b[index_b] or index_b == b_length):
                if dual_kernel:
                    t_merged[index_together, :] = np.array([t_a[index_a], label_a])
                else:
                    t_merged[index_together] = t_a[index_a]

                val_merged[index_together] = a[index_a]
                index_a += 1
                index_together += 1

            while index_b < b_length and (t_b[index_b] <= t_a[index_a] or index_a == a_length):
                if dual_kernel:
                    t_merged[index_together] = np.array([t_b[index_b], label_b])
                else:
                    t_merged[index_together] = t_a[index_a]

                val_merged[index_together] = b[index_b]
                index_b += 1
                index_together += 1

        return t_merged, val_merged

    def _gp_target_time(self, mode: Mode, t, x, t_hourly_out, window, points_in_window):
        """
        :param t: Time of all signals.
        :param x: Signal.
        :param t_hourly_out: Time at which prediction should happen.
        :param window: How much time left/right of prediction time should be taken into account.
        :param points_in_window: How many points should be in window.
        :return signal_hourly_out, signal_std_hourly_out: NumPy array of predictions, same shape as t_target.
            Prediction at position i corresponds to time at position i in t_target.
            For windows without points np.nan is returned.
        """
        t_hourly_length = t_hourly_out.shape[0]
        end_index_out = 0
        t_mean, t_std, x_mean, x_std = 0.0, 1.0, 0.0, 1.0
        signal_hourly_out = np.zeros((t_hourly_length,))
        signal_std_hourly_out = np.zeros((t_hourly_length,))

        if mode == Mode.GENERATOR:
            step = window / GPConsts.WINDOW_FRACTION
        else:
            step = window / GPConsts.WINDOW_FRACTION

        gpr = None
        cur_target_t_mid = t_hourly_out[0, 0]
        while True:
            # Determine points of t and x that fall within window
            start_index, end_index = extract_time_window(t[:, 0], cur_target_t_mid, window)

            cur_x = x[start_index:end_index, :].copy()
            cur_t = t[start_index:end_index, :].copy()

            # Determine points of t_hourly_out that fall within window
            start_index_out = end_index_out
            win_end = cur_target_t_mid + window / (GPConsts.WINDOW_FRACTION * 2.0)

            while end_index_out < t_hourly_length and t_hourly_out[end_index_out, 0] <= win_end:
                end_index_out += 1

            if end_index_out < t_hourly_length and t_hourly_out[end_index_out, 0] + step > t_hourly_out[-1, 0]:
                end_index_out = t_hourly_length

            cur_target_t = t_hourly_out[start_index_out:end_index_out, :].copy()

            percentage_merging = int(100 * end_index_out / t_hourly_length)
            percentage_overall = int(30 + 50 * end_index_out / t_hourly_length)
            status.update_progress("Merging at: " + str(percentage_merging) + " %", percentage_overall)
            logging.info(f"Local GP merging currently at {percentage_merging} %.")
            logging.info("Target indices = {:<20}\tt_mid = {:<10}\tt_window = {:<20}\tt_target = {:<20}"
                         .format(str((start_index_out, end_index_out)), str(np.around(cur_target_t_mid, 2)),
                                 str((np.around(cur_t[0, 0], 2), np.around(cur_t[-1, 0], 2))),
                                 str((np.around(cur_target_t[0, 0], 2), np.around(cur_target_t[-1, 0], 2)))))

            if np.size(cur_x) < 1:
                if self.normalization:
                    cur_target_t = normalize(cur_target_t, t_mean, t_std)
            else:
                # Normalize data
                x_mean, x_std = np.mean(cur_x), np.std(cur_x)
                t_mean, t_std = np.mean(cur_t[:, 0]), np.std(cur_t[:, 0])

                if self.normalization:
                    cur_t, cur_x = normalize(cur_t, t_mean, t_std), normalize(cur_x, x_mean, x_std)
                    cur_target_t = normalize(cur_target_t, t_mean, t_std)
                else:
                    cur_x -= x_mean

                # Clip values to 5 * std <= x <= 5 * std
                if self.clipping:
                    clip_std = np.std(cur_x[:, 0])
                    clip_indices = np.logical_and(np.greater_equal(cur_x[:, 0], -5 * clip_std),
                                                  np.less_equal(cur_x[:, 0], 5 * clip_std)).astype(np.bool).flatten()

                    cur_t, cur_x = cur_t[clip_indices, :], cur_x[clip_indices, :]

                # Downsample data
                if 2 * cur_x.size > points_in_window:
                    downsampling_factor = int(cur_x.size / points_in_window)
                else:
                    downsampling_factor = 1

                cur_t_down, cur_x_down = np.copy(cur_t)[::downsampling_factor], np.copy(cur_x)[::downsampling_factor]

                # Fit GP on transformed points
                if self.normalization:
                    scale = (1 - 1 / GPConsts.WINDOW_FRACTION) * window / (t_std * 4 * 2)
                else:
                    scale = (1 - 1 / GPConsts.WINDOW_FRACTION) * window / (4 * 2)

                if self.dual_kernel:
                    kernel = DualMatern(length_scale=scale,
                                        length_scale_bounds=(1e-10, scale), nu=GPConsts.MATERN_NU) + DualWhiteKernel()
                else:
                    kernel = Matern(length_scale=scale, length_scale_bounds=(1e-10, scale), nu=GPConsts.MATERN_NU) \
                             + WhiteKernel(GPConsts.WHITE_NOISE_LEVEL, GPConsts.WHITE_NOISE_LEVEL_BOUNDS)

                gpr = GaussianProcessRegressor(kernel=kernel,
                                               n_restarts_optimizer=GPConsts.N_RESTARTS_OPTIMIZER)

                gpr.fit(cur_t_down, cur_x_down)

            cur_x_pred, cur_std_pred = gpr.predict(cur_target_t, return_std=True)

            # Project back
            if self.normalization:
                cur_x_pred = unnormalize(cur_x_pred, x_mean, x_std)
                cur_std_pred = cur_std_pred * (x_std ** 2)
            else:
                cur_x_pred += x_mean

            # Store prediction
            signal_hourly_out[start_index_out:end_index_out] = cur_x_pred.ravel()
            signal_std_hourly_out[start_index_out:end_index_out] = cur_std_pred.ravel()

            if end_index_out >= t_hourly_length:
                break

            cur_target_t_mid = cur_target_t_mid + step

        return signal_hourly_out, signal_std_hourly_out


class GPModel(GPFamilyMixin, BaseOutputModel):
    def __init__(self,
                 dual_kernel=GPConsts.DUAL_KERNEL,
                 normalization=GPConsts.NORMALIZATION):
        super(GPModel, self).__init__(normalization, dual_kernel)

    def fit_and_predict(self, mode: Mode, base_signals: BaseSignals, final_result: FinalResult, t_hourly_out):
        # Training samples
        t_a, a = base_signals.t_a_nn, final_result.a_nn_corrected
        t_b, b = base_signals.t_b_nn, final_result.b_nn_corrected

        self.x_mean, self.x_std = np.mean(np.concatenate((a, b), axis=0)), np.std(np.concatenate((a, b), axis=0))
        self.t_mean, self.t_std = np.mean(np.concatenate((t_a, t_b), axis=0)), \
                                  np.std(np.concatenate((t_a, t_b), axis=0))

        if self.dual_kernel:
            labels_out = GPConsts.LABEL_OUT * np.ones(shape=t_hourly_out.shape)
            labels_out = labels_out.astype(np.int).reshape(-1, 1)
            t_hourly_out = np.concatenate([t_hourly_out, labels_out], axis=1).reshape(-1, 2)

        # Normalize data
        if self.normalization:
            logging.info(f"Data normalized with t_mean = {self.t_mean}, t_std = {self.t_std}, "
                         f"x_mean = {self.x_mean} and x_std = {self.x_std}.")
            t_hourly_out = normalize(t_hourly_out, self.t_mean, self.t_std)

        gpr = self._gp_initial_fit(mode, base_signals, final_result)

        # Prediction
        signal_hourly_out, signal_std_hourly_out = gpr.predict(t_hourly_out, return_std=True)

        if self.normalization:
            signal_hourly_out = unnormalize(signal_hourly_out, self.x_mean, self.x_std)
            signal_std_hourly_out = signal_std_hourly_out * (self.x_std ** 2)
        else:
            signal_hourly_out = signal_hourly_out + self.x_mean

        return signal_hourly_out.reshape(-1, 1), signal_std_hourly_out.reshape(-1, 1), OutParams()


class SVGPModel(GPFamilyMixin, BaseOutputModel):
    def __init__(self,
                 dual_kernel=GPConsts.DUAL_KERNEL,
                 normalization=GPConsts.NORMALIZATION,
                 clipping=GPConsts.CLIPPING,
                 initial_fit=GPConsts.INITIAL_FIT,
                 num_inducing_points=GPConsts.NUM_INDUCING_POINTS,
                 train_inducing_variables=GPConsts.TRAIN_INDUCING_VARIABLES,
                 minibatch_size=GPConsts.MINIBATCH_SIZE,
                 max_iterations=GPConsts.MAX_ITERATIONS):

        super(SVGPModel, self).__init__(normalization, dual_kernel)
        self.clipping = clipping
        self.initial_fit = initial_fit

        self.num_inducing_points = num_inducing_points
        self.train_inducing_variables = train_inducing_variables
        self.minibatch_size = minibatch_size
        self.max_iterations = max_iterations

    def _prepare_data(self, t_a, t_b, a, b, t_hourly_out):

        # Data standardization parameters
        self.x_mean, self.x_std = np.mean(np.concatenate((a, b), axis=0)), np.std(np.concatenate((a, b), axis=0))
        self.t_mean, self.t_std = np.mean(np.concatenate((t_a, t_b), axis=0)), \
                                  np.std(np.concatenate((t_a, t_b), axis=0))

        # # Downsample a
        # downsampling_rate_a = a.shape[0] // b.shape[0]
        # t_a, a = t_a[::downsampling_rate_a], a[::downsampling_rate_a]

        # Concatenate
        x = np.concatenate([a, b], axis=0).reshape(-1, 1)
        t = np.concatenate([t_a, t_b], axis=0).reshape(-1, 1)

        # Add labels for dual kernel
        if self.dual_kernel:
            # Training
            labels = np.concatenate([GPConsts.LABEL_A * np.ones(shape=a.shape),
                                     GPConsts.LABEL_B * np.ones(shape=b.shape)])
            labels = labels.astype(np.int).reshape(-1, 1)
            t = np.concatenate([t, labels], axis=1).reshape(-1, 2)

            # Output
            labels_out = GPConsts.LABEL_OUT * np.ones(shape=t_hourly_out.shape)
            labels_out = labels_out.astype(np.int).reshape(-1, 1)
            t_hourly_out = np.concatenate([t_hourly_out, labels_out], axis=1).reshape(-1, 2)

        # Normalize data
        if self.normalization:
            logging.info(f"Data normalized with t_mean = {self.t_mean}, t_std = {self.t_std}, "
                         f"x_mean = {self.x_mean} and x_std = {self.x_std}.")
            t, x = normalize(t, self.t_mean, self.t_std), normalize(x, self.x_mean, self.x_std)
            t_hourly_out = normalize(t_hourly_out, self.t_mean, self.t_std)
        else:
            logging.info(f"Data normalized with x_mean = {self.x_mean}.")
            x = x - self.x_mean

        # Clip values to mean - 5 * std <= x <= mean + 5 * std
        if self.clipping:
            clip_mean, clip_std = np.mean(x), np.std(x)
            clip_indices = np.multiply(np.greater_equal(x, clip_mean - 5 * clip_std),
                                       np.less_equal(x, clip_mean + 5 * clip_std)).astype(np.bool).flatten()
            t, x = t[clip_indices, :], x[clip_indices]
            logging.info(f"{clip_indices.shape[0] - clip_indices.sum()} measurements were clipped.")

        logging.info(f"Dual kernel is set to {self.dual_kernel}. Data shapes are t {t.shape}, "
                     f"x {x.shape} and t_hourly {t_hourly_out.shape}.")

        return t, x, t_hourly_out

    def fit_and_predict(self, mode: Mode, base_signals: BaseSignals, final_result: FinalResult, t_hourly_out):

        t_hourly_out = t_hourly_out.reshape(-1, 1)

        # Training samples
        t_a, a = base_signals.t_a_nn, final_result.a_nn_corrected
        t_b, b = base_signals.t_b_nn, final_result.b_nn_corrected

        t, x, t_hourly_out = self._prepare_data(t_a, t_b, a, b, t_hourly_out.copy())

        # Induction variables
        t_uniform = np.linspace(np.min(t[:, 0]), np.max(t[:, 0]), self.num_inducing_points)
        t_uniform_indices = find_nearest(t[:, 0], t_uniform).astype(int)
        inducing_points = t[t_uniform_indices, :].copy()

        # Kernel
        if self.dual_kernel:
            kernel = gpflow.kernels.Sum([Kernels.gpf_dual_matern12, Kernels.gpf_dual_white])
            # kernel = gpflow.kernels.Sum([Kernels.gpf_dual_matern32, Kernels.gpf_dual_white])
        else:
            kernel = gpflow.kernels.Sum([Kernels.gpf_matern12, Kernels.gpf_white])

        # Global trends
        m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), inducing_points, num_data=t.shape[0])

        if 2 * GPConsts.PRIOR_POSTERIOR_LENGTH < t_hourly_out.shape[0]:
            t_prior = t_posterior = t_hourly_out[::int(t_hourly_out.shape[0] / GPConsts.PRIOR_POSTERIOR_LENGTH), :]
        else:
            t_prior = t_posterior = t_hourly_out

        # Prior sampling
        x_prior = m.predict_f_samples(t_prior, GPConsts.PRIOR_POSTERIOR_SAMPLES)[:, :, 0].numpy().T
        logging.info(f"Samples prior of shape {x_prior.shape}.")

        # Initial fit
        if self.initial_fit:
            length_scale_initial, noise_level_a_initial, noise_level_b_initial = self._initial_fit(mode, base_signals,
                                                                                                   final_result)

            # Assign initial values
            m.kernel.kernels[0].lengthscale.assign(length_scale_initial)
            if self.dual_kernel:
                m.kernel.kernels[1].variance_a.assign(noise_level_a_initial)
                m.kernel.kernels[1].variance_b.assign(noise_level_b_initial)
            else:
                m.kernel.kernels[1].variance.assign(noise_level_a_initial)

        # Non-trainable parameters
        if not self.train_inducing_variables:
            gpflow.utilities.set_trainable(m.inducing_variable, False)

        logging.info("Model created.\n\n" + str(get_summary(m)) + "\n")

        # Train
        signal_hourly_out, signal_std_hourly_out, iter_loglikelihood = self.train_model(m, t, x, t_hourly_out)

        # Posterior sampling
        x_posterior = m.predict_f_samples(t_posterior, GPConsts.PRIOR_POSTERIOR_SAMPLES)[:, :, 0].numpy().T
        logging.info(f"Samples posterior of shape {x_posterior.shape}.")

        # Revert normalization
        if self.normalization:
            signal_hourly_out = unnormalize(signal_hourly_out, self.x_mean, self.x_std)
            signal_std_hourly_out = signal_std_hourly_out * (self.x_std ** 2)
            inducing_points = unnormalize(inducing_points, self.t_mean, self.t_std)
            for i in range(x_prior.shape[1]):
                x_prior[:, i] = unnormalize(x_prior[:, i], self.x_mean, self.x_std)
                x_posterior[:, i] = unnormalize(x_posterior[:, i], self.x_mean, self.x_std)

            t_prior = t_posterior = unnormalize(t_prior, self.t_mean, self.t_std)
        else:
            signal_hourly_out = signal_hourly_out + self.x_mean
            x_prior = x_prior + self.x_mean
            x_posterior = x_posterior + self.x_mean

        params_out = OutParams(iter_loglikelihood, inducing_points, svgp_prior_samples=x_prior,
                               svgp_t_prior=t_prior[:, 0], svgp_posterior_samples=x_posterior,
                               svgp_t_posterior=t_posterior[:, 0])

        return signal_hourly_out, signal_std_hourly_out, params_out

    def train_model(self, model, t, x, t_hourly_out):
        # Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((t, x)).repeat().shuffle(buffer_size=t.shape[0],
                                                                                    seed=Const.RANDOM_SEED)
        # Training
        start = time.time()
        iter_loglikelihood = SVGaussianProcess.run_optimization(model=model,
                                                                iterations=ci_niter(self.max_iterations),
                                                                train_dataset=train_dataset,
                                                                minibatch_size=self.minibatch_size)
        end = time.time()

        logging.info("Training finished after: {:>10} sec".format(end - start))
        logging.info("Trained model.\n\n" + str(get_summary(model)) + "\n")

        # Prediction
        signal_hourly_out, signal_var_hourly_out = model.predict_y(t_hourly_out)
        signal_std_hourly_out = tf.sqrt(signal_var_hourly_out)

        signal_hourly_out = tf.reshape(signal_hourly_out, [-1]).numpy()
        signal_std_hourly_out = tf.reshape(signal_std_hourly_out, [-1]).numpy()

        return signal_hourly_out.reshape(-1, 1), signal_std_hourly_out.reshape(-1, 1), iter_loglikelihood
