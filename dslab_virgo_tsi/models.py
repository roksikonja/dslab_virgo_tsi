import logging
from typing import List, Tuple

import cvxpy as cp
import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splrep, interp1d
from scipy.optimize import curve_fit, minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from dslab_virgo_tsi.base import BaseModel, BaseSignals, Params, FinalResult, FitResult, Corrections, \
    CorrectionMethod
from dslab_virgo_tsi.model_constants import EnsembleConstants as EnsConsts
from dslab_virgo_tsi.model_constants import ExpConstants as ExpConsts
from dslab_virgo_tsi.model_constants import IsotonicConstants as IsoConsts
from dslab_virgo_tsi.model_constants import SmoothMonotoneRegressionConstants as SMRConsts
from dslab_virgo_tsi.model_constants import SplineConstants as SplConsts


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
    def __init__(self, k, steps):
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
        end = 1
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
    def __init__(self, k=SplConsts.K, steps=SplConsts.STEPS, thinning=SplConsts.THINNING):
        """

        Parameters
        ----------
        k : int
            Degree of polynomial used in spline fitting.
        steps : int
            Number of steps in search method for decreasing spline.
        thinning : int
            Take each thinning-th sample of signal when fitting spline to speed up the process.
        """
        self.k = k
        self.steps = steps
        self.thinning = thinning
        self.model = DegradationSpline(self.k, steps=self.steps)

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
            self.model = DegradationSpline(self.k, steps=self.steps)
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


class SmoothMonotoneRegression(BaseModel):
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

# class SVGPModel(BaseOutputModel):
#     def __init__(self, base_signals: BaseSignals, final_result: FinalResult):
#         pass
#
#     # Data standardization parameters
#     x_mean = np.mean(np.concatenate((final_result.a_nn_corrected, final_result.b_nn_corrected), axis=0))
#     x_std = np.std(np.concatenate((final_result.a_nn_corrected, final_result.b_nn_corrected), axis=0))
#     t_mean = np.mean(np.concatenate((base_signals.t_a_nn, base_signals.t_b_nn), axis=0))
#     t_std = np.std(np.concatenate((base_signals.t_a_nn, base_signals.t_b_nn), axis=0))
#
#
#
# # Training samples
# t_a_downsampled, a_downsampled = base_signals.t_a_nn, final_result.a_nn_corrected
# t_b_downsampled, b_downsampled = base_signals.t_b_nn, final_result.b_nn_corrected
#
# x, t = np.zeros(0), np.zeros(0)
# if output_method == OutputMethod.LOCAL_GP:
#     t, x = self._interleave(t_a_downsampled, t_b_downsampled, a_downsampled, b_downsampled)
#
# elif output_method == OutputMethod.SVGP:
#     # Subsample a
#     subsampling_rate_a = a_downsampled.shape[0] // b_downsampled.shape[0]
#     t_a_downsampled, a_downsampled = t_a_downsampled[::subsampling_rate_a], a_downsampled[::subsampling_rate_a]
#
#     # Concatenate
#     x = np.concatenate([a_downsampled, b_downsampled], axis=0).reshape(-1, 1)
#     t = np.concatenate([t_a_downsampled, t_b_downsampled], axis=0).reshape(-1, 1)
#
#     # Add labels for dual kernel
#     if GPConsts.DUAL_KERNEL:
#         # Training
#         labels = np.concatenate([GPConsts.LABEL_A * np.ones(shape=a_downsampled.shape),
#                                  GPConsts.LABEL_B * np.ones(shape=b_downsampled.shape)])
#         labels = labels.astype(np.int).reshape(-1, 1)
#
#         t = np.concatenate([t, labels], axis=1).reshape(-1, 2)
#
#         # Output
#         labels_out = GPConsts.LABEL_OUT * np.ones(shape=t_hourly_out.shape)
#         labels_out = labels_out.astype(np.int).reshape(-1, 1)
#         t_hourly_out = np.concatenate([t_hourly_out, labels_out], axis=1).reshape(-1, 2)
#
#         # Normalize data
#         if GPConsts.NORMALIZE:
#             logging.info(f"Data normalized with t_mean = {t_mean}, t_std = {t_std}, "
#                          f"x_mean = {x_mean} and x_std = {x_std}.")
#             t, x = normalize(t, t_mean, t_std), normalize(x, x_mean, x_std)
#             t_hourly_out = normalize(t_hourly_out, t_mean, t_std)
#         else:
#             x = x - x_mean
#
#         # Clip values to mean - 5 * std <= x <= mean + 5 * std
#         clip_indices = np.zeros(1)
#         if GPConsts.CLIP:
#             clip_std = np.std(x)
#             clip_mean = np.mean(x)
#             clip_indices = np.multiply(np.greater_equal(x[:], clip_mean - 5 * clip_std),
#                                        np.less_equal(x[:], clip_mean + 5 * clip_std)).astype(np.bool).flatten()
#             t, x = t[clip_indices, :], x[clip_indices]
#
#         logging.info(f"Dual kernel is set to {GPConsts.DUAL_KERNEL}. Data shapes are t {t.shape}, "
#                      f"x {x.shape} and t_hourly {t_hourly_out.shape}. "
#                      f"{clip_indices.shape[0] - clip_indices.sum()} were clipped.")
#
# inducing_points = None
# iter_loglikelihood = None
# if output_method == OutputMethod.GP:
#     logging.info(f"{output_method} started.")
#
#     gpr = self._gp_initial_fit(self.mode, base_signals, final_result, t_mean, t_std, x_mean, x_std)
#
#     # Prediction
#     signal_hourly_out, signal_std_hourly_out = gpr.predict(t_hourly_out, return_std=True)
# elif output_method == OutputMethod.LOCAL_GP:
#     logging.info(f"{output_method} started.")
#
#     t_hourly_out = t_hourly_out[::24]
#     signal_hourly_out, signal_std_hourly_out = self._gp_target_time(t, x, t_hourly_out,
#                                                                     GPConsts.WINDOW, GPConsts.POINTS_IN_WINDOW)
# else:
#     logging.info(f"{output_method} started.")
#
#     # Initial fit
#     length_scale_initial, noise_level_a_initial, noise_level_b_initial = None, None, None
#     if GPConsts.INITIAL_FIT:
#         gpr = self._gp_initial_fit(self.mode, base_signals, final_result, t_mean, t_std, x_mean, x_std)
#         length_scale_initial = gpr.kernel_.get_params()["k1__length_scale"]
#         if GPConsts.DUAL_KERNEL:
#             noise_level_a_initial = gpr.kernel_.get_params()["k2__noise_level_a"]
#             noise_level_b_initial = gpr.kernel_.get_params()["k2__noise_level_b"]
#         else:
#             noise_level_a_initial = gpr.kernel_.get_params()["k2__noise_level"]
#
#     # Induction variables
#     t_uniform = np.linspace(np.min(t[:, 0]), np.max(t[:, 0]), GPConsts.NUM_INDUCING_POINTS)
#     t_uniform_indices = find_nearest(t[:, 0], t_uniform).astype(int)
#     z = t[t_uniform_indices, :].copy()
#
#     # Select kernel
#     if GPConsts.DUAL_KERNEL:
#         kernel = gpflow.kernels.Sum([Kernels.gpf_dual_matern12, Kernels.gpf_dual_white])
#     else:
#         kernel = gpflow.kernels.Sum([Kernels.gpf_matern12, Kernels.gpf_white])
#
#     # Model
#     m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), z, num_data=t.shape[0])
#
#     # Initial guess
#     if GPConsts.INITIAL_FIT:
#         # Initialize length scale
#         m.kernel.kernels[0].lengthscale.assign(length_scale_initial)
#
#         # Initialize noise level
#         if GPConsts.DUAL_KERNEL:
#             m.kernel.kernels[1].variance_a.assign(noise_level_a_initial)
#             m.kernel.kernels[1].variance_b.assign(noise_level_b_initial)
#         else:
#             m.kernel.kernels[1].variance.assign(noise_level_a_initial)
#
#     # Non-trainable parameters
#     if not GPConsts.TRAIN_INDUCING_VARIBLES:
#         gpflow.utilities.set_trainable(m.inducing_variable, False)
#
#     logging.info("Model created.\n\n" + str(get_summary(m)) + "\n")
#
#     # Dataset
#     train_dataset = tf.data.Dataset.from_tensor_slices((t, x)).repeat().shuffle(buffer_size=t.shape[0],
#                                                                                 seed=Const.RANDOM_SEED)
#     # Training
#     start = time.time()
#     iter_loglikelihood = SVGaussianProcess.run_adam(model=m,
#                                                     iterations=ci_niter(GPConsts.MAX_ITERATIONS),
#                                                     train_dataset=train_dataset,
#                                                     minibatch_size=GPConsts.MINIBATCH_SIZE)
#     inducing_points = z
#
#     end = time.time()
#     logging.info("Training finished after: {:>10} sec".format(end - start))
#     logging.info("Trained model.\n\n" + str(get_summary(m)) + "\n")
#
#     # Prediction
#     signal_hourly_out, signal_var_hourly_out = m.predict_y(t_hourly_out)
#     signal_std_hourly_out = tf.sqrt(signal_var_hourly_out)
#     signal_hourly_out = tf.reshape(signal_hourly_out, [-1]).numpy()
#     signal_std_hourly_out = tf.reshape(signal_std_hourly_out, [-1]).numpy()
#
#     # Compute final output
#     # Revert normalization
#     if GPConsts.NORMALIZE:
#         signal_hourly_out = unnormalize(signal_hourly_out, x_mean, x_std)
#         signal_std_hourly_out = signal_std_hourly_out * (x_std ** 2)
#         t_hourly_out = unnormalize(t_hourly_out, t_mean, t_std)
#         if isinstance(inducing_points, np.ndarray):
#             inducing_points = unnormalize(inducing_points, t_mean, t_std)
#     else:
#         signal_hourly_out = signal_hourly_out + x_mean
#
#     if GPConsts.DUAL_KERNEL:
#         t_hourly_out = t_hourly_out[:, 0].reshape(-1, 1)
