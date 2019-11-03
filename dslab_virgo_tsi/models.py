from typing import List, Tuple

import cvxpy as cp
import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splrep, interp1d
from scipy.optimize import curve_fit, minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from dslab_virgo_tsi.base import BaseModel, BaseSignals, Params, FinalResult, FitResult, Corrections, CorrectionMethod
from dslab_virgo_tsi.model_constants import EnsembleConstants as EnsConsts
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
        return np.all(spline_derivative(x.ravel()) < 0)

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
                mid = (end - start) / 2
                spline = self._spline_dirichlet(x.ravel(), y.ravel(), k=self.k, s=mid)
            else:
                start = mid
                mid = (end - start) / 2
                spline = self._spline_dirichlet(x.ravel(), y.ravel(), k=self.k, s=mid)
            step += 1
        spline = self._spline_dirichlet(x.ravel(), y.ravel(), k=self.k, s=end)
        if not self._is_convex(spline, x):
            print("Spline is decreasing but not convex.")
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
        return Params(all=[lambda_initial, e_0_initial])

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        params, _ = curve_fit(self._exp, base_signals.exposure_a_mutual_nn,
                              fit_result.ratio_a_b_mutual_nn_corrected,
                              p0=initial_params.kwargs.get('all'))

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


class ExpLinModel(BaseModel, ExpFamilyMixin):
    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        ratio_a_b_mutual_nn = np.divide(base_signals.a_mutual_nn, base_signals.b_mutual_nn)
        lambda_initial, e_0_initial = self._initial_fit(ratio_a_b_mutual_nn, base_signals.exposure_a_mutual_nn)
        return Params(all=[lambda_initial, e_0_initial, 0])

    def fit_and_correct(self, base_signals: BaseSignals, fit_result: FitResult,
                        initial_params: Params, method: CorrectionMethod) -> Tuple[Corrections, Params]:
        params, _ = curve_fit(self._exp_lin, base_signals.exposure_a_mutual_nn,
                              fit_result.ratio_a_b_mutual_nn_corrected,
                              p0=initial_params.kwargs.get('all'))

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
