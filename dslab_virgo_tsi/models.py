import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splrep
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from dslab_virgo_tsi.base import BaseModel, BaseSignals, Params, FinalResult, FitResult


class ExpFamilyMixin:
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


class DegradationSpline:
    def __init__(self, k=3, steps=30):
        self.k = k
        self.sp = None
        self.steps = steps

    @staticmethod
    def _guess(x, y, k, s, w=None):
        """Do an ordinary spline fit to provide knots"""
        return splrep(x, y, w, k=k, s=s)

    @staticmethod
    def _err(c, x, y, t, k, w=None):
        """The error function to minimize"""
        diff = y - splev(x, (t, c, k))
        if w is None:
            diff = np.einsum('...i,...i', diff, diff)
        else:
            diff = np.dot(diff * diff, w)
        return np.abs(diff)

    def _spline_dirichlet(self, x, y, k=3, s=0.0, w=None):
        t, c0, k = self._guess(x, y, k, s, w=w)
        con = {'type': 'eq',
               'fun': lambda c: splev(0, (t, c, k), der=0) - 1,
               }
        opt = minimize(self._err, c0, (x, y, t, k, w), constraints=con)
        copt = opt.x
        return UnivariateSpline._from_tck((t, copt, k))

    @staticmethod
    def _is_decreasing(spline, x):
        spline_derivative = spline.derivative()
        return np.all(spline_derivative(x.ravel()) < 0)

    @staticmethod
    def _is_convex(spline, x):
        spline_derivative_2 = spline.derivative().derivative()
        return np.all(spline_derivative_2(x.ravel()) > 0)

    def _find_convex_decreasing_spline_binary_search(self, x, y):
        """
        for start it should be weird
        for end it should be decreasing
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
        self.sp = self._find_convex_decreasing_spline_binary_search(x, y)
        return self.sp

    def predict(self, x):
        return self.sp(x)


class ExpModel(BaseModel, ExpFamilyMixin):
    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        ratio_a_b_mutual_nn = np.divide(base_signals.a_mutual_nn, base_signals.b_mutual_nn)
        lambda_initial, e_0_initial = self._initial_fit(ratio_a_b_mutual_nn, base_signals.exposure_a_mutual_nn)
        return Params(all=[lambda_initial, e_0_initial])

    def fit_and_correct(self, base_signals: BaseSignals, initial_params: Params, ratio) -> FitResult:
        params, _ = curve_fit(self._exp, base_signals.exposure_a_mutual_nn, ratio,
                              p0=initial_params.kwargs.get('all'))
        a_corrected = np.divide(base_signals.a_mutual_nn, self._exp(base_signals.exposure_a_mutual_nn, *params))
        b_corrected = np.divide(base_signals.b_mutual_nn, self._exp(base_signals.exposure_b_mutual_nn, *params))

        return FitResult(a_corrected, b_corrected, Params(all=params))

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        optimal_params_list = optimal_params.kwargs.get('all')

        degradation_a_nn = self._exp(base_signals.exposure_a_nn, *optimal_params_list)
        degradation_b_nn = self._exp(base_signals.exposure_b_nn, *optimal_params_list)

        return FinalResult(base_signals, degradation_a_nn, degradation_b_nn)

    @staticmethod
    def _exp(x, lambda_, e_0):
        """Constrained exponential degradation model: y(0) = 1."""
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0))
        return y


class ExpLinModel(BaseModel, ExpFamilyMixin):
    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        ratio_a_b_mutual_nn = np.divide(base_signals.a_mutual_nn, base_signals.b_mutual_nn)
        lambda_initial, e_0_initial = self._initial_fit(ratio_a_b_mutual_nn, base_signals.exposure_a_mutual_nn)
        return Params(all=[lambda_initial, e_0_initial, 0])

    def fit_and_correct(self, base_signals: BaseSignals, initial_params: Params, ratio) -> FitResult:
        optimal_params, _ = curve_fit(self._exp_lin, base_signals.exposure_a_mutual_nn, ratio,
                                      p0=initial_params.kwargs.get('all'))
        a_corrected = np.divide(base_signals.a_mutual_nn, self._exp_lin(base_signals.exposure_a_mutual_nn,
                                                                        *optimal_params))
        b_corrected = np.divide(base_signals.b_mutual_nn, self._exp_lin(base_signals.exposure_b_mutual_nn,
                                                                        *optimal_params))

        return FitResult(a_corrected, b_corrected, Params(all=optimal_params))

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        optimal_params_list = optimal_params.kwargs.get('all')

        degradation_a_nn = self._exp_lin(base_signals.exposure_a_nn, *optimal_params_list)
        degradation_b_nn = self._exp_lin(base_signals.exposure_b_nn, *optimal_params_list)

        return FinalResult(base_signals, degradation_a_nn, degradation_b_nn)

    @staticmethod
    def _exp_lin(x, lambda_, e_0, linear):
        """Constrained exponential-linear degradation model: y(0) = 1."""
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0)) + linear * x
        return y


class SplineModel(BaseModel):
    def __init__(self, k=3, steps=30, thinning=100):
        self.k = k
        self.steps = steps
        self.thinning = thinning
        self.model = DegradationSpline(self.k, steps=self.steps)

    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        return Params()

    def fit_and_correct(self, base_signals: BaseSignals, initial_params: Params, ratio) -> FitResult:
        self.model.fit(base_signals.exposure_a_mutual_nn[::self.thinning], ratio[::self.thinning])
        a_corrected = np.divide(base_signals.a_mutual_nn, self.model.predict(base_signals.exposure_a_mutual_nn))
        b_corrected = np.divide(base_signals.b_mutual_nn, self.model.predict(base_signals.exposure_b_mutual_nn))

        return FitResult(a_corrected, b_corrected, Params(sp=self.model))

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        sp = optimal_params.kwargs.get('sp')
        return FinalResult(base_signals, sp.predict(base_signals.exposure_a_nn), sp.predict(base_signals.exposure_b_nn))


class IsotonicModel(BaseModel):
    def __init__(self, smoothing=False, y_max=1, y_min=0, increasing=False, out_of_bounds='clip', k=3, steps=30,
                 number_of_points=250):
        self.k = k
        self.steps = steps
        self.smoothing = smoothing
        self.number_of_points = number_of_points
        self.model = IsotonicRegression(y_max=y_max, y_min=y_min, increasing=increasing, out_of_bounds=out_of_bounds)

    def get_initial_params(self, base_signals: BaseSignals) -> Params:
        return Params()

    def fit_and_correct(self, base_signals: BaseSignals, initial_params: Params, ratio) -> FitResult:
        self.model.fit(base_signals.exposure_a_mutual_nn, ratio)
        if self.smoothing:
            max_exposure = base_signals.exposure_a_mutual_nn[-1]
            exposure = np.linspace(0, max_exposure, self.number_of_points)
            ratio = self.model.predict(exposure)
            self.model = DegradationSpline(self.k, steps=self.steps)
            self.model.fit(exposure, ratio)

        a_corrected = np.divide(base_signals.a_mutual_nn, self.model.predict(base_signals.exposure_a_mutual_nn))
        b_corrected = np.divide(base_signals.b_mutual_nn, self.model.predict(base_signals.exposure_b_mutual_nn))

        return FitResult(a_corrected, b_corrected, Params(model=self.model))

    def compute_final_result(self, base_signals: BaseSignals, optimal_params: Params) -> FinalResult:
        model = optimal_params.kwargs.get('model')
        return FinalResult(base_signals, model.predict(base_signals.exposure_a_nn),
                           model.predict(base_signals.exposure_b_nn))
