import math
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import kv, gamma
from sklearn.gaussian_process.kernels import StationaryKernelMixin, Kernel, Hyperparameter, RBF
from sklearn.gaussian_process.kernels import _approx_fprime, _check_length_scale

from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts


class DualWhiteKernel(StationaryKernelMixin, Kernel):
    """Dual White kernel.
    This is generalization of White Kernel to two separate noise levels.
    Parameters
    ----------
    mask : array_like
        Value True at position i denotes b signal and value False a signal.
    noise_level_a : float, default: 1.0
        Parameter controlling the noise level of first signal (variance)
    noise_level_b : float, default: 1.0
        Parameter controlling the noise level of second signal (variance)
    noise_level_a_bounds : Tuple[float], default: (1e-5, 1e5)
        The lower and upper bound on noise_level_a
    noise_level_a_bounds : Tuple[float], default: (1e-5, 1e5)
        The lower and upper bound on noise_level_a
    """

    def __init__(self, mask,
                 noise_level_a: float = GPConsts.WHITE_NOISE_LEVEL,
                 noise_level_b: float = GPConsts.WHITE_NOISE_LEVEL,
                 noise_level_a_bounds: Tuple[float] = GPConsts.WHITE_NOISE_LEVEL_BOUNDS,
                 noise_level_b_bounds: Tuple[float] = GPConsts.WHITE_NOISE_LEVEL_BOUNDS):
        self.mask = mask
        self.noise_level_a = noise_level_a
        self.noise_level_b = noise_level_b
        self.noise_level_a_bounds = noise_level_a_bounds
        self.noise_level_b_bounds = noise_level_b_bounds
        self.diag_K = self.noise_level_b * self.mask + self.noise_level_a * np.logical_not(self.mask)

    @property
    def hyperparameter_noise_level_a(self):
        return Hyperparameter(
            "noise_level_a", "numeric", self.noise_level_a_bounds)

    @property
    def hyperparameter_noise_level_b(self):
        return Hyperparameter(
            "noise_level_b", "numeric", self.noise_level_b_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag_K)
            if eval_gradient:
                if not self.hyperparameter_noise_level_a.fixed and not self.hyperparameter_noise_level_b.fixed:
                    return K, K[:, :, np.newaxis]
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.diag_K


class DualWhiteKernelTest(StationaryKernelMixin, Kernel):
    """Dual White kernel.
    This is generalization of White Kernel to two separate noise levels.
    Parameters
    ----------
    noise_level_a : float, default: 1.0
        Parameter controlling the noise level of first signal (variance)
    noise_level_b : float, default: 1.0
        Parameter controlling the noise level of second signal (variance)
    noise_level_a_bounds : Tuple[float], default: (1e-5, 1e5)
        The lower and upper bound on noise_level_a
    noise_level_a_bounds : Tuple[float], default: (1e-5, 1e5)
        The lower and upper bound on noise_level_a
    """

    def __init__(self,
                 noise_level_a: float = GPConsts.WHITE_NOISE_LEVEL,
                 noise_level_b: float = GPConsts.WHITE_NOISE_LEVEL,
                 noise_level_a_bounds: Tuple[float] = GPConsts.WHITE_NOISE_LEVEL_BOUNDS,
                 noise_level_b_bounds: Tuple[float] = GPConsts.WHITE_NOISE_LEVEL_BOUNDS):

        self.noise_level_a = noise_level_a
        self.noise_level_b = noise_level_b
        self.noise_level_a_bounds = noise_level_a_bounds
        self.noise_level_b_bounds = noise_level_b_bounds

    @property
    def hyperparameter_noise_level_a(self):
        return Hyperparameter(
            "noise_level_a", "numeric", self.noise_level_a_bounds)

    @property
    def hyperparameter_noise_level_b(self):
        return Hyperparameter(
            "noise_level_b", "numeric", self.noise_level_b_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        mask = X[:, ]
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag_K)
            if eval_gradient:
                if not self.hyperparameter_noise_level_a.fixed and not self.hyperparameter_noise_level_b.fixed:
                    return K, K[:, :, np.newaxis]
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))


class Matern(RBF):
    """ Matern kernel.

    The class of Matern kernels is a generalization of the RBF and the
    absolute exponential kernel parameterized by an additional parameter
    nu. The smaller nu, the less smooth the approximated function is.
    For nu=inf, the kernel becomes equivalent to the RBF kernel and for nu=0.5
    to the absolute exponential kernel. Important intermediate values are
    nu=1.5 (once differentiable functions) and nu=2.5 (twice differentiable
    functions).

    See Rasmussen and Williams 2006, pp84 for details regarding the
    different variants of the Matern kernel.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale

    nu : float, default: 1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 nu=1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric='euclidean')
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='euclidean')

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * self.nu) * K)
            K.fill((2 ** (1. - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
            else:
                D = squareform(dists ** 2)[:, :, np.newaxis]

            if self.nu == 0.5:
                K_gradient = K[..., np.newaxis] * D \
                             / np.sqrt(D.sum(2))[:, :, np.newaxis]
                K_gradient[~np.isfinite(K_gradient)] = 0
            elif self.nu == 1.5:
                K_gradient = \
                    3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu)
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0],
                self.nu)
