import math
from typing import Tuple

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive
from gpflow.utilities.ops import square_distance
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import kv, gamma
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process.kernels import StationaryKernelMixin, Kernel, Hyperparameter, _check_length_scale, RBF, \
    _approx_fprime

from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts


class DualWhiteKernel(StationaryKernelMixin, Kernel):
    """Dual White kernel.
    This is generalization of White Kernel to two separate noise levels.
    Parameters
    ----------
    noise_level_a : float, default: 1.0
        Parameter controlling the noise level of first signal (variance)
    noise_level_b : float, default: 1.0
        Parameter controlling the noise level of second signal (variance)
    noise_level_a_bounds : Tuple[float, float], default: (1e-5, 1e5)
        The lower and upper bound on noise_level_a
    noise_level_a_bounds : Tuple[float, float], default: (1e-5, 1e5)
        The lower and upper bound on noise_level_a
    """

    def __init__(self,
                 label_a=GPConsts.LABEL_A,
                 label_b=GPConsts.LABEL_B,
                 noise_level_a: float = GPConsts.WHITE_NOISE_LEVEL,
                 noise_level_b: float = GPConsts.WHITE_NOISE_LEVEL,
                 noise_level_a_bounds: Tuple[float] = GPConsts.WHITE_NOISE_LEVEL_BOUNDS,
                 noise_level_b_bounds: Tuple[float] = GPConsts.WHITE_NOISE_LEVEL_BOUNDS):
        self.label_a = label_a
        self.label_b = label_b
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
        X_values = X[:, 0].reshape(-1, 1)
        X_mask = X[:, 1].reshape(-1, 1)

        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            diag_a = (X_mask == self.label_a) * self.noise_level_a
            diag_b = (X_mask == self.label_b) * self.noise_level_b
            K1 = np.diag(np.squeeze(diag_a))
            K2 = np.diag(np.squeeze(diag_b))
            K = K1 + K2
            if eval_gradient:
                if not self.hyperparameter_noise_level_a.fixed and not self.hyperparameter_noise_level_b.fixed:
                    return K, np.dstack((K1[:, :, np.newaxis], K2[:, :, np.newaxis]))
                else:
                    return K, np.dstack((np.empty((X_values.shape[0], X_values.shape[0], 0)),
                                         np.empty((X_values.shape[0], X_values.shape[0], 0))))
            else:
                return K
        else:
            Y_values = Y[:, 0].reshape(-1, 1)
            return np.zeros((X_values.shape[0], Y_values.shape[0]))

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
        X_mask = X[:, 1].reshape(-1, 1)
        diag_a = (X_mask == self.label_a) * self.noise_level_a
        diag_b = (X_mask == self.label_b) * self.noise_level_b
        return np.squeeze(diag_a + diag_b)


class DualMatern(RBF):
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
        X_values = X[:, 0].reshape(-1, 1)

        length_scale = _check_length_scale(X_values, self.length_scale)
        if Y is None:
            dists = pdist(X_values / length_scale, metric='euclidean')
        else:
            Y_values = Y[:, 0].reshape(-1, 1)
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X_values / length_scale, Y_values / length_scale,
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
                K_gradient = np.empty((X_values.shape[0], X_values.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X_values[:, np.newaxis, :] - X_values[np.newaxis, :, :]) ** 2 \
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
                    return self.clone_with_theta(theta)(X_values, Y_values)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K


class VirgoWhiteKernel(gpflow.kernels.Kernel):
    def __init__(self, label_a=GPConsts.LABEL_A, label_b=GPConsts.LABEL_B):
        super().__init__(active_dims=[0, 1])
        self.variance_a = gpflow.Parameter(1.0, transform=positive())
        self.variance_b = gpflow.Parameter(1.0, transform=positive())
        self.label_a = label_a
        self.label_b = label_b

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            d_a = tf.fill((X.shape[0],), tf.squeeze(self.variance_a))
            d_b = tf.fill((X.shape[0],), tf.squeeze(self.variance_b))
            indices_a = tf.cast(tf.equal(X[:, 1], self.label_a), dtype=tf.float64)
            indices_b = tf.cast(tf.equal(X[:, 1], self.label_b), dtype=tf.float64)
            a = tf.linalg.diag(tf.multiply(d_a, indices_a) + tf.multiply(d_b, indices_b))
            return a
        else:
            shape = [X.shape[0], X2.shape[0]]
            return tf.zeros(shape, dtype=X.dtype)

    def K_diag(self, X, presliced=None):
        d_a = tf.fill((X.shape[0],), tf.squeeze(self.variance_a))
        d_b = tf.fill((X.shape[0],), tf.squeeze(self.variance_b))
        indices_a = tf.cast(tf.equal(X[:, 1], self.label_a), dtype=tf.float64)
        indices_b = tf.cast(tf.equal(X[:, 1], self.label_b), dtype=tf.float64)
        a = tf.multiply(d_a, indices_a) + tf.multiply(d_b, indices_b)
        return a


class VirgoMatern12Kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=[0, 1])
        self.variance = gpflow.Parameter(1.0, transform=positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=positive())

    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Returns ||(X - X2ᵀ) / ℓ||² i.e. squared L2-norm.
        """
        X_scaled = X / self.lengthscale
        X2_scaled = X2 / self.lengthscale if X2 is not None else X2
        return square_distance(X_scaled, X2_scaled)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        X = tf.reshape(X[:, 0], [-1, 1])
        X2 = tf.reshape(X2[:, 0], [-1, 1])
        r2 = self.scaled_squared_euclid_dist(X, X2)
        k = self.K_r2(r2)
        return k

    def K_diag(self, X, presliced=False):
        k_diag = tf.fill((X.shape[0],), tf.squeeze(self.variance))
        return k_diag

    def K_r(self, r):
        return self.variance * tf.exp(-r)

    def K_r2(self, r2):
        """
        Returns the kernel evaluated on r² (`r2`), which is the squared scaled Euclidean distance
        Should operate element-wise on r²
        """
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-40))
            return self.K_r(r)  # pylint: disable=no-member

        raise NotImplementedError


class VirgoMatern32Kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=slice(0, 2, 1))
        self.variance = gpflow.Parameter(1.0, transform=positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=positive())

    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Returns ||(X - X2ᵀ) / ℓ||² i.e. squared L2-norm.
        """
        X_scaled = X / self.lengthscale
        X2_scaled = X2 / self.lengthscale if X2 is not None else X2
        return square_distance(X_scaled, X2_scaled)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        X = tf.reshape(X[:, 0], [-1, 1])
        X2 = tf.reshape(X2[:, 0], [-1, 1])
        r2 = self.scaled_squared_euclid_dist(X, X2)
        k = self.K_r2(r2)
        return k

    def K_diag(self, X, presliced=False):
        k_diag = tf.fill((X.shape[0],), tf.squeeze(self.variance))
        return k_diag

    def K_r(self, r):
        sqrt3 = np.sqrt(3.)
        return self.variance * (1. + sqrt3 * r) * tf.exp(-sqrt3 * r)

    def K_r2(self, r2):
        """
        Returns the kernel evaluated on r² (`r2`), which is the squared scaled Euclidean distance
        Should operate element-wise on r²
        """
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-40))
            return self.K_r(r)  # pylint: disable=no-member

        raise NotImplementedError


class Kernels:
    # scikit-learn kernels
    matern_kernel = Matern(length_scale=GPConsts.MATERN_LENGTH_SCALE,
                           length_scale_bounds=GPConsts.MATERN_LENGTH_SCALE_BOUNDS,
                           nu=GPConsts.MATERN_NU)

    white_kernel = WhiteKernel(noise_level=GPConsts.WHITE_NOISE_LEVEL,
                               noise_level_bounds=GPConsts.WHITE_NOISE_LEVEL_BOUNDS)

    dual_matern_kernel = DualMatern()
    dual_white_kernel = DualWhiteKernel()

    # gpflow kernels
    gpf_matern32 = gpflow.kernels.Matern32()
    gpf_matern12 = gpflow.kernels.Matern12()
    gpf_white = gpflow.kernels.White()

    gpf_dual_matern32 = VirgoMatern32Kernel()
    gpf_dual_matern12 = VirgoMatern12Kernel()
    gpf_dual_white = VirgoWhiteKernel()
