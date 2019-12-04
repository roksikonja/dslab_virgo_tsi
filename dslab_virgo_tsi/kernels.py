import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive
from gpflow.utilities.ops import square_distance
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts


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

    # gpflow kernels
    gpf_matern32 = gpflow.kernels.Matern32()
    gpf_matern12 = gpflow.kernels.Matern12()
    gpf_white = gpflow.kernels.White()

    gpf_dual_matern12 = VirgoMatern12Kernel()
    gpf_dual_white = VirgoWhiteKernel()
