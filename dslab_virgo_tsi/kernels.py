import numpy as np
from sklearn.gaussian_process.kernels import StationaryKernelMixin, Kernel, Hyperparameter
from typing import Tuple


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

    def __init__(self, mask, noise_level_a: float = 1.0, noise_level_b: float = 1.0,
                 noise_level_a_bounds: Tuple[float] = (1e-5, 1e5), noise_level_b_bounds: Tuple[float] = (1e-5, 1e5)):
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
