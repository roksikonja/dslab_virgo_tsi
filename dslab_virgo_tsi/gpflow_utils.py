import logging

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive
from gpflow.utilities.ops import square_distance


@tf.function(autograph=False)
def optimization_step(optimizer, model: gpflow.models.SVGP, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = - model.elbo(*batch)
        grads = tape.gradient(objective, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return objective


class SVGaussianProcess(object):

    @staticmethod
    def run_adam(model, iterations, train_dataset, minibatch_size):
        """
        Utility function running the Adam optimiser

        :param model: GPflow model
        :param iterations: number of iterations
        :param train_dataset: Tensorflow Dataset placeholder
        :param minibatch_size: size of mini batch
        """
        # Create an Adam Optimiser action
        logf = []
        train_it = iter(train_dataset.batch(minibatch_size))
        adam = tf.optimizers.Adam()
        step, elbo_step = None, None
        for step in range(iterations):
            elbo = - optimization_step(adam, model, next(train_it))
            if step % 10 == 0:
                elbo_step = elbo.numpy()
                if step % 1000 == 0:
                    logging.info("Step:\t{:<30}ELBO:\t{:>10}".format(step, elbo_step))

                logf.append(elbo_step)

        logging.info("Step:\t{:<30}ELBO:\t{:>10}".format(step, elbo_step))
        return logf


class VirgoWhiteKernel(gpflow.kernels.Kernel):
    def __init__(self, label_a, label_b):
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


class VirgoMatern32Kernel(gpflow.kernels.Kernel):
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
