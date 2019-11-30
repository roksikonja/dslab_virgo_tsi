import pickle
import time

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from gpflow.utilities import positive
from gpflow.utilities import print_summary
from gpflow.utilities.ops import square_distance

plt.style.use('ggplot')
path_to_pkl = './results/2019-11-04_12-20-59_smooth_monotonic/smooth_monotonic_modeling_result.pkl'


# Kernel classes
class VirgoWhiteKernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=slice(0, 2, 1))
        self.variance_a = gpflow.Parameter(1.0, transform=positive())
        self.variance_b = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            d_a = tf.fill((X.shape[0],), tf.squeeze(self.variance_a))
            d_b = tf.fill((X.shape[0],), tf.squeeze(self.variance_b))
            indices_a = tf.cast(tf.equal(X[:, 1], 0), dtype=tf.float64)
            indices_b = tf.cast(tf.equal(X[:, 1], 1), dtype=tf.float64)
            a = tf.linalg.diag(tf.multiply(d_a, indices_a) + tf.multiply(d_b, indices_b))
            # print(a.shape)
            return a
        else:
            shape = [X.shape[0], X2.shape[0]]
            # print(shape)
            return tf.zeros(shape, dtype=X.dtype)

    def K_diag(self, X, presliced=None):
        d_a = tf.fill((X.shape[0],), tf.squeeze(self.variance_a))
        d_b = tf.fill((X.shape[0],), tf.squeeze(self.variance_b))
        indices_a = tf.cast(tf.equal(X[:, 1], 0), dtype=tf.float64)
        indices_b = tf.cast(tf.equal(X[:, 1], 1), dtype=tf.float64)
        a = tf.multiply(d_a, indices_a) + tf.multiply(d_b, indices_b)
        # print(a.shape)
        return a


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


class VirgoMatern12Kernel(gpflow.kernels.Kernel):
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


with open(path_to_pkl, 'rb') as handle:
    x = pickle.load(handle)

x_subsample_a = x.final.a_nn_corrected[::300]
time_subsample_a = x.base_signals.t_a_nn[::300]

x_subsample_b = x.final.b_nn_corrected[:]
time_subsample_b = x.base_signals.t_b_nn[:]

mean = np.mean(np.concatenate((x.final.b_nn_corrected, x.final.a_nn_corrected), axis=0))
std = np.std(np.concatenate((x.final.b_nn_corrected, x.final.a_nn_corrected), axis=0))

mean_t = np.mean(np.concatenate((x.base_signals.t_b_nn, x.base_signals.t_a_nn), axis=0))
std_t = np.std(np.concatenate((x.base_signals.t_b_nn, x.base_signals.t_a_nn), axis=0))

X_a, Y_a = (time_subsample_a - mean_t) / std_t, (x_subsample_a - mean) / std
X_a, Y_a = X_a[np.greater_equal(Y_a, -4)], Y_a[np.greater_equal(Y_a, -4)]
X_a, Y_a = X_a[np.less_equal(Y_a, 4)], Y_a[np.less_equal(Y_a, 4)]
X_a = np.stack((X_a, np.ones(X_a.shape)))

X_b, Y_b = (time_subsample_b - mean_t) / std_t, (x_subsample_b - mean) / std
X_b, Y_b = X_b[np.greater_equal(Y_b, -4)], Y_b[np.greater_equal(Y_b, -4)]
X_b, Y_b = X_b[np.less_equal(Y_b, 4)], Y_b[np.less_equal(Y_b, 4)]
X_b = np.stack((X_b, np.zeros(X_b.shape)))

X = np.transpose(np.concatenate((X_a, X_b), axis=1)).reshape(-1, 2)
Y = np.transpose(np.concatenate((Y_a, Y_b), axis=0)).reshape(-1, 1)

N = X.shape[0]

number_of_induced_points = 200
interval = (np.min(X[:, 0]), np.max(X[:, 0]))
time_indicies = np.linspace(interval[0], interval[1], number_of_induced_points)


def find_nearest(array, values):
    indices_ = np.zeros(values.shape)
    for index, value in enumerate(values):
        indices_[index] = np.abs(array - value).argmin()
    return indices_


indices = find_nearest(X[:, 0], time_indicies).astype(int)
Z = X[indices, :]

kernel = gpflow.kernels.Sum([VirgoMatern12Kernel(), VirgoWhiteKernel()])
m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

print_summary(m)

log_likelihood = tf.function(autograph=False)(m.log_likelihood)

train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)) \
    .repeat() \
    .shuffle(N)


def plot(title=''):
    plt.figure(figsize=(16, 8))
    plt.title(title)
    p_x = np.linspace(-2, 2, 100)  # Test locations
    p_x = np.transpose(np.stack((p_x, 0 * np.ones(100))))
    p_y, p_y_v = m.predict_y(p_x)  # Predict Y values at test locations
    plt.plot(X[:, 0], Y, 'x', label='Training points', alpha=0.2)
    line, = plt.plot(p_x[:, 0], p_y, lw=1.5, label='Mean of predictive posterior')
    col = line.get_color()
    plt.fill_between(p_x[:, 0], (p_y - 2 * p_y_v ** 0.5)[:, 0], (p_y + 2 * p_y_v ** 0.5)[:, 0],
                     color=col, alpha=0.6, lw=1.5)
    Z = m.inducing_variable.Z.numpy()
    print(Z.shape)
    plt.plot(Z, np.zeros_like(Z), 'k|', mew=2, label='Inducing locations')
    plt.legend(loc='lower right')
    plt.ylim(-3, 3)
    plt.show()


plot(title="Predictions before training")

minibatch_size = 1000

# We turn of training for inducing point locations
gpflow.utilities.set_trainable(m.inducing_variable, False)


@tf.function(autograph=False)
def optimization_step(optimizer, model: gpflow.models.SVGP, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = - model.elbo(*batch)
        grads = tape.gradient(objective, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return objective


def run_adam(model, iterations):
    """
    Utility function running the Adam optimiser

    :param model: GPflow model
    :param iterations: number of iterations
    """
    # Create an Adam Optimiser action
    logf_ = []
    train_it = iter(train_dataset.batch(minibatch_size))
    adam = tf.optimizers.Adam()
    for step in range(iterations):
        elbo = - optimization_step(adam, model, next(train_it))
        if step % 1000 == 0:
            print('We are on the step {}'.format(step))
            print("Step:\t{:<30}ELBO:\t{:>10}".format(step, elbo.numpy()))
        if step % 10 == 0:
            logf_.append(elbo.numpy())
    return logf_


start = time.time()

maxiter = ci_niter(16000)

logf = run_adam(m, maxiter)
plt.plot(np.arange(maxiter)[::10], logf)
plt.xlabel('iteration')
plt.ylabel('ELBO')
plt.show()

end = time.time()
print(end - start)

plot("Predictions after training")
print_summary(m)
