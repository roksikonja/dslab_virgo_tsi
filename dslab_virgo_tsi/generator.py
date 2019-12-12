import logging

import numpy as np
from scipy.stats import norm

from dslab_virgo_tsi.base import ExposureMethod
from dslab_virgo_tsi.model_constants import GeneratorConstants as GenConsts


class SignalGenerator(object):

    def __init__(self, length, random_seed=0, exposure_method: ExposureMethod = ExposureMethod.NUM_MEASUREMENTS):
        np.random.seed(random_seed)

        self.length = length
        self.time = self.generate_time()

        # Exposure method
        if exposure_method == "measurements":
            exposure_method = ExposureMethod.NUM_MEASUREMENTS
        else:
            exposure_method = ExposureMethod.EXPOSURE_SUM

        self.exposure_method = exposure_method

        # Ground truth signal
        self.x = None
        self.a = None
        self.generate_signal()

    def generate_time(self):
        return np.linspace(0, 1, self.length)

    def generate_signal(self):
        """
        Generates a signal given by random parameters a.
        :return: x(t) = 10 + sin(10 * pi * a[2] * (t - a[1])) + (2 * int(a[4] >= 0.5) - 1) * a[3] * t.
        """
        a = np.random.rand(5)
        # x_ = 10 + a[0] / 10 * np.sin(10 * np.pi * a[2] * (self.time - a[1])) + \
        #     (2 * int(a[4] >= 0.5) - 1) * a[3] * self.time
        x_ = 10

        dt = self.time[1] - self.time[0]
        x_ = x_ + self.brownian(0, self.time.shape[0], dt, 5)

        self.x = x_
        self.a = a

    @staticmethod
    def brownian(x0, n, dt, delta):
        """
        Generate Brownian motion.
        :param x0: initial value
        :param n: length
        :param dt: time step
        :param delta: unit step variance
        :return: array following Brownian motion
        """
        x0 = np.asarray(x0)

        r = norm.rvs(size=x0.shape + (n,), scale=delta * np.sqrt(dt))

        b = np.cumsum(r, axis=-1)

        return b

    def generate_raw_signal(self, x_, random_seed, degradation_model="exp", rate=1.0):
        np.random.seed(random_seed)
        srange = x_.max() - x_.min()

        x_a, t_a = self.remove_measurements(x_.copy(), self.time.copy(), 0.1)
        x_b, t_b = self.remove_measurements(x_.copy(), self.time.copy(), 0.9)

        mean_b = float(np.mean(x_b[~np.isnan(x_b)]))
        length_a = x_a.shape[0]

        exposure_a = self.compute_exposure(x_a, mode=self.exposure_method, mean=mean_b, length=length_a)
        exposure_b = self.compute_exposure(x_b, mode=self.exposure_method, mean=mean_b, length=length_a)

        x_a_raw_, x_b_raw_, params = self.degrade_signal(x_a, x_b, exposure_a, exposure_b,
                                                         degradation_model=degradation_model, rate=rate)

        noise_std_a = srange * 0.05 / 2
        noise_std_b = srange * 0.03 / 2

        x_a_raw_ = x_a_raw_ + self.generate_noise(x_a.shape, std=noise_std_a)
        x_b_raw_ = x_b_raw_ + self.generate_noise(x_b.shape, std=noise_std_b)

        logging.info("Generator noise variance A:\t{:>10}".format(noise_std_a ** 2))
        logging.info("Generator noise variance B:\t{:>10}".format(noise_std_b ** 2))

        return x_a_raw_, x_b_raw_, params

    @staticmethod
    def remove_measurements(x_, t_, rate):
        # Remove rate-fraction of measurements
        nan_indices = np.random.rand(*x_.shape) <= rate
        x_[nan_indices] = np.nan
        t_[nan_indices] = np.nan
        return x_, t_

    @staticmethod
    def generate_noise(shape, noise_type="normal", std=1.0):
        noise = None
        if std == 0:
            return np.zeros(shape)

        if noise_type == "normal":
            noise = np.random.normal(0, std, shape)
        return noise

    @staticmethod
    def compute_exposure(x_, mode: ExposureMethod, mean=1.0, length=None):
        if mode == ExposureMethod.NUM_MEASUREMENTS:
            x_ = np.nan_to_num(x_) > 0
        elif mode == ExposureMethod.EXPOSURE_SUM:
            x_ = np.nan_to_num(x_)
            x_ = x_ / mean

        if length:
            x_ = x_ / length
        return np.cumsum(x_)

    @staticmethod
    def degrade_signal(x_a, x_b, exposure_a, exposure_b, degradation_model="exp", rate=2.0):
        degradation_a = None
        degradation_b = None
        params = None
        if degradation_model == "exp":
            params = np.random.rand(2)
            params[0] *= 2
            params[1] = np.random.uniform(0.2, 1.0)
            degradation_a = (1 - params[1]) * np.exp(- 10 * rate * params[0] * exposure_a) + params[1]
            degradation_b = (1 - params[1]) * np.exp(- 10 * rate * params[0] * exposure_b) + params[1]

        return x_a * degradation_a, x_b * degradation_b, params


if __name__ == "__main__":
    generator = SignalGenerator(length=GenConsts.SIGNAL_LENGTH,
                                random_seed=0,
                                exposure_method=ExposureMethod.EXPOSURE_SUM)

    t = generator.time
    x = generator.x
