from data_utils import make_dir
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
from constants import Constants as C
import pandas as pd
import os
import datetime
import argparse


class SignalGenerator(object):

    def __init__(self, length):
        np.random.seed(0)

        self.length = length
        self.time = self.generate_time()

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
        x = 10 + a[0]/10 * np.sin(10 * np.pi * a[2] * (self.time - a[1])) + (2 * int(a[4] >= 0.5) - 1) * a[3] * self.time

        self.x = x
        self.a = a

    def generate_raw_signal(self, x, random_seed, degradation_model="exp", rate=1.0):
        np.random.seed(random_seed)
        srange = x.max() - x.min()

        x_a = self.remove_measurements(x.copy(), 0.1)
        x_b = self.remove_measurements(x.copy(), 0.9)
        mean_b = float(np.mean(x_b[~np.isnan(x_b)]))

        exposure_a = self.compute_exposure(x_a, "sum", mean_b)
        exposure_b = self.compute_exposure(x_b, "sum", mean_b)

        x_a_raw, x_b_raw, params = self.degrade_signal(x_a, x_b, exposure_a, exposure_b,
                                                       degradation_model=degradation_model, rate=rate)

        x_a_raw = x_a_raw + self.generate_noise(x_a.shape, std=srange * 0.05)
        x_b_raw = x_b_raw + self.generate_noise(x_b.shape, std=srange * 0.05)

        return x_a_raw, x_b_raw, exposure_a, exposure_b, params

    @staticmethod
    def remove_measurements(x, rate):
        # Remove rate-fraction of measurements
        nan_indices = np.random.rand(*x.shape) <= rate
        x[nan_indices] = np.nan
        return x

    @staticmethod
    def generate_noise(shape, noise_type="normal", std=1.0):
        noise = None
        if noise_type == "normal":
            noise = np.random.normal(0, std, shape)
        return noise

    @staticmethod
    def compute_exposure(x, mode="measurements", mean=1.0):
        if mode == "measurements":
            x = np.nan_to_num(x) > 0
        elif mode == "sum":
            x = np.nan_to_num(x)
            x = x / mean

        x = x / x.shape[0]
        return np.cumsum(x)

    @staticmethod
    def degrade_signal(x_a, x_b, exposure_a, exposure_b, degradation_model="exp", rate=1.0):
        degradation_a = None
        degradation_b = None
        params = None
        if degradation_model == "exp":
            params = np.random.rand(2)
            params[1] = np.random.uniform(0.2, 1)
            degradation_a = (1 - params[1]) * np.exp(- 10 * rate * params[0] * exposure_a) + params[1]
            degradation_b = (1 - params[1]) * np.exp(- 10 * rate * params[0] * exposure_b) + params[1]

        return x_a * degradation_a, x_b * degradation_b, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation_model", type=str, default="exp", help="Model to train.")
    parser.add_argument("--degradation_rate", type=float, default=1.0, help="Tuning parameter for degradation.")
    ARGS = parser.parse_args()

    # Constants
    data_dir = C.DATA_DIR
    results_dir = C.RESULTS_DIR
    results_dir = os.path.join(results_dir, datetime.datetime.now().strftime("modeling_%Y-%m-%d"))
    make_dir(results_dir)

    style.use(C.MATPLOTLIB_STYLE)
    matplotlib.rcParams['lines.linewidth'] = C.MATPLOTLIB_STYLE_LINEWIDTH
    signal_length = int(1e5)

    # Parameters
    degradation_model = ARGS.degradation_model
    degradation_rate = ARGS.degradation_rate

    # Generator
    Generator = SignalGenerator(signal_length)
    t = Generator.time
    data = pd.DataFrame()
    data["t"] = t

    for s_idx in range(5):
        x = Generator.x
        data["x-{}".format(s_idx)] = x

        for i in range(5):
            x_a_raw, x_b_raw, exposure_a, exposure_b, params = Generator.generate_raw_signal(x, 13 * i + s_idx * 17,
                                                                                             rate=degradation_rate)

            data["x_a-{}-{}".format(s_idx, i)] = x_a_raw
            data["x_b-{}-{}".format(s_idx, i)] = x_b_raw
            data["exposure_a-{}-{}".format(s_idx, i)] = exposure_a
            data["exposure_b-{}-{}".format(s_idx, i)] = exposure_b

            plt.figure(i+1, figsize=(12, 6))
            plt.plot(t, x, color="black", label="x")
            plt.plot(t, x_a_raw, label="x_a-{}".format(i+1))
            plt.plot(t, x_b_raw, label="x_b-{}".format(i+1))
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(results_dir,  "synthetic_{}_{}_s{}-{}.png".format(degradation_model,
                                                                                       signal_length, s_idx, i)),
                        bbox_inches="tight", quality=100, dpi=200)
            plt.clf()
            plt.close()

        Generator.generate_signal()
    # data.to_pickle(os.path.join(data_dir, "synthetic_{}_{}.pkl".format(degradation_model, signal_length)))
