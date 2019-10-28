import datetime
import os

import numpy as np

from dslab_virgo_tsi.base import Result, FitResult, BaseSignals, FinalResult
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import make_dir
from dslab_virgo_tsi.visualizer import Visualizer

visualizer = Visualizer()
visualizer.set_figsize()


class SignalGenerator(object):

    def __init__(self, length, random_seed=0):
        np.random.seed(random_seed)

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
        x_ = 10 + a[0]/10 * np.sin(10 * np.pi * a[2] * (self.time - a[1])) + \
            (2 * int(a[4] >= 0.5) - 1) * a[3] * self.time

        self.x = x_
        self.a = a

    def generate_raw_signal(self, x_, random_seed, degradation_model="exp", rate=1.0):
        np.random.seed(random_seed)
        srange = x_.max() - x_.min()

        x_a, t_a = self.remove_measurements(x_.copy(), self.time.copy(), 0.1)
        x_b, t_b = self.remove_measurements(x_.copy(), self.time.copy(), 0.9)
        mean_b = float(np.mean(x_b[~np.isnan(x_b)]))

        exposure_a = self.compute_exposure(x_a, "sum", mean_b)
        exposure_b = self.compute_exposure(x_b, "sum", mean_b)

        x_a_raw_, x_b_raw_, params = self.degrade_signal(x_a, x_b, exposure_a, exposure_b,
                                                         degradation_model=degradation_model, rate=rate)

        x_a_raw_ = x_a_raw_ + self.generate_noise(x_a.shape, std=srange * 0.05)
        x_b_raw_ = x_b_raw_ + self.generate_noise(x_b.shape, std=srange * 0.05)

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
        if noise_type == "normal":
            noise = np.random.normal(0, std, shape)
        return noise

    @staticmethod
    def compute_exposure(x_, mode="measurements", mean=1.0):
        if mode == "measurements":
            x_ = np.nan_to_num(x_) > 0
        elif mode == "sum":
            x_ = np.nan_to_num(x_)
            x_ = x_ / mean

        x_ = x_ / x_.shape[0]
        return np.cumsum(x_)

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


def create_results_dir(model_type):
    results_dir = make_dir(os.path.join(Const.RESULTS_DIR,
                                        datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_{model_type}")))
    return results_dir


def plot_results(t_, x_, result_: Result, results_dir, model_name):
    before_fit: FitResult = result_.history_mutual_nn[0]
    last_iter: FitResult = result_.history_mutual_nn[-1]

    base_sig: BaseSignals = result_.base_signals
    final_res: FinalResult = result_.final

    print("plotting results ...")
    figs = []

    fig = visualizer.plot_signals(
        [
            (base_sig.t_a_nn, final_res.degradation_a_nn, "DEGRADATION_A", False),
            (base_sig.t_b_nn, final_res.degradation_b_nn, "DEGRADATION_B", False)
        ],
        results_dir, f"DEGRADATION_{model_name}", legend="upper right",
        x_label="t", y_label="d(t)")
    figs.append(fig)

    fig = visualizer.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.a_mutual_nn_corrected, "A_mutual_nn", False),
            (base_sig.t_mutual_nn, before_fit.b_mutual_nn_corrected, "B_mutual_nn", False),
            (base_sig.t_mutual_nn, last_iter.a_mutual_nn_corrected, "A_mutual_nn_corrected", False),
            (base_sig.t_mutual_nn, last_iter.b_mutual_nn_corrected, "B_mutual_nn_corrected", False),
            (t_, x_, "ground_truth", False),
        ],
        results_dir, f"{model_name}_mutual_corrected", legend="upper right",
        x_label="t", y_label="x(t)")
    figs.append(fig)

    fig = visualizer.plot_signals(
        [
            (base_sig.t_mutual_nn, before_fit.ratio_a_b_mutual_nn_corrected, f"RATIO_A_B_raw", False),
            (base_sig.t_mutual_nn, last_iter.ratio_a_b_mutual_nn_corrected, f"RATIO_A_not_B_corrected",
             False),
            (base_sig.t_mutual_nn, np.divide(last_iter.a_mutual_nn_corrected, last_iter.b_mutual_nn_corrected),
             f"RATIO_A_corrected_B_corrected", False)
             ],
        results_dir, f"{model_name}_RATIO_DEGRADATION_A_B_raw_corrected",
        legend="upper right", x_label="t", y_label="r(t)")
    figs.append(fig)

    fig = visualizer.plot_signals(
        [
            (base_sig.t_a_nn, base_sig.a_nn, "A_raw", False),
            (base_sig.t_b_nn, base_sig.b_nn, "B_raw", False),
            (base_sig.t_a_nn, final_res.a_nn_corrected, "A_raw_corrected", False),
            (base_sig.t_b_nn, final_res.b_nn_corrected, "B_raw_corrected", False),
            (t_, x_, "ground_truth", False),
        ],
        results_dir, f"{model_name}_A_B_raw_corrected_full",
        legend="upper right", x_label="t", y_label="x(t)")
    figs.append(fig)

    fig = visualizer.plot_signal_history(base_sig.t_mutual_nn, result_.history_mutual_nn,
                                         results_dir, f"{model_name}_history",
                                         ground_truth_triplet=(t_, x_, "ground_truth"),
                                         legend="upper right", x_label="t", y_label="x(t)")
    figs.append(fig)
