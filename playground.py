import logging
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

from dslab_virgo_tsi.base import Result
from dslab_virgo_tsi.data_utils import normalize, unnormalize
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts


class LocalGPMerger:
    def __init__(self, result: Result):
        self.t_nn_merged, self.val_nn_merged = self._interleave(result.base_signals.t_a_nn,
                                                                result.base_signals.t_b_nn,
                                                                result.final.a_nn_corrected,
                                                                result.final.b_nn_corrected)
        logging.info("LocalGPMerger initialized.")

    @staticmethod
    @njit(cache=True)
    def _interleave(t_a, t_b, a, b):
        index_a = 0
        index_b = 0
        index_together = 0
        a_length = t_a.size
        b_length = t_b.size
        t_merged = np.empty((a_length + b_length,))
        val_merged = np.empty((a_length + b_length,))

        while index_together < a_length + b_length:
            while index_a < a_length and (t_a[index_a] <= t_b[index_b] or index_b == b_length):
                t_merged[index_together] = t_a[index_a]
                val_merged[index_together] = a[index_a]
                index_a += 1
                index_together += 1

            while index_b < b_length and (t_b[index_b] <= t_a[index_a] or index_a == a_length):
                t_merged[index_together] = t_b[index_b]
                val_merged[index_together] = b[index_b]
                index_b += 1
                index_together += 1

        return t_merged, val_merged

    def merge(self, t_target, window, points_in_window: int = 50, plot=False):
        logging.info(f"Merging for target time with shape {t_target.shape}, window span {window} "
                     f"(in days, each direction) and {points_in_window} points in each window.")

        return self._gp_target_time(self.t_nn_merged, self.val_nn_merged, t_target, window, points_in_window, plot)

    @staticmethod
    def _gp_target_time(t, x, t_target, window, points_in_window, plot):
        """
        :param t: Time of all signals.
        :param x: Signal.
        :param t_target: Time at which prediction should happen.
        :param window: How much time left/right of prediction time should be taken into account.
        :param points_in_window: How many points should be in window.
        :param plot: True if plot should be included.
        :return: NumPy array of predictions, same shape as t_target. Prediction at position i corresponds to time
            at position i in t_target. For windows without points np.NaN is returned.
        """
        t_length = t.shape[0]
        start_index = 0
        end_index = 0
        predictions = np.empty_like(t_target)

        # Iterate through target times
        for i in range(t_target.shape[0]):
            if i % 100 == 0 and i > 0:
                percentage = str(int(100 * i / predictions.size))
                logging.info(f"Merging currently at {percentage} %.")

            cur_target_t = t_target[i]

            # Determine points that fall within window
            win_beginning = cur_target_t - window
            win_end = cur_target_t + window
            while end_index < t_length and t[end_index] < win_end:
                end_index += 1

            while t[start_index] < win_beginning:
                start_index += 1

            cur_x = x[start_index:end_index]
            cur_t = t[start_index:end_index]

            # Downsampling
            if cur_x.size == 0:
                predictions[i] = np.NaN
                continue

            downsample_factor = int(np.ceil(cur_x.size / points_in_window))
            cur_x_down = np.copy(cur_x)[::downsample_factor]
            cur_t_down = np.copy(cur_t)[::downsample_factor]
            # sample_indices = np.random.choice(np.arange(cur_x.size), points_in_window, replace=False)
            # sample_indices = np.sort(sample_indices)
            # cur_x_down = cur_x[sample_indices]
            # cur_t_down = cur_t[sample_indices]

            # Normalize data
            x_mean = np.mean(cur_x_down)
            x_std = np.std(cur_x_down)
            t_mean = np.mean(cur_t_down)
            t_std = np.std(cur_t_down)
            if GPConsts.NORMALIZE:
                cur_t_down, cur_x_down = normalize(cur_t_down, t_mean, t_std), normalize(cur_x_down, x_mean, x_std)
                cur_target_t = normalize(cur_target_t, t_mean, t_std)
            else:
                cur_x_down -= x_mean

            # Clip values to -5 * std <= cur_x_down <= 5 * std
            clip_std = np.std(cur_x_down)
            clip_indices = np.logical_and(np.greater_equal(cur_x_down[:], -5 * clip_std),
                                          np.less_equal(cur_x_down[:], 5 * clip_std)).astype(np.bool).flatten()
            cur_t_down, cur_x_down = cur_t_down[clip_indices], cur_x_down[clip_indices]

            # Fit GP on transformed points
            scale = window / (t_std * 3)
            kernel = Matern(length_scale=scale, length_scale_bounds=(scale, scale), nu=GPConsts.MATERN_NU)

            kernel += WhiteKernel(GPConsts.WHITE_NOISE_LEVEL, GPConsts.WHITE_NOISE_LEVEL_BOUNDS)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=GPConsts.N_RESTARTS_OPTIMIZER)
            gpr.fit(cur_t_down.reshape(-1, 1), cur_x_down.reshape(-1, 1))

            # Predict value at target time
            prediction = gpr.predict(np.array([cur_target_t]).reshape(-1, 1)).item(0)

            # Project back
            if GPConsts.NORMALIZE:
                prediction = unnormalize(prediction, x_mean, x_std)
            else:
                prediction += x_mean

            # Store prediction
            predictions[i] = prediction

            if plot:
                # Evenly spaced points on which predictions are made
                if GPConsts.NORMALIZE:
                    win_beginning_norm = normalize(win_beginning, t_mean, t_std)
                    win_end_norm = normalize(win_end, t_mean, t_std)
                    time_to_predict = np.linspace(win_beginning_norm, win_end_norm, 1000)
                else:
                    time_to_predict = np.linspace(win_beginning, win_end, 1000)

                predict_values, sigma = gpr.predict(time_to_predict.reshape(-1, 1), return_std=True)

                # Project back
                if GPConsts.NORMALIZE:
                    time_to_predict = unnormalize(time_to_predict, t_mean, t_std)
                    predict_values = unnormalize(predict_values, x_mean, x_std)
                    cur_t_down, cur_x_down = unnormalize(cur_t_down, t_mean, t_std), unnormalize(cur_x_down, x_mean,
                                                                                                 x_std)
                    sigma *= x_std
                    cur_target_t = unnormalize(cur_target_t, t_mean, t_std)
                else:
                    predict_values += x_mean
                    cur_x_down += x_mean

                # Plot the function, the prediction and the 95% confidence interval
                sigma = sigma.reshape(-1, 1)
                plt.figure(5, figsize=(16, 8))
                # plt.plot(cur_t, cur_x, 'r.', markersize=1, label='Observations', alpha=0.1)
                # plt.plot(cur_t_down, cur_x_down, 'k.', markersize=5, label='Downscaled')
                plt.plot(time_to_predict, predict_values, 'b-', label='Prediction')
                plt.axvline(x=cur_target_t)
                plt.fill(np.concatenate([time_to_predict, time_to_predict[::-1]]),
                         np.concatenate([predict_values - 1.9600 * sigma,
                                         (predict_values + 1.9600 * sigma)[::-1]]),
                         alpha=0.5, fc='b', ec='None')

        if plot:
            plt.show()

        return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open('./results/2019-12-05_13-53-46_smooth_monotonic/SMOOTH_MONOTONIC_modeling_result.pkl', 'rb') as handle:
        res: Result = pickle.load(handle)

    merger = LocalGPMerger(res)
    # 0.5 value of window means half a day before prediction time and half a day after it
    # predictions_ = merger.merge(res.out.t_daily_out[300:1500:100], window=50, points_in_window=50, plot=True)
    predictions_ = merger.merge(res.out.t_daily_out[0::200], window=100, points_in_window=200, plot=True)
    print(predictions_)
