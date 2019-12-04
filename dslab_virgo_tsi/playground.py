import logging
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel

from dslab_virgo_tsi.base import Result
from dslab_virgo_tsi.data_utils import normalize, unnormalize
from dslab_virgo_tsi.kernels import MaternT
from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts


# from sklearn.gaussian_process.kernels import Matern, WhiteKernel


# @njit(cache=True)
# def interleave(t_a_, t_b_, a_, b_):
#     index_a = 0
#     index_b = 0
#     index_together = 0
#     a_length = t_a_.size
#     b_length = t_b_.size
#     t_together_ = np.empty((a_length + b_length,))
#     values_together_ = np.empty((a_length + b_length,))
#     mask_ = np.empty((a_length + b_length))
#
#     # 0 = a, 1 = b
#     while index_together < a_length + b_length:
#         while index_a < a_length and (t_a_[index_a] <= t_b_[index_b] or index_b == b_length):
#             t_together_[index_together] = t_a_[index_a]
#             values_together_[index_together] = a_[index_a]
#             mask_[index_together] = 0
#             index_a += 1
#             index_together += 1
#
#         while index_b < b_length and (t_b_[index_b] <= t_a_[index_a] or index_a == a_length):
#             t_together_[index_together] = t_b_[index_b]
#             values_together_[index_together] = b_[index_b]
#             mask_[index_together] = 1
#             index_b += 1
#             index_together += 1
#
#     return t_together_, values_together_, mask_
#
#
# def gp_target_time(t_, val_, mask_, t_target_, window_: float = 10, points_in_window=50):
#     """
#     :param t_: Time of all signals.
#     :param val_: Signal.
#     :param mask_: Mask.
#     :param t_target_: Time.
#     :param window_: How much time left/right of prediction time should be taken into account.
#     :param points_in_window: How many points should be in left/right of window.
#     :return:
#     """
#     t_ = t_.reshape(-1, 1)
#     val_ = val_.reshape(-1, 1)
#     mask_ = mask_.reshape(-1, 1)
#
#     # TODO
#     t_target_ = t_target_[20:]
#     t_target_ = t_target_.reshape(-1, 1)
#
#     t_length = t_.shape[0]
#     start_index = 0
#     end_index = 0
#
#     for i in range(t_target_.shape[0]):
#         cur_target_t = t_target_[i]
#
#         while end_index < t_length and t_[end_index] < cur_target_t + window_:
#             end_index += 1
#
#         while t_[start_index] < cur_target_t - window_:
#             start_index += 1
#
#         cur_values = val_[start_index:end_index]
#         cur_t = t_[start_index:end_index]
#         cur_mask = mask_[start_index:end_index]
#
#         # Downscaling
#         ratio_too_much = int(np.ceil(cur_values.size / points_in_window))
#         print(cur_values.size)
#         print(points_in_window)
#         print("Ratio: ", ratio_too_much)
#         # cur_values_down = cur_values[::ratio_too_much]
#         # cur_t_down = cur_t[::ratio_too_much]
#         # cur_mask_down = cur_mask[::ratio_too_much]
#
#         cur_values_a = cur_values[cur_mask == 0]
#         cur_values_b = cur_values[cur_mask == 1]
#         cur_t_a = cur_t[cur_mask == 0]
#         cur_t_b = cur_t[cur_mask == 1]
#         cur_mask_a = cur_mask[cur_mask == 0]
#         cur_mask_b = cur_mask[cur_mask == 1]
#
#         samples_a = min(cur_values_a.size, points_in_window)
#         samples_b = min(cur_values_b.size, points_in_window)
#         indices_a = np.random.choice(np.arange(cur_values_a.size), samples_a, False)
#         indices_b = np.random.choice(np.arange(cur_values_b.size), samples_b, False)
#
#         cur_values_down = np.concatenate((cur_values_a[indices_a], cur_values_b[indices_b])).reshape(-1, 1)
#         cur_t_down = np.concatenate((cur_t_a[indices_a], cur_t_b[indices_b])).reshape(-1, 1)
#         cur_mask_down = np.concatenate((cur_mask_a[indices_a], cur_mask_b[indices_b])).reshape(-1, 1)
#         # cur_values_down = np.concatenate((cur_values_a[::ratio_too_much], cur_values_b)).reshape(-1, 1)
#         # cur_t_down = np.concatenate((cur_t_a[::ratio_too_much], cur_t_b)).reshape(-1, 1)
#         # cur_mask_down = np.concatenate((cur_mask_a[::ratio_too_much], cur_mask_b)).reshape(-1, 1)
#         # print("How many b?: ", np.sum(cur_mask_down == 1))
#         # print("How many b?: ", np.sum(cur_mask == 1))
#
#         # TODO
#         # cur_values_down = cur_values
#         # cur_t_down = cur_t
#
#         # STANDARDIZATION
#         mean = np.mean(cur_values_down)
#         std = np.std(cur_values_down)
#         mean_t = np.mean(cur_t_down)
#         std_t = np.std(cur_t_down)
#
#         cur_t_down_norm, cur_values_down_norm = (cur_t_down - mean_t) / std_t, (cur_values_down - mean) / std
#         outliers = np.logical_and(np.greater_equal(cur_values_down_norm, -5), np.less_equal(cur_values_down_norm, 5))
#
#         cur_values_down_norm = cur_values_down_norm[outliers].reshape(-1, 1)
#         cur_t_down_norm = cur_t_down_norm[outliers].reshape(-1, 1)
#         cur_mask_down = cur_mask_down[outliers].reshape(-1, 1)
#         # t, val = t[np.greater_equal(val, -5)], val[np.greater_equal(val, -5)]
#         # t, val = t[np.less_equal(val, 5)], val[np.less_equal(val, 5)]
#
#         scale = window_ / 3
#         scale /= std_t
#         kernel = Matern(length_scale=scale,
#                         length_scale_bounds=(scale, scale),
#                         nu=1.5)
#
#         # kernel += WhiteKernel(1, (1e-5, 1e5))
#
#         # TODO
#         kernel += DualWhiteKernel()
#         gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10)
#
#         # TODO
#         gpr.fit(np.hstack((cur_t_down_norm, cur_mask_down)), cur_values_down_norm)
#         # gpr.fit(cur_t_down_norm, cur_values_down_norm)
#
#         print(gpr.kernel_)
#         print(gpr.kernel_.get_params())
#
#         # Result
#         # time_to_predict = np.linspace(cur_t_down_norm[0], cur_t_down_norm[-1], 20).reshape(-1, 1)
#         time_to_predict = cur_t_down_norm
#         time_to_predict_with_mask = np.hstack((cur_t_down_norm, cur_mask_down))
#         # time_to_predict_with_mask = np.hstack((time_to_predict, np.full_like(time_to_predict, GPConsts.LABEL_A)))
#
#         # TODO
#         predict_values, sigma = gpr.predict(time_to_predict_with_mask, return_std=True)
#         # predict_values, sigma = gpr.predict(time_to_predict, return_std=True)
#
#         # Project to non-standardized setting
#         time_to_predict = (time_to_predict * std_t) + mean_t
#         predict_values = (predict_values * std) + mean
#         sigma = std * sigma
#
#         # Plot the function, the prediction and the 95% confidence interval
#         plt.figure(5, figsize=(16, 8))
#
#         cur_t = np.squeeze(cur_t)
#         cur_mask_down = np.squeeze(cur_mask_down)
#         cur_values = np.squeeze(cur_values)
#         cur_t_down = np.squeeze(cur_t_down)
#         cur_values_down = np.squeeze(cur_values_down)
#         time_to_predict = np.squeeze(time_to_predict)
#         predict_values = np.squeeze(predict_values)
#         sigma = np.squeeze(sigma)
#
#         a1 = np.argsort(cur_t)
#         print("Cur t before: ", cur_t)
#         cur_t = cur_t[a1]
#         print("Cur t after: ", cur_t)
#         cur_values = cur_values[a1]
#
#         a2 = np.argsort(cur_t_down)
#         cur_t_down = cur_t_down[a2]
#         cur_values_down = cur_values_down[a2]
#         cur_mask_down = cur_mask_down[a2]
#
#         a3 = np.argsort(time_to_predict)
#         sigma = sigma[a3]
#         time_to_predict = time_to_predict[a3]
#         predict_values = predict_values[a3]
#
#         plt.plot(cur_t, cur_values, 'r.', markersize=2, label='Observations')
#         plt.plot(cur_t_down, cur_values_down, 'k.', markersize=5, label='Downscaled')
#         plt.plot(cur_t_down[cur_mask_down == 1], cur_values_down[cur_mask_down == 1], 'g.', markersize=5)
#         plt.plot(time_to_predict, predict_values, 'b-', label='Prediction')
#         plt.axvline(x=cur_target_t)
#
#         plt.fill(np.concatenate([time_to_predict, time_to_predict[::-1]]),
#                  np.concatenate([predict_values - 1.9600 * sigma,
#                                  (predict_values + 1.9600 * sigma)[::-1]]),
#                  alpha=.5, fc='b', ec='None')
#         plt.xlabel('$x$')
#         plt.ylabel('$f(x)$')
#
#         # plt.plot(cur_t_down_norm, cur_values_down_norm, alpha=0.6)
#
#         print(i)
#         if i == 20:
#             plt.show()
#             break


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
            kernel = MaternT(length_scale=scale, length_scale_bounds=(scale, scale), nu=GPConsts.MATERN_NU)

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
                plt.plot(cur_t, cur_x, 'r.', markersize=1, label='Observations', alpha=0.1)
                plt.plot(cur_t_down, cur_x_down, 'k.', markersize=5, label='Downscaled')
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
    with open('./../results/2019-11-04_12-20-59_smooth_monotonic/smooth_monotonic_modeling_result.pkl', 'rb') as handle:
        res: Result = pickle.load(handle)

    merger = LocalGPMerger(res)
    # 0.5 value of window means half a day before prediction time and half a day after it
    # predictions_ = merger.merge(res.out.t_daily_out[300:1500:100], window=50, points_in_window=50, plot=True)
    predictions_ = merger.merge(res.out.t_daily_out[100:500:50], window=25, points_in_window=50, plot=True)
    print(predictions_)
