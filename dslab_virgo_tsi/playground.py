import pickle

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

from dslab_virgo_tsi.base import Result
# from dslab_virgo_tsi.kernels import DualWhiteKernel, Matern


@njit(cache=True)
def interleave(t_a_, t_b_, a_, b_):
    index_a = 0
    index_b = 0
    index_together = 0
    a_length = t_a_.size
    b_length = t_b_.size
    t_together_ = np.empty((a_length + b_length,))
    values_together_ = np.empty((a_length + b_length,))
    mask_ = np.empty((a_length + b_length))

    # 0 = a, 1 = b
    while index_together < a_length + b_length:
        while index_a < a_length and (t_a_[index_a] <= t_b_[index_b] or index_b == b_length):
            t_together_[index_together] = t_a_[index_a]
            values_together_[index_together] = a_[index_a]
            mask_[index_together] = 0
            index_a += 1
            index_together += 1

        while index_b < b_length and (t_b_[index_b] <= t_a_[index_a] or index_a == a_length):
            t_together_[index_together] = t_b_[index_b]
            values_together_[index_together] = b_[index_b]
            mask_[index_together] = 1
            index_b += 1
            index_together += 1

    return t_together_, values_together_, mask_


def gp_target_time(t_, val_, mask_, t_target_, window_: float = 10, points_in_window=50):
    """
    :param t_: Time of all signals.
    :param val_: Signal.
    :param mask_: Mask.
    :param t_target_: Time.
    :param window_: How much time left/right of prediction time should be taken into account.
    :param points_in_window: How many points should be in left/right of window.
    :return:
    """
    t_ = t_.reshape(-1, 1)
    val_ = val_.reshape(-1, 1)
    mask_ = mask_.reshape(-1, 1)

    # TODO
    t_target_ = t_target_[20:]
    t_target_ = t_target_.reshape(-1, 1)

    t_length = t_.shape[0]
    start_index = 0
    end_index = 0

    for i in range(t_target_.shape[0]):
        cur_target_t = t_target_[i]

        while end_index < t_length and t_[end_index] < cur_target_t + window_:
            end_index += 1

        while t_[start_index] < cur_target_t - window_:
            start_index += 1

        cur_values = val_[start_index:end_index]
        cur_t = t_[start_index:end_index]
        cur_mask = mask_[start_index:end_index]

        # Downscaling
        ratio_too_much = int(np.ceil(cur_values.size / points_in_window))
        print(cur_values.size)
        print(points_in_window)
        print("Ratio: ", ratio_too_much)
        cur_values_down = cur_values[::ratio_too_much]
        cur_t_down = cur_t[::ratio_too_much]
        cur_mask_down = cur_mask[::ratio_too_much]

        # TODO
        # cur_values_down = cur_values
        # cur_t_down = cur_t

        # STANDARDIZATION
        mean = np.mean(cur_values_down)
        std = np.std(cur_values_down)
        mean_t = np.mean(cur_t_down)
        std_t = np.std(cur_t_down)

        cur_t_down_norm, cur_values_down_norm = (cur_t_down - mean_t) / std_t, (cur_values_down - mean) / std
        # t, val = t[np.greater_equal(val, -5)], val[np.greater_equal(val, -5)]
        # t, val = t[np.less_equal(val, 5)], val[np.less_equal(val, 5)]

        scale = window_ / 3
        scale /= std_t
        kernel = Matern(length_scale=scale,
                        length_scale_bounds=(scale, scale),
                        nu=1.5)

        # kernel += DualWhiteKernel(cur_mask_down, 1, 1, (1e-5, 1e5), (1e-5, 1e5))
        kernel += WhiteKernel(1, (1e-5, 1e5))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10)

        gpr.fit(cur_t_down_norm, cur_values_down_norm)

        # Result
        time_to_predict = np.linspace(cur_t_down_norm[0], cur_t_down_norm[-1], 1000).reshape(-1, 1)
        predict_values, sigma = gpr.predict(time_to_predict, return_std=True)
        sigma = sigma.reshape(-1, 1)

        # Project to non-standardized setting
        time_to_predict = (time_to_predict * std_t) + mean_t
        predict_values = (predict_values * std) + mean
        sigma = std * sigma

        # Plot the function, the prediction and the 95% confidence interval
        plt.figure(5, figsize=(16, 8))
        plt.plot(cur_t, cur_values, 'r.', markersize=2, label='Observations')
        plt.plot(cur_t_down, cur_values_down, 'k.', markersize=5, label='Downscaled')
        plt.plot(time_to_predict, predict_values, 'b-', label='Prediction')
        plt.axvline(x=cur_target_t)
        plt.fill(np.concatenate([time_to_predict, time_to_predict[::-1]]),
                 np.concatenate([predict_values - 1.9600 * sigma,
                                 (predict_values + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')

        # plt.plot(cur_t_down_norm, cur_values_down_norm, alpha=0.6)

        print(i)
        if i == 0:
            plt.show()
            break


def main():
    with open('./../results/2019-11-04_12-20-59_smooth_monotonic/smooth_monotonic_modeling_result.pkl', 'rb') as handle:
        x: Result = pickle.load(handle)

    a = x.final.a_nn_corrected
    b = x.final.b_nn_corrected
    t_a = x.base_signals.t_a_nn
    t_b = x.base_signals.t_b_nn
    t_target = x.out.t_daily_out[::30]

    t, val, mask = interleave(t_a, t_b, a, b)
    print("Interleaving done")

    half_hour = 1 / (24 * 2)
    half_day = 1 / 2
    five_days = 5
    fifteen_days = 15
    gp_target_time(t, val, mask, t_target, fifteen_days)


if __name__ == "__main__":
    main()
