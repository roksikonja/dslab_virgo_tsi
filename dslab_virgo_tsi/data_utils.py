import datetime
import logging
import os
from numbers import Real

import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.utilities.utilities import tabulate_module_summary
from numba import jit, jitclass, int32, float64
from sklearn.covariance import EllipticEnvelope

from dslab_virgo_tsi.constants import Constants as Const


def check_data(file_dir, num_cols=4):
    logging.info(f"Checking data file {file_dir} ...")
    with open(file_dir, "r") as f:
        error_count = 0
        for line in f.readlines():
            line = line.strip().split()
            if len(line) != num_cols:
                error_count = error_count + 1

    if error_count > 0:
        logging.info(f"{error_count} errors found.")
        return False
    else:
        logging.info("Check passed.")
        return True


def load_data_from_frontend(data_path):
    return pd.read_csv(data_path, header=None, delimiter=r"\s+").rename(columns={0: Const.T, 1: Const.A, 2: Const.B})


def load_data(data_dir_path, file_name, data_type="virgo"):
    file_name_no_extension, extension = os.path.splitext(file_name)

    h5_file_path = os.path.join(data_dir_path, file_name_no_extension + ".h5")
    # If HDF5 file exists, load from it
    if os.path.isfile(h5_file_path):
        data = pd.read_hdf(h5_file_path, "table")
        return data

    # HDF5 file does not exist, load from regular format
    raw_file_path = os.path.join(data_dir_path, file_name)
    if data_type == "virgo":
        data = pd.read_csv(raw_file_path,
                           header=None,
                           delimiter=r"\s+").rename(columns={0: Const.T,
                                                             1: Const.A,
                                                             2: Const.B,
                                                             3: Const.TEMP})

        # Store to HDF5
        data.to_hdf(h5_file_path, "table")
        return data
    elif data_type == "virgo_tsi":
        data = []
        with open(os.path.join("./data/", file_name), "r") as f:
            for idx, line in enumerate(f.readlines()):
                if line[0] != ";" and idx != 0:
                    line = line.strip().split()

                    # Time in mission days
                    date = datetime.datetime.strptime("-".join(line[0:2]), "%Y%m%d-%H%M%S") \
                        - datetime.datetime(1996, 1, 1, 0, 0, 0)
                    date = float(date.days) + float(date.seconds) / (3600 * 24.0)
                    values = line[2:]

                    array = [date]
                    array.extend(values)
                    data.append(array)

        data = pd.DataFrame(data, columns=[Const.T,
                                           Const.VIRGO_TSI_NEW,
                                           Const.VIRGO_TSI_OLD,
                                           Const.DIARAD_OLD,
                                           Const.PMO6V_OLD], dtype=float)
        # Replace missing values with NaN
        data[data == Const.VIRGO_TSI_NAN_VALUE] = np.nan

        # Store to HDF5
        data.to_hdf(h5_file_path, "table")
        return data
    return None


def is_integer(num):
    return isinstance(num, (int, np.int, np.int32, np.int64))


def downsample_signal(x, k=1):
    if not is_integer(k):
        raise Exception("Downsampling factor must be an integer.")
    if k > 1:
        return x[::k]
    else:
        return x


@jit(nopython=True, cache=True)
def notnan_indices(x):
    return ~np.isnan(x)


def mission_day_to_year(day):
    start = datetime.datetime(1996, 1, 1)
    years = start.year + day / 365.25

    return years


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


@jit(nopython=True, cache=True)
def moving_average_std(x, t, w):
    """

    :param x: Signal.
    :param t: Time.
    :param w: Window size.
    :return:
    """
    num_elements = x.shape[0]
    w = w + 1e-10
    nn_indices = notnan_indices(x)
    x_mean = np.empty(shape=x.shape)
    x_std = np.empty(shape=x.shape)

    slice_ = NumpyQueue(num_elements)
    start_index = 0
    end_index = 0
    for i in range(num_elements):
        while end_index < x.shape[0] and t[end_index] < t[i] + w:
            if nn_indices[end_index]:
                slice_.append(x[end_index])
            end_index += 1

        while t[start_index] < t[i] - w:
            if nn_indices[start_index]:
                slice_.pop_left()
            start_index += 1

        x_mean[i] = np.mean(slice_.get())
        x_std[i] = np.std(slice_.get())

    return x_mean, x_std


spec = [
    ('size', int32),  # a simple scalar field
    ('array', float64[:]),  # an array field
    ('beginning', int32),
    ('end', int32)
]


@jitclass(spec)
class NumpyQueue:
    def __init__(self, size):
        self.size = size
        self.array = np.empty((size,))
        self.beginning = 0
        self.end = 0

    def is_empty(self):
        return self.beginning == self.end

    def append(self, element):
        if self.end >= self.size:
            raise Exception("Capacity exceeded")
        self.array[self.end] = element
        self.end += 1

    def pop_left(self):
        if self.is_empty():
            raise Exception("Cannot pop from empty queue")
        self.beginning += 1

    def get(self):
        return self.array[self.beginning:self.end]


def interpolate_nearest(x):
    ind = np.where(~np.isnan(x))[0]
    first, last = ind[0], ind[-1]
    x[:first] = x[first]
    x[last + 1:] = x[last]
    return x


def resampling_average_std(x, w, std=True):
    w = int(w)
    x_resampled_mean = None
    x_resampled_std = None

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    if w > 1:
        x = x.rolling(w, center=True)
        x_resampled_mean = x.mean()
        x_resampled_mean = interpolate_nearest(x_resampled_mean)
        if std:
            x_resampled_std = x.std()
            x_resampled_std = interpolate_nearest(x_resampled_std)

    return x_resampled_mean, x_resampled_std


def get_sampling_intervals(t, x):
    sampling_intervals = []
    sampling = False
    start = None

    for index in range(t.shape[0]):
        if not np.isnan(x[index]) and not sampling:
            sampling = True
            start = index

        elif np.isnan(x[index]) and sampling:
            sampling = False
            end = index
            sampling_intervals.append((start, end))

    return sampling_intervals


def fft_spectrum(x, sampling_period):
    n = x.shape[0]
    mean = x.mean()
    x_fft = np.fft.fft(x - mean) / n
    s = np.square(np.abs(x_fft))

    f_0 = 1 / n
    fs = 1 / sampling_period
    k = np.arange(n)
    freq = k * f_0 * fs

    return s, fs, freq


def detect_outliers(x_fit, x=None, outlier_fraction=1e-3):
    x_fit = np.reshape(x_fit, newshape=(-1, 1))

    if not x:
        x = x_fit
    else:
        x = np.reshape(x, newshape=(-1, 1))

    if outlier_fraction <= 0:
        outliers = np.zeros(shape=x.shape, dtype=np.bool)
    else:
        envelope = EllipticEnvelope(contamination=outlier_fraction)
        envelope.fit(x_fit)
        outliers = envelope.predict(x) == -1

    return outliers


def median_downsample_by_factor(x, n):
    return np.array([np.median(x[i:n + i]) for i in range(0, np.size(x), n)])


def median_downsample_by_max_points(x, max_points=500):
    n = np.ceil(x.shape[0] / max_points).astype(int)
    return np.array([np.median(x[i:n + i]) for i in range(0, np.size(x), n)])


def get_summary(module: tf.Module):
    """
    Returns a summary of the parameters and variables contained in a tf.Module.
    """
    return tabulate_module_summary(module, None)


def normalize(x, mean, std):
    if isinstance(x, Real) or len(x.shape) <= 1:
        y = (x - mean) / std
    else:
        y = x
        y[:, 0] = (x[:, 0] - mean) / std
    return y


def unnormalize(y, mean, std):
    if isinstance(y, Real) or len(y.shape) <= 1:
        x = std * y + mean
    else:
        x = y
        x[:, 0] = std * y[:, 0] + mean
    return x


def find_nearest(array, values):
    indices = np.zeros(values.shape)
    for index, value in enumerate(values):
        indices[index] = np.abs(array - value).argmin()
    return indices


def extract_time_window(t, t_mid, window):
    start_index = 0
    end_index = 0
    win_beginning = t_mid - window / 2.0
    win_end = t_mid + window / 2.0
    while end_index < np.size(t) and t[end_index] <= win_end:
        end_index += 1

    while t[start_index] <= win_beginning:
        start_index += 1

    return start_index, end_index


if __name__ == "__main__":
    data_dir = "./data"
    virgo_file = "VIRGO_Level1.txt"

    # Check if each row contains all columns
    check_data(os.path.join(data_dir, virgo_file), num_cols=4)

    # Load data
    virgo_data = load_data(data_dir, virgo_file)
