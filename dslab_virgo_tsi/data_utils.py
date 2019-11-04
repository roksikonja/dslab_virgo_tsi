import datetime
import os
import pickle

import numpy as np
import pandas as pd
from numba import jit, jitclass, int32, float64
from sklearn.covariance import EllipticEnvelope

from dslab_virgo_tsi.constants import Constants as Const


def check_data(file_dir, num_cols=4):
    print("checking data file", file_dir, "...")
    with open(file_dir, "r") as f:
        error_count = 0
        for line in f.readlines():
            line = line.strip().split()
            if len(line) != num_cols:
                error_count = error_count + 1

    if error_count > 0:
        print(error_count, "errors found")
        return False
    else:
        print("valid file")
        return True


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


def create_results_dir(results_dir_path, model_type):
    results_dir = make_dir(os.path.join(results_dir_path,
                                        datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_{model_type}")))
    return results_dir


def save_modeling_result(results_dir, model_results, model_name):
    with open(os.path.join(results_dir, f"{model_name}_modeling_result.pkl"), 'wb') as f:
        pickle.dump(model_results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_config(results_dir, config):
    with open(os.path.join(results_dir, "config.txt"), "w") as f:
        for key in config:
            if not str(key)[0:2] == "__" and not str(key) == "return_config":
                f.write("{:<30}{}\n".format(key + ":", config[key]))


if __name__ == "__main__":
    data_dir = "./data"
    virgo_file = "VIRGO_Level1.txt"

    # Check if each row contains all columns
    check_data(os.path.join(data_dir, virgo_file), num_cols=4)

    # Load data
    virgo_data = load_data(data_dir, virgo_file)
