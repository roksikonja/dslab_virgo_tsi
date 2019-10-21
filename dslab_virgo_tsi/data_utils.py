import datetime
import os

import numpy as np
import pandas as pd
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


def downsample_signal(x, k=1):
    if k > 1:
        return x[::k]
    else:
        return x


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


def moving_average_std(x, t, w):
    if t.tolist():
        w = w + 1e-10
        x_mean = np.zeros(shape=x.shape)
        x_std = np.zeros(shape=x.shape)
        for i in range(x.shape[0]):
            t_center = t[i]
            window = np.multiply(np.greater_equal(t_center + w, t), np.less_equal(t_center - w, t))
            slice_ = x[window]

            indices = notnan_indices(slice_)
            slice_ = slice_[indices]
            x_mean[i] = slice_.mean()
            x_std[i] = slice_.std()
        return x_mean, x_std
    else:
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        if w > 1:
            x = x.rolling(w, center=True)
            return x.mean(), x.std()
        else:
            return x, None


def resampling_average_std(x, w):
    w = int(w)
    x_resampled_mean = None
    x_resampled_std = None

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    if w > 1:
        x = x.rolling(w, center=True)
        x_resampled_mean = x.mean()
        x_resampled_std = x.std()

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


if __name__ == "__main__":
    data_dir = "./data"
    virgo_file = "VIRGO_Level1.txt"

    # Check if each row contains all columns
    check_data(os.path.join(data_dir, virgo_file), num_cols=4)

    # Load data
    virgo_data = load_data(data_dir, virgo_file)
