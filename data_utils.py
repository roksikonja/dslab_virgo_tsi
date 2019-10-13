import pandas as pd
import numpy as np
import os
import datetime


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


def load_data(file_dir, data_type="virgo"):
    if data_type == "virgo":
        data = pd.read_csv(file_dir,
                           header=None,
                           delimiter=r"\s+").rename(columns={0: "timestamp",
                                                             1: "pmo6v_a",
                                                             2: "pmo6v_b",
                                                             3: "temperature"})
        return data
    return None


def downsample_signal(x, k=1):
    if k > 1:
        return x[::k]
    else:
        return x


def notnan_indices(x):
    return ~np.isnan(x)


def moving_average_std(x, w, center=True):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    if w > 1:
        x = x.rolling(w, center=center)
        return x.mean(), x.std()
    else:
        return x, np.zeros(shape=x.shape)


def mission_day_to_year(day):
    start = datetime.datetime(1996, 1, 1)
    years = start.year + day / 365.25

    return years


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


if __name__ == "__main__":
    data_dir = "./data"
    virgo_file = "VIRGO_Level1.txt"

    # Check if each row contains all columns
    check_data(os.path.join(data_dir, virgo_file), num_cols=4)

    # Load data
    virgo_data = load_data(os.path.join(data_dir, virgo_file))
