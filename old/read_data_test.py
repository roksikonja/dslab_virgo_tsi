import os
from astropy.io import fits
from scipy.io import readsav

def read_dat(file_dir_):
    lines = []
    idx = -1
    print("reading " + file_dir_ + " ...")
    with open(file_dir_) as f:
        for line_ in f.readlines():
            idx = idx + 1
            if line_[0] == ";" or idx == 0:
                continue
            else:
                lines.append(line_.strip().split())

    return lines


def read_fits(file_dir_):
    hdulist = fits.open(file_dir_)
    print(hdulist.info())

    hdu = hdulist[0]
    print("data shape", hdu.data.shape)
    print(hdu.header)

    return hdu.data


def read_idl(file_dir_):
    return readsav(file_dir_, python_dict=True, verbose=True)


data_dir = "./data/virgo"

# file = "virgo_tsi_d_v4_902.dat"
# data = read_dat(os.path.join(data_dir, file))
#
# print(data)
# for line in data:
#     print("\t".join(line))

fits_file = "VIRGO_1min_0083-7404.fits"
idl_file = "VIRGO_1min_0083-7404.idl"
# data = read_fits(os.path.join(data_dir, "1-minute_Data", fits_file))
data = read_idl(os.path.join(data_dir, "1-minute_Data", idl_file))

for key in data:
    print(key, data[key])

print(data)