import os


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


data_dir = "./data/virgo"
file = "virgo_tsi_d_v4_902.dat"

data = read_dat(os.path.join(data_dir, file))

print(data)

for line in data:
    print("\t".join(line))
