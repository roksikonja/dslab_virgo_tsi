import ftplib
import os
import time


def is_dir(ftp_, name_):
    try:
        ftp_.cwd(name_)
        ftp_.cwd('..')
        return True
    except ftplib.error_perm:
        return False


def download_file(ftp_, remote_dir_, local_dir_):
    format_ = remote_dir_.split(".")[-1]
    if format_ not in ["pdf", "jpg", "jpeg", "html", "htm"] and "":
        with open(local_dir_, "wb") as f:
            ftp_.retrbinary("RETR %s" % remote_dir_, lambda data: f.write(data))


def download_folder(ftp_, remote_dir_, local_dir_, depth=1):
    ftp_.cwd(remote_dir_)

    current_local_dir_ = os.path.join(local_dir_, os.path.basename(os.path.normpath(remote_dir_)))

    if not os.path.exists(current_local_dir_):
        os.mkdir(current_local_dir_)

    filenames_ = ftp.nlst()
    for idx_, filename_ in enumerate(filenames_):
        file_is_dir_ = is_dir(ftp, filename_)
        print("{}{:<10}\t{:<20}\tfile = {:<5}\tdir = {:<}".format("\t\t" * depth + "-->\t", idx_, filename_,
                                                                  str(not file_is_dir_), str(file_is_dir_)))

        if file_is_dir_:
            if filename_ != "SSI":
                download_folder(ftp_, os.path.join(remote_dir_, filename_), current_local_dir_, depth=depth + 1)
        else:
            download_file(ftp_, remote_dir_ + "/" + filename_, os.path.join(current_local_dir_, filename_))
            time.sleep(1)

    ftp_.cwd("..")


local_data_dir = "../data/"
remote_data_dir = "/pub/data/irradiance/virgo/"

ftp = ftplib.FTP("ftp.pmodwrc.ch")

print("Welcome: ", ftp.getwelcome())
print("Login: ", ftp.login())


download_folder(ftp, remote_data_dir, local_data_dir, depth=1)

ftp.quit()
