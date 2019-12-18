import os
import pickle
from os.path import basename, splitext

from dslab_virgo_tsi.base import ModelFitter, ExposureMethod
from dslab_virgo_tsi.constants import Constants as Const
from dslab_virgo_tsi.data_utils import load_data_from_frontend
from flask_backend import db
from flask_backend.models import Dataset
from dslab_virgo_tsi.status_utils import status


def add_dataset(dataset: Dataset):
    db.session.add(dataset)
    db.session.commit()
    update_table()


def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    print("Remove dataset")
    os.remove(dataset.dataset_location)

    print("Remove pickle")
    os.remove(dataset.pickle_location)
    db.session.delete(dataset)
    db.session.commit()
    update_table()


def update_table():
    datasets = Dataset.query.all()
    status.set_dataset_list(datasets)


def import_data_job(dataset_name: str, filename: str, dataset_location: str, outlier_fraction: float,
                    exposure_method: str):

    try:
        # Load pandas object
        status.set_percentage(10)
        data = load_data_from_frontend(dataset_location)

        # Get fitter object
        status.update_progress("Removing outliers", 30)
        fitter = ModelFitter(data, Const.A, Const.B, Const.T, ExposureMethod(exposure_method), outlier_fraction)

        # Store fitter
        status.update_progress("Storing processed signals", 70)
        path, _ = splitext(dataset_location)
        pickle_location = path + ".pickle"

        with open(pickle_location, 'wb') as f:
            pickle.dump(fitter, f, pickle.HIGHEST_PROTOCOL)

        # Prepare table entry
        if dataset_name == "":
            dataset_name, _ = splitext(basename(filename))
        dataset = Dataset(name=dataset_name, exposure_method=exposure_method, outlier_fraction=outlier_fraction,
                          dataset_location=dataset_location, pickle_location=pickle_location)

        # Add entry to database
        add_dataset(dataset)

    except Exception as e:
        print(e)

    finally:
        # Release block
        status.release()
