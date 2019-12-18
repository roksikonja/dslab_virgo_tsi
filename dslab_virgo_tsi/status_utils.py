from enum import Enum
from threading import Lock
from time import time

from flask import render_template


class StatusField(Enum):
    RUNNING = "is_running"
    JOB_NAME = "job_name"
    JOB_PERCENTAGE = "job_percentage"
    JOB_DESCRIPTION = "job_description"
    DATASET_TABLE = "dataset_table"
    DATASET_LIST = []
    LAST_DB_UPDATE = "last_db_update"
    LAST_JOB_UPDATE = "last_job_update"
    LAST_DB_TABLE_RENDER = "last_db_table_render"
    RESULT_FOLDER = "result_folder"
    JOB_TYPE = "job_type"


class JobType(Enum):
    ANALYSIS = "analysis"
    IMPORT = "import"


class Status:
    def __init__(self):
        self._data = {StatusField.RUNNING: False,
                      StatusField.JOB_NAME: "",
                      StatusField.JOB_PERCENTAGE: 0,
                      StatusField.JOB_DESCRIPTION: "",
                      StatusField.DATASET_TABLE: "",
                      StatusField.DATASET_LIST: None,
                      StatusField.LAST_DB_UPDATE: 0.0,
                      StatusField.LAST_JOB_UPDATE: 0.0,
                      StatusField.LAST_DB_TABLE_RENDER: 0.0,
                      StatusField.RESULT_FOLDER: ""}
        self._lock = Lock()
        self._no_include = {StatusField.DATASET_LIST, StatusField.LAST_DB_TABLE_RENDER, StatusField.RESULT_FOLDER}

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            current_time = time()
            if key == StatusField.DATASET_LIST:
                self._data[StatusField.LAST_DB_UPDATE] = current_time
            elif key != StatusField.DATASET_TABLE:
                self._data[StatusField.LAST_JOB_UPDATE] = current_time

    def set_description(self, description):
        self.set(StatusField.JOB_DESCRIPTION, description)

    def set_percentage(self, percentage):
        self.set(StatusField.JOB_PERCENTAGE, percentage)

    def set_folder(self, folder):
        self.set(StatusField.RESULT_FOLDER, folder)

    def update_progress(self, description, percentage):
        with self._lock:
            self._data[StatusField.JOB_DESCRIPTION] = description
            self._data[StatusField.JOB_PERCENTAGE] = percentage
            self._data[StatusField.LAST_JOB_UPDATE] = time()

    def set_dataset_list(self, dataset_list):
        self.set(StatusField.DATASET_LIST, dataset_list)

    def release(self):
        self.set(StatusField.RUNNING, False)

    def new_job(self, job_type: JobType, description, name):
        with self._lock:
            self._data[StatusField.RUNNING] = True
            self._data[StatusField.JOB_TYPE] = job_type.value
            self._data[StatusField.JOB_DESCRIPTION] = description
            self._data[StatusField.JOB_NAME] = name
            self._data[StatusField.JOB_PERCENTAGE] = 0
            self._data[StatusField.LAST_JOB_UPDATE] = time()

    def get(self, key):
        with self._lock:
            return self._data[key]

    def get_folder(self):
        return self.get(StatusField.RESULT_FOLDER)

    def get_dataset_list(self):
        return self.get(StatusField.DATASET_LIST)

    def is_running(self):
        return self.get(StatusField.RUNNING)

    def get_json(self):
        with self._lock:
            # Only render table if it has changed since it was last rendered
            if self._data[StatusField.LAST_DB_TABLE_RENDER] < self._data[StatusField.LAST_DB_UPDATE]:
                # Render table
                self._data[StatusField.DATASET_TABLE] = render_template("dataset_table.html",
                                                                        datasets=self._data[StatusField.DATASET_LIST])
                self._data[StatusField.LAST_DB_TABLE_RENDER] = self._data[StatusField.LAST_DB_UPDATE]
            return {key.value: self._data[key] for key in self._data if key not in self._no_include}


status = Status()
