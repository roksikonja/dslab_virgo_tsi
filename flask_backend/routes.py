import os
import secrets
from time import sleep

from flask import render_template, redirect, url_for, jsonify, Response, current_app

from dataset_handling_utils import import_data_job, update_table
from flask_backend import app, executor, status
from flask_backend.forms import NewDataForm, AnalysisForm
from flask_backend.models import Dataset
from status_utils import StatusField as sF, JobType as jT


@app.route("/get_update")
def get_update() -> Response:
    if status.get_dataset_list() is None:
        update_table()

    return jsonify(status.get_json())


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/import_data", methods=["GET", "POST"])
def import_data():
    form = NewDataForm()

    # Check whether import can be performed
    if form.validate_on_submit() and not status.is_running():

        # Block other operations
        status.new_job(jT.IMPORT, "Importing CSV", "Import data")

        # Store file to server (temporarily, overwritten on each call)
        # - Generate a random name
        name = secrets.token_hex(8)

        # - Store file in static/data/<name>.csv
        dataset_location = os.path.join(current_app.root_path, "static", "data", name + ".csv")
        form.file.data.save(dataset_location)

        # Perform import (new thread)
        executor.submit(import_data_job, form.name.data, form.file.data.filename, dataset_location,
                        form.outlier_fraction.data, form.exposure_method.data)
        # import_data_job(form.name.data, form.file.data.filename, dataset_location, form.outlier_fraction.data,
        #                 form.exposure_method.data)
        return redirect(url_for("home"))

    return render_template("import_data.html", form=form)


@app.route("/delete_data/<int:dataset_id>", methods=["POST"])
def delete_data(dataset_id):
    # delete_dataset(dataset_id)
    return redirect(url_for("home"))


def _analysis():
    sleep(3)
    status.set(sF.JOB_PERCENTAGE, 30)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 60)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 90)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 100)
    status.release()


@app.route("/analysis/<int:dataset_id>", methods=["GET", "POST"])
def analysis(dataset_id):
    form = AnalysisForm()
    if form.validate_on_submit() and not status.is_running():
        # Block other operation
        status.new_job(jT.ANALYSIS, "Analysis in progress", "Data Analysis")

        # Perform analysis (new thread)
        executor.submit(_analysis)

        return redirect(url_for("home"))

    name = Dataset.query.get_or_404(dataset_id).name
    return render_template("analysis.html", title=name, form=form)


@app.errorhandler(404)
def error_404(_):
    return render_template('404.html'), 404


@app.route("/results")
def results():
    pass
