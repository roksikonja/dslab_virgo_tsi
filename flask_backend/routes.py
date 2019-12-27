import os
import secrets

from flask import render_template, redirect, url_for, jsonify, Response, current_app

from dslab_virgo_tsi.status_utils import JobType as jT
from dslab_virgo_tsi.status_utils import status
from flask_backend import app, executor
from flask_backend.analysis_utils import analysis_job
from flask_backend.dataset_handling_utils import import_data_job, update_table, delete_dataset
from flask_backend.forms import NewDataForm, AnalysisForm
from flask_backend.models import Dataset


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
    delete_dataset(dataset_id)
    return redirect(url_for("home"))


@app.route("/analysis/<int:dataset_id>", methods=["GET", "POST"])
def analysis(dataset_id):
    form = AnalysisForm()

    # Check whether analysis can be performed
    if form.validate_on_submit() and not status.is_running():
        # Get dataset first to ensure that it was not deleted while waiting
        # User could have deleted dataset in another tab while having analysis tab open
        dataset = Dataset.query.get_or_404(dataset_id)

        # Block other operation
        status.new_job(jT.ANALYSIS, "Analysis in progress", "Data Analysis")

        # Perform analysis (new thread)
        executor.submit(analysis_job, dataset, form.model.data, form.output.data, form.model_params.data,
                        form.correction.data)
        # analysis_job(dataset, form.model.data, form.output.data, form.model_params.data, form.correction.data)

        return redirect(url_for("home"))

    name = Dataset.query.get_or_404(dataset_id).name
    return render_template("analysis.html", title=name, form=form)


@app.errorhandler(404)
def error_404(_):
    return render_template("404.html"), 404


@app.errorhandler(405)
def error_405(_):
    return render_template("405.html"), 405


@app.route("/results")
def results():
    result_folder = status.get_folder()
    if result_folder == "":
        return error_404("Error")

    # Prefix of all files within folder
    prefix = "_".join(result_folder.split("_")[2:]) + "_"
    first = prefix + "PMO6V-A_PMO6V-B_mutual_corrected.pdf"
    second = prefix + "PMO6V-A_PMO6V-B_raw_corrected_full.pdf"
    third = prefix + "DEGRADATION_PMO6V-A_PMO6V-B.pdf"
    fourth = prefix + "TSI_hourly_points_95_conf_interval.pdf"

    return render_template("results.html", folder=result_folder, first=first, second=second, third=third, fourth=fourth)
