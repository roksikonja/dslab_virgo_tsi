from os.path import basename, splitext
from time import sleep

from flask import render_template, redirect, url_for, jsonify, Response

from flask_backend import app, db, executor, status
from flask_backend.forms import NewDataForm, AnalysisForm
from flask_backend.models import Dataset
from status_utils import StatusField as sF, JobType as jT


def add_dataset(dataset: Dataset):
    db.session.add(dataset)
    db.session.commit()
    update_table()


def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    db.session.delete(dataset)
    db.session.commit()
    update_table()


def update_table():
    datasets = Dataset.query.all()
    status.set(sF.DATASET_LIST, datasets)


@app.route("/get_update")
def get_update() -> Response:
    if status.get(sF.DATASET_LIST) is None:
        update_table()

    return jsonify(status.get_json())


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


def _import_data(dataset: Dataset):
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 30)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 60)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 90)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 100)
    add_dataset(dataset)
    status.set(sF.RUNNING, False)


@app.route("/import_data", methods=["GET", "POST"])
def import_data():
    form = NewDataForm()
    if form.validate_on_submit() and not status.get(sF.RUNNING):

        # Block other operation
        status.new_job(jT.IMPORT, "Importing CSV", "Import data")

        # Prepare table entry
        name = form.name.data
        if name == "":
            name, _ = splitext(basename(form.file.data))
            print(name)
        dataset = Dataset(name=name,
                          exposure_mode=form.exposure_mode.data,
                          outlier_fraction=form.outlier_fraction.data)

        # Perform import (new thread)
        executor.submit(_import_data, dataset)
        return redirect(url_for("home"))

    return render_template("import_data.html", form=form)


@app.route("/delete_data/<int:dataset_id>", methods=["POST"])
def delete_data(dataset_id):
    delete_dataset(dataset_id)
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
    status.set(sF.RUNNING, False)


@app.route("/analysis/<int:dataset_id>", methods=["GET", "POST"])
def analysis(dataset_id):
    print("ANALYSIS")
    form = AnalysisForm()
    if form.validate_on_submit() and not status.get(sF.RUNNING):

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
