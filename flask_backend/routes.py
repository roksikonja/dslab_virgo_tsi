from os.path import basename, splitext
from time import sleep

from flask import render_template, redirect, url_for, jsonify, Response

from flask_backend import app, db, executor, status
from flask_backend.forms import NewDataForm, AnalysisForm
from flask_backend.models import Dataset
from status_utils import StatusField as sF


def update_table():
    datasets = Dataset.query.all()
    status.set(sF.DATASET_LIST, datasets)


@app.route("/get_update")
def get_update() -> Response:
    if status.get(sF.DATASET_LIST) is None:
        update_table()

    # status.set(sF.DATASET_TABLE, render_template("dataset_table.html", datasets=status.get(sF.DATASET_LIST)))
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
    status.set(sF.JOB_DESCRIPTION, "<a href='{{ url_for('home') }}'> Neki </a>")
    db.session.add(dataset)
    db.session.commit()
    update_table()
    status.set(sF.RUNNING, False)


@app.route("/import_data", methods=["GET", "POST"])
def import_data():
    form = NewDataForm()
    if form.validate_on_submit() and not status.get(sF.RUNNING):

        # Block other operation
        status.set(sF.RUNNING, True)
        status.set(sF.JOB_DESCRIPTION, "Importing CSV")
        status.set(sF.JOB_NAME, "Import data")
        status.set(sF.JOB_PERCENTAGE, 0)

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
    dataset = Dataset.query.get_or_404(dataset_id)
    db.session.delete(dataset)
    db.session.commit()
    update_table()
    return redirect(url_for("home"))


def _analysis():
    sleep(3)
    status.set(sF.JOB_PERCENTAGE, 30)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 60)
    sleep(1)
    status.set(sF.JOB_PERCENTAGE, 90)
    sleep(1)
    sleep(10)
    status.set(sF.RUNNING, False)


@app.route("/analysis/<int:dataset_id>", methods=["GET", "POST"])
def analysis(dataset_id):
    print("ANALYSIS")
    form = AnalysisForm()
    if form.validate_on_submit() and not status.get(sF.RUNNING):

        # Block other operation
        status.set(sF.RUNNING, True)
        status.set(sF.JOB_DESCRIPTION, "Analysis in progress")
        status.set(sF.JOB_NAME, "Data Analysis")
        status.set(sF.JOB_PERCENTAGE, 0)

        # Perform analysis (new thread)
        executor.submit(_analysis)

        return redirect(url_for("home"))

    name = Dataset.query.get_or_404(dataset_id).name
    return render_template("analysis.html", title=name, form=form)
