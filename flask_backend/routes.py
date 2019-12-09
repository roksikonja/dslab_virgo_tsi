from os.path import basename, splitext
from time import sleep

from flask import render_template, redirect, url_for, jsonify, Response

from flask_backend import app, db, executor
from flask_backend.forms import NewDataForm, AnalysisForm
from flask_backend.models import Dataset, ConstantsAccess


@app.route("/get_update")
def get_update() -> Response:
    datasets = Dataset.query.all()
    dataset_table = render_template('dataset_table.html', datasets=datasets)
    is_running = ConstantsAccess.get_running()
    job_description = ConstantsAccess.get_job_description()
    job_name = ConstantsAccess.get_job_name()
    job_percentage = ConstantsAccess.get_job_percentage()
    message = {"dataset_table": dataset_table,
               "is_running": is_running,
               "job_description": job_description,
               "job_name": job_name,
               "job_percentage": job_percentage}
    return jsonify(message)


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


def _import_data(dataset: Dataset):
    sleep(1)
    ConstantsAccess.set_job_percentage("10")
    sleep(1)
    ConstantsAccess.set_job_percentage("20")
    sleep(1)
    ConstantsAccess.set_job_percentage("30")
    sleep(1)
    ConstantsAccess.set_job_percentage("40")
    sleep(1)
    ConstantsAccess.set_job_description("Removing outliers")
    ConstantsAccess.set_job_percentage("50")
    sleep(1)
    ConstantsAccess.set_job_percentage("70")
    sleep(1)
    ConstantsAccess.set_job_percentage("90")
    sleep(1)
    db.session.add(dataset)
    db.session.commit()
    ConstantsAccess.set_running("False")


@app.route("/import_data", methods=["GET", "POST"])
def import_data():
    form = NewDataForm()
    if form.validate_on_submit() and ConstantsAccess.get_running() == "False":

        # Block other operation
        ConstantsAccess.set_running("True")
        ConstantsAccess.set_job_description("Importing CSV")
        ConstantsAccess.set_job_name("Import data")
        ConstantsAccess.set_job_percentage("0")

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
    return redirect(url_for("home"))


@app.route("/analysis/<int:dataset_id>", methods=["GET", "POST"])
def analysis(dataset_id):
    print("ANALYSIS")
    form = AnalysisForm()
    if form.validate_on_submit():
        return redirect(url_for("home"))

    name = Dataset.query.get_or_404(dataset_id).name
    return render_template("analysis.html", title=name, form=form)


if __name__ == '__main__':
    app.run(debug=True)
