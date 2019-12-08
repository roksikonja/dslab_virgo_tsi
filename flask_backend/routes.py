from os.path import basename, splitext
from time import sleep

from flask import render_template, flash, redirect, url_for

from flask_backend import app, db, executor
from flask_backend.forms import NewDataForm, AnalysisForm
from flask_backend.models import Dataset, ConstantsAccess


@app.route("/")
@app.route("/home")
def home():
    datasets = Dataset.query.all()
    return render_template("home.html", datasets=datasets)


def _import_data(dataset: Dataset):
    sleep(5)
    db.session.add(dataset)
    db.session.commit()
    ConstantsAccess.set_import(False)
    print("Imported")


@app.route("/import_data", methods=["GET", "POST"])
def import_data():
    form = NewDataForm()
    if form.validate_on_submit():
        ConstantsAccess.set_import(False)
        if ConstantsAccess.get_import():
            flash("Import already running!", "danger")
            return redirect(url_for("home"))
        elif ConstantsAccess.get_analysis():
            flash("Analysis already running!", "danger")
            return redirect(url_for("home"))

        # Block other operations
        ConstantsAccess.set_import(True)

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
        # _import_data(dataset)

        flash("Dataset import started successfully!", "success")
        return redirect(url_for("home"))

    return render_template("import_data.html", title="Import Dataset", form=form)


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
        flash("Analysis started successfully!", "success")
        return redirect(url_for("home"))

    name = Dataset.query.get_or_404(dataset_id).name
    return render_template("analysis.html", title="Analysis for " + name, form=form)


if __name__ == '__main__':
    app.run(debug=True)
