from flask import Flask, render_template, flash, redirect, url_for, request

from forms import NewDataForm, AnalysisForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "4c52401121326497c5baeae8f039a0c3"


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/import_data", methods=["GET", "POST"])
def import_data():
    form = NewDataForm()
    if form.validate_on_submit():
        flash("Dataset imported successfully!", "success")
        return redirect(url_for("home"))

    return render_template("import_data.html", title="Import Dataset", form=form)


mapping = {1: "DIARAD", 2: "PMVO6", 3: "TIM"}


@app.route("/analysis/<int:dataset_id>", methods=["GET", "POST"])
def analysis(dataset_id):
    form = AnalysisForm()
    print(request.method)
    if form.validate_on_submit():
        flash("Analysis started successfully!", "success")
        return redirect(url_for("home"))

    name = mapping[dataset_id]
    return render_template("analysis.html", title="Analysis for " + name, form=form)


if __name__ == '__main__':
    app.run(debug=True)
