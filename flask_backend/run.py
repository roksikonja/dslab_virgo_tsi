from flask import Flask, render_template, flash, redirect, url_for

from forms import NewDataForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "4c52401121326497c5baeae9f039a0c3"
app.config["APPLICATION_ROOT"] = "flask_backend"


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


if __name__ == '__main__':
    app.run(debug=True)
