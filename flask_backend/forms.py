from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, SelectField, FloatField
from wtforms.validators import InputRequired, NumberRange, Length


class NewDataForm(FlaskForm):
    name = StringField("Dataset Name", validators=[Length(max=100)])
    file = FileField("Dataset File", validators=[InputRequired()])
    exposure_mode = SelectField("Exposure Mode",
                                choices=[("Exposure sum", "Exposure sum"),
                                         ("Num. measurements", "Number of measurements")])
    outlier_fraction = FloatField("Outlier Fraction", validators=[NumberRange(0.0, 1.0), InputRequired()], default=0.0)
    submit = SubmitField("Import")


class AnalysisForm(FlaskForm):
    model = SelectField("Model",
                        choices=[("EXP_MODEL", "Exponential model"),
                                 ("EXP_LIN_MODEL", "Exponential linear model"),
                                 ("SPLINE", "Spline model"),
                                 ("ISOTONIC", "Isotonic model"),
                                 ("SMOOTH MONOTONE", "Smooth monotone regression model")])
    model_params = StringField("Model Parameters")
    correction = SelectField("Correction method",
                             choices=[("CORRECT_ONE", "Correct one"),
                                      ("CORRECT_BOTH", "Correct both")])
    output = SelectField("Output method",
                         choices=[("GP", "Local Gaussian process"),
                                  ("SVGP", "Sparse variational Gaussian process")])
    submit = SubmitField("Submit")
