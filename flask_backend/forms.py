from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, SelectField, FloatField, IntegerField
from wtforms.validators import InputRequired, NumberRange


class NewDataForm(FlaskForm):
    name = StringField("Dataset Name")
    file = FileField("Dataset File", validators=[InputRequired()])
    a_field = StringField("A Field Name", validators=[InputRequired()])
    b_field = StringField("B Field Name", validators=[InputRequired()])
    time_field = StringField("Time Field Name", validators=[InputRequired()])
    exposure_mode = SelectField("Exposure Mode",
                                choices=[("EXPOSURE SUM", "Exposure sum"),
                                         ("NUM_MEASUREMENTS", "Number of measurements")])
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
    window = IntegerField("Moving Average Window", validators=[InputRequired(), NumberRange(min=1)], default=81)
    submit = SubmitField("Submit")
