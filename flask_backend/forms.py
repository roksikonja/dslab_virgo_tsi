from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, SelectField, FloatField
from wtforms.validators import DataRequired, InputRequired, NumberRange


class NewDataForm(FlaskForm):
    name = StringField("Dataset Name")
    file = FileField("Dataset File", validators=[InputRequired()])
    a_field = StringField("A Field Name", validators=[InputRequired()])
    b_field = StringField("B Field Name", validators=[InputRequired()])
    time_field = StringField("Time Field Name", validators=[InputRequired()])
    exposure_mode = SelectField("Exposure Mode",
                                choices=[("EXPOSURE_SUM", "Exposure sum"),
                                         ("NUM_MEASUREMENTS", "Number of measurements")])
    outlier_fraction = FloatField("Outlier Fraction", validators=[NumberRange(0.0, 1.0), InputRequired()])
    submit = SubmitField("Import")
