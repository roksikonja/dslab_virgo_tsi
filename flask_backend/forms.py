from numbers import Real

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import StringField, SubmitField, FileField, SelectField, FloatField, TextAreaField
from wtforms.validators import InputRequired, NumberRange, Length, ValidationError

from dslab_virgo_tsi.base import ExposureMethod, CorrectionMethod


class DictionaryRequired:
    def __call__(self, form, field):
        data = field.data
        try:
            dictionary = eval(data)
            if not isinstance(dictionary, dict):
                raise Exception
        except Exception:
            raise ValidationError(u'Input must be a valid dictionary.')

        for key in dictionary:
            value = dictionary[key]

            if key == "NUM_INDUCING_POINTS":
                if not isinstance(value, int) or value < 1:
                    raise ValidationError(u'Invalid value for NUM_INDUCING_POINTS.')

            elif key == "WINDOW":
                if not isinstance(value, Real) or value < 0:
                    raise ValidationError(u'Invalid value for WINDOW.')

            elif key == "POINTS_IN_WINDOW":
                if not isinstance(value, int) or value <= 0:
                    raise ValidationError(u'Invalid value for POINTS_IN_WINDOW.')

            elif key == "WINDOW_FRACTION":
                if not isinstance(value, Real) or value < 1:
                    raise ValidationError(u'Invalid value for WINDOW_FRACTION.')

            else:
                raise ValidationError(f'Unknown key: {key}.')


class NewDataForm(FlaskForm):
    name = StringField("Dataset Name", validators=[Length(max=100)])
    file = FileField("Dataset File", validators=[InputRequired(), FileAllowed(['csv', 'txt'])])
    exposure_method = SelectField("Exposure Method",
                                  choices=[(ExposureMethod.NUM_MEASUREMENTS.value,
                                            ExposureMethod.NUM_MEASUREMENTS.value),
                                           (ExposureMethod.EXPOSURE_SUM.value,
                                            ExposureMethod.EXPOSURE_SUM.value)])
    outlier_fraction = FloatField("Outlier Fraction", validators=[NumberRange(0.0, 1.0), InputRequired()], default=0.0)
    submit = SubmitField("Import")


class AnalysisForm(FlaskForm):
    model = SelectField("Model",
                        choices=[("EXP", "Exponential model"),
                                 ("EXP_LIN", "Exponential linear model"),
                                 ("SPLINE", "Spline model"),
                                 ("ISOTONIC", "Isotonic model"),
                                 ("SMOOTH_MONOTONIC", "Smooth monotone regression model")])
    model_params = TextAreaField("Model Parameters", validators=[DictionaryRequired()],
                                 default='{\n\t"NUM_INDUCING_POINTS": 1000,\n\t"WINDOW": 100,'
                                         '\n\t"POINTS_IN_WINDOW": 100,\n\t"WINDOW_FRACTION": 5\n}')
    correction = SelectField("Correction method",
                             choices=[(CorrectionMethod.CORRECT_ONE.value,
                                       CorrectionMethod.CORRECT_ONE.value),
                                      (CorrectionMethod.CORRECT_BOTH.value,
                                       CorrectionMethod.CORRECT_BOTH.value)])
    output = SelectField("Output method",
                         choices=[("LOCALGP", "Local Gaussian process"),
                                  ("SVGP", "Sparse variational Gaussian process")])
    submit = SubmitField("Submit")
