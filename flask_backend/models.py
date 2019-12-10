from flask_backend import db


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    exposure_mode = db.Column(db.String(30), nullable=False)
    outlier_fraction = db.Column(db.Float(), nullable=False)

    def __repr__(self):
        return f"Dataset('{self.name}', '{self.exposure_mode}, {self.outlier_fraction}')"


class Constant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(100))


class DBConsts:
    RUNNING = 1
    JOB_NAME = 2
    JOB_PERCENTAGE = 3
    JOB_DESCRIPTION = 4


class ConstantsAccess:
    @staticmethod
    def get(key):
        return Constant.query.filter_by(key=key).first().value

    @staticmethod
    def set(key, value):
        c = Constant.query.filter_by(key=key).first()
        c.value = value
        db.session.commit()

    @staticmethod
    def get_running():
        return ConstantsAccess.get(DBConsts.RUNNING)

    @staticmethod
    def set_running(value: str):
        ConstantsAccess.set(DBConsts.RUNNING, value)

    @staticmethod
    def get_job_name():
        return ConstantsAccess.get(DBConsts.JOB_NAME)

    @staticmethod
    def set_job_name(value: str):
        ConstantsAccess.set(DBConsts.JOB_NAME, value)

    @staticmethod
    def get_job_percentage():
        return ConstantsAccess.get(DBConsts.JOB_PERCENTAGE)

    @staticmethod
    def set_job_percentage(value: str):
        ConstantsAccess.set(DBConsts.JOB_PERCENTAGE, value)

    @staticmethod
    def get_job_description():
        return ConstantsAccess.get(DBConsts.JOB_DESCRIPTION)

    @staticmethod
    def set_job_description(value: str):
        ConstantsAccess.set(DBConsts.JOB_DESCRIPTION, value)
