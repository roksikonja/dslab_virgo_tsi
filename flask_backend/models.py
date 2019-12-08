from flask_backend import db


class DBConsts:
    IMPORT_RUNNING = 1
    ANALYSIS_RUNNING = 2


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
    value = db.Column(db.Boolean, nullable=False)


class ConstantsAccess:
    @staticmethod
    def get_import():
        return Constant.query.filter_by(key=DBConsts.IMPORT_RUNNING).first().value

    @staticmethod
    def get_analysis():
        return Constant.query.filter_by(key=DBConsts.ANALYSIS_RUNNING).first().value

    @staticmethod
    def set_import(value: bool):
        c = Constant.query.filter_by(key=DBConsts.IMPORT_RUNNING).first()
        c.value = value
        db.session.commit()

    @staticmethod
    def set_analysis(value: bool):
        c = Constant.query.filter_by(key=DBConsts.ANALYSIS_RUNNING).first()
        c.value = value
        db.session.commit()

