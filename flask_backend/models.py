from flask_backend import db


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    exposure_mode = db.Column(db.String(30), nullable=False)
    outlier_fraction = db.Column(db.Float(), nullable=False)

    def __repr__(self):
        return f"Dataset('{self.name}', '{self.exposure_mode}, {self.outlier_fraction}')"
