from flask_backend import db


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    exposure_method = db.Column(db.String(30), nullable=False)
    outlier_fraction = db.Column(db.Float(), nullable=False)
    dataset_location = db.Column(db.String(1000), nullable=False)
    pickle_location = db.Column(db.String(1000), nullable=False)

    def __repr__(self):
        return f"Dataset('{self.name}', '{self.exposure_method}, {self.outlier_fraction}, {self.location}')"
