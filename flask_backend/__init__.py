from concurrent.futures.thread import ThreadPoolExecutor

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SECRET_KEY"] = "4c52401121326497c5baeae8f039a0c3"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
executor = ThreadPoolExecutor(2)


