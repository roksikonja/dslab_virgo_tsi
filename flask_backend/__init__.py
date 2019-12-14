from concurrent.futures.thread import ThreadPoolExecutor

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from status_utils import Status

app = Flask(__name__)
app.config["SECRET_KEY"] = "4c52401121326497c5baeae8f039a0c3"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
executor = ThreadPoolExecutor(2)
status = Status()


from flask_backend import routes
