from flask_backend.models import db, Constant, DBConsts

db.create_all()
c = Constant(key=DBConsts.RUNNING, value="False")
c2 = Constant(key=DBConsts.JOB_DESCRIPTION, value="")
c3 = Constant(key=DBConsts.JOB_PERCENTAGE, value="0")
c4 = Constant(key=DBConsts.JOB_NAME, value="")

db.session.add(c)
db.session.add(c2)
db.session.add(c3)
db.session.add(c4)
db.session.commit()
