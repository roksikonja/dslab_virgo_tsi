from flask_backend.models import db, Constant, DBConsts

db.create_all()
c = Constant(key=DBConsts.IMPORT_RUNNING, value=False)
c2 = Constant(key=DBConsts.ANALYSIS_RUNNING, value=False)

db.session.add(c)
db.session.add(c2)
db.session.commit()
