from app.models.user import User
from app.database import db

def create_user(username, password):
    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user

def validate_user(username, password):
    user = User.query.filter_by(username=username).first()
    return user if user and user.check_password(password) else None
