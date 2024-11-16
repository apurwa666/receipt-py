# app/__init__.py
from flask import Flask
from app.config import Config
from app.database import db
from app.routes.auth_routes import auth_bp
from app.routes.receipt_routes import receipt_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(receipt_bp, url_prefix='/receipt')
    
    return app
