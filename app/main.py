from flask import Flask
from config import Config
from database import db
from routes.auth_routes import auth_bp
from routes.receipt_routes import receipt_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(receipt_bp, url_prefix='/receipt')

    with app.app_context():
        db.create_all()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
