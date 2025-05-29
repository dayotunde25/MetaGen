import os
from flask import Flask
from flask_login import current_user
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.config import Config
from app.models import login_manager
from app.services.dataset_service import get_dataset_service

def create_app(config_class=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize MongoDB connection
    import mongoengine
    mongodb_uri = app.config['MONGODB_SETTINGS']['host']
    try:
        mongoengine.connect(host=mongodb_uri, serverSelectionTimeoutMS=5000)
        print(f"Connected to MongoDB: {mongodb_uri}")
    except Exception as e:
        print(f"Warning: Could not connect to MongoDB: {e}")
        print("The app will still start but database operations will fail.")

    # Initialize Flask-Login
    login_manager.init_app(app)

    # Initialize CSRF Protection
    csrf = CSRFProtect(app)

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        try:
            return User.objects(id=user_id).first()
        except:
            return None

    # Call config init method
    config_class.init_app(app)

    # Initialize dataset service
    get_dataset_service(app.config['UPLOAD_FOLDER'])

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.datasets import datasets_bp
    from app.routes.reports import reports_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(reports_bp)

    # Configure template context
    @app.context_processor
    def inject_user():
        return dict(current_user=current_user)

    # Register error handlers
    register_error_handlers(app)

    # Initialize sample data
    try:
        with app.app_context():
            init_sample_data()
    except Exception as e:
        print(f"Warning: Could not initialize sample data: {e}")

    # Note: Background processing service will be added later
    print("File upload functionality enabled")

    return app

def register_error_handlers(app):
    """Register error handlers"""

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errors/500.html'), 500

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('errors/403.html'), 403

# Import models to ensure they are registered with MongoEngine
from app.models.user import User
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue

# Import render_template for error handlers
from flask import render_template

# Simple sample data initialization function
def init_sample_data():
    """Initialize sample data if needed"""
    pass