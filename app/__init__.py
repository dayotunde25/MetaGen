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

    # Initialize FAIR enhancement service
    from app.services.fair_enhancement_service import get_fair_enhancement_service
    get_fair_enhancement_service()

    # Initialize admin user (only if MongoDB is connected)
    try:
        init_admin_user()
    except Exception as e:
        print(f"⚠️  Could not initialize admin user: {e}")

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.datasets import datasets_bp
    from app.routes.reports import reports_bp
    from app.routes.feedback import feedback_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(feedback_bp)

    # Configure template context
    @app.context_processor
    def inject_user():
        return dict(current_user=current_user)

    # Add custom template filters
    @app.template_filter('from_json')
    def from_json_filter(json_str):
        """Convert JSON string to Python object"""
        if not json_str:
            return {}
        try:
            import json
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    @app.template_filter('fair_status_text')
    def fair_status_text_filter(fair_score, fair_compliant=None):
        """Get FAIR status text based on score"""
        if fair_score is None:
            return "Unknown"

        if fair_score >= 80.0:
            return "FAIR Compliant"
        elif fair_score >= 50.0:
            return "Partially Compliant"
        else:
            return "Not Compliant"

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

# Initialize default admin user
def init_admin_user():
    """Initialize default admin user if it doesn't exist"""
    try:
        from app.models.user import User

        # Check if admin user already exists
        admin_user = User.find_by_username('admin')

        if not admin_user:
            # Create default admin user
            admin_user = User.create(
                username='admin',
                email='admin@aimetaharvest.local',
                password='admin123',
                is_admin=True
            )
            print("✅ Default admin user created: admin / admin123")
            return admin_user
        else:
            print("ℹ️  Admin user already exists")
            return admin_user

    except Exception as e:
        print(f"⚠️  Error creating admin user: {e}")
        return None

# Simple sample data initialization function
def init_sample_data():
    """Initialize sample data if needed"""
    pass