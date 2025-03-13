import os
from flask import Flask
from flask_login import current_user

from app.config import Config
from app.models import db, migrate, login_manager
from app.services.dataset_service import get_dataset_service

def create_app(config_class=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    
    # Call config init method
    config_class.init_app(app)
    
    # Initialize dataset service
    get_dataset_service(app.config['UPLOAD_FOLDER'])
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.datasets import datasets_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(api_bp)
    
    # Configure template context
    @app.context_processor
    def inject_user():
        return dict(current_user=current_user)
    
    # Register error handlers
    register_error_handlers(app)
    
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

# Import models to ensure they are registered with SQLAlchemy
from app.models.user import User
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue

# Import render_template for error handlers
from flask import render_template