"""
Routes package initialization.
"""

from flask import Blueprint

from app.routes.auth import auth_bp
from app.routes.main import main_bp
from app.routes.datasets import datasets_bp
from app.routes.reports import reports_bp

def register_blueprints(app):
    """
    Register all blueprints with the Flask application.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(reports_bp)