"""
Routes package initialization.
"""

from flask import Blueprint

from app.routes.auth import auth
from app.routes.main import main
from app.routes.datasets import datasets
from app.routes.api import api
from app.routes.api_quality import api_quality
from app.routes.reports import reports

def register_blueprints(app):
    """
    Register all blueprints with the Flask application.
    
    Args:
        app: Flask application instance
    """
    app.register_blueprint(auth)
    app.register_blueprint(main)
    app.register_blueprint(datasets)
    app.register_blueprint(api)
    app.register_blueprint(api_quality)
    app.register_blueprint(reports)