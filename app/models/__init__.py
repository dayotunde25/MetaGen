"""
Models package initialization.
"""

from flask_login import LoginManager

from app.models.base import db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

__all__ = [
    'db',
    'login_manager',
    'User',
    'Dataset',
    'MetadataQuality',
    'ProcessingQueue'
]