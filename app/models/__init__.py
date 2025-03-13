"""
Models package initialization.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

from app.models.base import db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue

__all__ = [
    'db',
    'User',
    'Dataset',
    'MetadataQuality',
    'ProcessingQueue'
]