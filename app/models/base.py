"""
Base model and database setup.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Base(db.Model):
    """
    Base model class for SQLAlchemy models.
    """
    __abstract__ = True