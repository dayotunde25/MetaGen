"""
User model and authentication functionality.
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from app.models.base import Base, db


class User(Base, UserMixin):
    """
    User model for authentication and user management.
    
    Attributes:
        id: Unique identifier
        username: Username for login
        email: Email address
        password_hash: Hashed password
        is_admin: Whether the user is an administrator
        created_at: When the user was created
        last_login: When the user last logged in
    """
    
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    datasets = relationship('Dataset', back_populates='user')
    
    @property
    def password(self):
        """
        Password getter - raises AttributeError.
        """
        raise AttributeError('Password is not a readable attribute')
    
    @password.setter
    def password(self, password):
        """
        Set password hash.
        
        Args:
            password: Plain text password
        """
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """
        Check if password matches hash.
        
        Args:
            password: Plain text password to check
            
        Returns:
            True if password matches
        """
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp."""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    @classmethod
    def find_by_username(cls, username):
        """
        Find user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User or None
        """
        return db.session.query(cls).filter_by(username=username).first()
    
    @classmethod
    def find_by_email(cls, email):
        """
        Find user by email.
        
        Args:
            email: Email to search for
            
        Returns:
            User or None
        """
        return db.session.query(cls).filter_by(email=email).first()
    
    @classmethod
    def create(cls, username, email, password, is_admin=False):
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            is_admin: Whether user is admin
            
        Returns:
            New User instance
        """
        user = cls(username=username, email=email, is_admin=is_admin)
        user.password = password
        db.session.add(user)
        db.session.commit()
        return user
    
    def to_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation (without password)
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at,
            'last_login': self.last_login
        }