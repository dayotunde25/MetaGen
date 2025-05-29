"""
User model and authentication functionality.
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from mongoengine import Document, StringField, BooleanField, DateTimeField
from datetime import datetime

# No need to import db for MongoEngine


class User(Document, UserMixin):
    """
    User model for authentication and user management.

    Attributes:
        username: Username for login
        email: Email address
        password_hash: Hashed password
        is_admin: Whether the user is an administrator
        created_at: When the user was created
        last_login: When the user last logged in
    """

    username = StringField(max_length=64, required=True, unique=True)
    email = StringField(max_length=120, required=True, unique=True)
    password_hash = StringField(max_length=256, required=True)
    is_admin = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    last_login = DateTimeField()

    meta = {
        'collection': 'users',
        'indexes': ['username', 'email']
    }

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

    def get_id(self):
        """Return the user ID as a string for Flask-Login."""
        return str(self.id)

    def update_last_login(self):
        """Update last login timestamp."""
        self.last_login = datetime.utcnow()
        self.save()

    @classmethod
    def find_by_username(cls, username):
        """
        Find user by username.

        Args:
            username: Username to search for

        Returns:
            User or None
        """
        try:
            return cls.objects(username=username).first()
        except cls.DoesNotExist:
            return None

    @classmethod
    def find_by_email(cls, email):
        """
        Find user by email.

        Args:
            email: Email to search for

        Returns:
            User or None
        """
        try:
            return cls.objects(email=email).first()
        except cls.DoesNotExist:
            return None

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
        user.save()
        return user

    def verify_password(self, password):
        """
        Verify password against stored hash.

        Args:
            password: Plain text password to verify

        Returns:
            True if password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)

    # Flask-Login required methods
    def is_authenticated(self):
        """Return True if user is authenticated."""
        return True

    def is_active(self):
        """Return True if user is active."""
        return True

    def is_anonymous(self):
        """Return False as this is not an anonymous user."""
        return False

    def get_id(self):
        """Return user ID as string for Flask-Login."""
        return str(self.id)

    def to_dict(self):
        """
        Convert to dictionary.

        Returns:
            Dictionary representation (without password)
        """
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at,
            'last_login': self.last_login
        }