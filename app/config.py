import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-development'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    @staticmethod
    def init_app(app):
        """Initialize application with this configuration"""
        # Create upload folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    

class ProductionConfig(Config):
    """Production configuration"""
    # Use secure cookies in production
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    
    # Stronger CSRF protection
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour
    
    @classmethod
    def init_app(cls, app):
        """Initialize production application"""
        # Call parent class init_app
        super().init_app(app)
        
        # Production specific initialization
        # e.g. log to syslog
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Create logs directory
        if not os.path.exists('logs'):
            os.mkdir('logs')
            
        # Configure file handler
        file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')


# Dictionary of configurations
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}