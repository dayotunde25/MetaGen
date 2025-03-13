import os
from app import create_app
from app.config import config

# Get configuration from environment
config_name = os.getenv('FLASK_CONFIG', 'development')

# Create and configure the app
app = create_app(config[config_name])

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=5000)