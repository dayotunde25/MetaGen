#!/bin/bash

echo "Setting up AIMetaHarvest..."

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )) || (( $(echo "$python_version >= 3.11" | bc -l) )); then
    echo "Error: Python version must be between 3.8 and 3.10"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p uploads/temp
mkdir -p app/cache/checkpoints
mkdir -p app/cache/search
mkdir -p app/cache/tasks
mkdir -p instance
mkdir -p logs

# Download NLP models
echo "Downloading SpaCy models..."
python -m spacy download en_core_web_md

echo "Downloading FLAN-T5 model (this may take a while)..."
python -c "
from transformers import T5Tokenizer, T5ForConditionalGeneration
print('Downloading FLAN-T5 tokenizer...')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
print('Downloading FLAN-T5 model...')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
print('FLAN-T5 model downloaded successfully')
"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Database Configuration
MONGODB_URI=mongodb://localhost:27017/dataset_metadata_manager

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Flask Configuration
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
FLASK_ENV=development

# Upload Configuration
MAX_CONTENT_LENGTH=5368709120
UPLOAD_FOLDER=uploads

# Processing Configuration
ENABLE_BACKGROUND_PROCESSING=true
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Optional: Free AI API Keys
# MISTRAL_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
EOL
fi

echo "Setup complete! Please ensure MongoDB and Redis are installed and running."
echo "Run 'source venv/bin/activate' to activate the virtual environment"
echo "Then run 'python run.py' to start the application"
