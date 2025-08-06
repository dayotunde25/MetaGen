# AI Meta Harvest - Complete Installation Guide

## Overview
AI Meta Harvest is a comprehensive dataset metadata management system with advanced NLP capabilities, semantic search, and AI-powered description generation. This guide will help you set up the application on a new system.

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+ (for NLP models)
- **Storage**: Minimum 10GB free space (for models and data)
- **CPU**: Multi-core processor recommended for background processing

### Software Requirements
- **Python**: 3.8 or higher (3.10 recommended)
- **MongoDB**: 4.4 or higher
- **Redis**: 6.0 or higher (for background processing)
- **Git**: For cloning the repository

## Step 1: System Dependencies

### Windows
```bash
# Install Python 3.10 from python.org
# Install MongoDB Community Edition
# Install Redis for Windows
# Install Git for Windows
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
sudo apt install mongodb redis-server git
sudo systemctl start mongodb
sudo systemctl start redis-server
sudo systemctl enable mongodb
sudo systemctl enable redis-server
```

### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 mongodb-community redis git
brew services start mongodb-community
brew services start redis
```

## Step 2: Clone and Setup Project

```bash
# Clone the repository
git clone <repository-url>
cd AIMetaHarvest

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Step 3: Download Required NLP Models

### SpaCy Models
```bash
# Download spaCy models (choose based on your needs)
python -m spacy download en_core_web_md    # Medium model (recommended)
python -m spacy download en_core_web_lg    # Large model (better accuracy)
python -m spacy download en_core_web_trf   # Transformer model (best accuracy)
```

### FLAN-T5 Model (Offline Description Generation)
The FLAN-T5 model will be automatically downloaded on first use. To pre-download:

```bash
python -c "
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
print('FLAN-T5 Base model downloaded successfully')
"
```

## Step 4: Environment Configuration

Create a `.env` file in the project root:

```bash
# Database Configuration
MONGODB_URI=mongodb://localhost:27017/dataset_metadata_manager

# Redis Configuration (for background processing)
REDIS_URL=redis://localhost:6379/0

# Free AI API Keys (Optional - for enhanced descriptions)
MISTRAL_API_KEY=your_mistral_key_here
GROQ_API_KEY=your_groq_key_here

# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Upload Configuration
MAX_CONTENT_LENGTH=5368709120  # 5GB
UPLOAD_FOLDER=uploads

# Processing Configuration
ENABLE_BACKGROUND_PROCESSING=true
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Step 5: Database Setup

```bash
# Start MongoDB (if not already running)
# Windows: Start MongoDB service from Services
# Linux: sudo systemctl start mongodb
# macOS: brew services start mongodb-community

# The application will automatically create the database and collections
# Admin user will be created automatically on first run
```

## Step 6: API Keys Setup (Optional)

### Mistral AI (Free Tier Available)
1. Visit https://console.mistral.ai/
2. Create account and get API key
3. Add to `.env`: `MISTRAL_API_KEY=your_key_here`

### Groq (Free Tier Available)
1. Visit https://console.groq.com/
2. Create account and get API key  
3. Add to `.env`: `GROQ_API_KEY=your_key_here`

**Note**: Free AI keys are optional. The app works with offline FLAN-T5 model if no keys provided.

## Step 7: Start the Application

### Method 1: Simple Start (Development)
```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Start the application
python run.py
```

### Method 2: With Background Processing (Recommended)
```bash
# Terminal 1: Start Celery Worker
# Windows:
start_celery_worker.bat
# Linux/macOS:
chmod +x start_celery_worker.sh
./start_celery_worker.sh

# Terminal 2: Start Flask Application
python run.py
```

## Step 8: Access the Application

1. Open browser and go to: `http://127.0.0.1:5001`
2. Login with default admin credentials:
   - **Username**: `admin`
   - **Password**: `admin123`
3. Change admin password in user settings

## Features Overview

### Core Features
- **Dataset Upload**: Support for CSV, JSON, XML, XLSX, XLS, ZIP collections
- **Automatic Processing**: NLP analysis, quality scoring, metadata generation
- **Semantic Search**: Advanced search with BERT embeddings and TF-IDF
- **FAIR Compliance**: Automatic FAIR principle assessment
- **Visualizations**: Interactive charts and data analysis
- **Export**: PDF, Markdown, JSON metadata export

###  Features
- **Description Generation**: Using Mistral AI, Groq, or offline FLAN-T5
- **Keyword Extraction**: Advanced NLP with BERT and spaCy
- **Entity Recognition**: Named entity recognition and classification
- **Use Case Suggestions**: AI-generated potential applications
- **Quality Assessment**: Automated data quality scoring

## Troubleshooting

### Common Issues

#### 1. MongoDB Connection Error
```bash
# Check if MongoDB is running
# Windows: Check Services
# Linux: sudo systemctl status mongodb
# macOS: brew services list | grep mongodb
```

#### 2. Redis Connection Error
```bash
# Check if Redis is running
# Windows: Check if Redis service is running
# Linux: sudo systemctl status redis-server
# macOS: brew services list | grep redis
```

#### 3. NLP Models Not Found
```bash
# Reinstall spaCy models
python -m spacy download en_core_web_md --force
```

#### 4. Memory Issues
- Reduce batch sizes in processing
- Use smaller spaCy models (en_core_web_md instead of en_core_web_lg)
- Increase system RAM or swap space

#### 5. Background Processing Not Working
- Ensure Redis is running
- Check Celery worker logs
- Verify CELERY_BROKER_URL in .env

### Performance Optimization

#### For Better Performance:
1. **Use SSD storage** for faster model loading
2. **Increase RAM** for larger datasets
3. **Use GPU** if available (configure PyTorch for CUDA)
4. **Enable background processing** with Celery/Redis

#### For Lower Resource Usage:
1. **Use smaller NLP models** (en_core_web_md)
2. **Disable background processing** (set ENABLE_BACKGROUND_PROCESSING=false)
3. **Reduce batch sizes** in configuration

## Directory Structure

```
AIMetaHarvest/
├── app/                    # Main application package
│   ├── models/            # Database models
│   ├── routes/            # Flask routes
│   ├── services/          # Business logic services
│   ├── static/            # CSS, JS, images
│   ├── templates/         # HTML templates
│   └── uploads/           # File upload directory
├── flan-t5-base/          # Offline FLAN-T5 model (auto-created)
├── migrations/            # Database migrations
├── uploads/               # User uploaded files
├── venv/                  # Virtual environment
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── run.py                # Application entry point
├── celery_app.py         # Celery configuration
└── README.md             # Project documentation
```

## Security Notes

1. **Change default admin password** immediately after setup
2. **Use strong SECRET_KEY** in production
3. **Secure MongoDB** with authentication in production
4. **Use HTTPS** in production deployment
5. **Keep API keys secure** and never commit to version control

## Production Deployment

For production deployment, consider:
- Using Gunicorn or uWSGI instead of Flask dev server
- Setting up proper MongoDB authentication
- Using environment-specific configuration
- Setting up proper logging and monitoring
- Using Docker for containerized deployment

## Support

For issues and questions:
1. Check this installation guide
2. Review application logs
3. Check MongoDB and Redis logs
4. Verify all dependencies are installed correctly

---

**Note**: This application requires significant computational resources for NLP processing. Ensure your system meets the minimum requirements for optimal performance.
