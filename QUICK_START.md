# 🚀 AIMetaHarvest Quick Start Guide

Get AIMetaHarvest running locally in just a few minutes!

## 📋 Prerequisites

Before starting, make sure you have:
- **Python 3.8+** installed
- **MongoDB** installed and running
- **Git** installed (if cloning from repository)

## ⚡ Quick Setup Options

### Option 1: Automated Setup (Recommended)

#### Windows Users:
```cmd
# Clone the repository
git clone https://github.com/dayotunde25/AIMetaHarvest.git
cd AIMetaHarvest

# Run automated setup
setup.bat
```

#### macOS/Linux Users:
```bash
# Clone the repository
git clone https://github.com/dayotunde25/AIMetaHarvest.git
cd AIMetaHarvest

# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

#### Python Setup Script (Cross-platform):
```bash
# Clone the repository
git clone https://github.com/dayotunde25/AIMetaHarvest.git
cd AIMetaHarvest

# Run Python setup script
python setup.py
```

### Option 2: Manual Setup

If automated setup doesn't work, follow the detailed guide in [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md).

## 🚀 Running the Application

After setup is complete:

### 1. Activate Virtual Environment

#### Windows:
```cmd
venv\Scripts\activate
```

#### macOS/Linux:
```bash
source venv/bin/activate
```

### 2. Start the Application
```bash
python run.py
```

### 3. Access the Application
- Open your browser to: **http://localhost:5001**
- Login with default credentials:
  - **Username**: `admin`
  - **Password**: `admin123`

## 🎯 First Steps

Once the application is running:

1. **Upload a Dataset**
   - Click "Upload Dataset"
   - Choose a CSV, JSON, or XML file
   - Watch the real-time processing progress

2. **Explore Search**
   - Go to "Search Datasets"
   - Try semantic searches like "climate data" or "machine learning"

3. **Check Dashboard**
   - View your dataset statistics
   - See featured datasets from the community

4. **Generate Reports**
   - View dataset quality assessments
   - Download FAIR compliance reports

## 🛠️ Troubleshooting

### Common Issues:

#### MongoDB Not Running:
```bash
# Windows
net start MongoDB

# macOS
brew services start mongodb/brew/mongodb-community

# Linux
sudo systemctl start mongod
```

#### Port 5001 Already in Use:
- Check what's using the port
- Kill the process or change the port in `run.py`

#### Python Package Issues:
```bash
# Reinstall dependencies
pip install --upgrade pip
pip install -r semantic_search_requirements.txt
```

#### Virtual Environment Issues:
```bash
# Delete and recreate virtual environment
rm -rf venv  # or rmdir /s venv on Windows
python -m venv venv
# Then rerun setup
```

## 📊 Features Overview

### Core Features:
- ✅ **Multi-format Upload**: CSV, JSON, XML support
- ✅ **Real-time Processing**: Live progress tracking
- ✅ **Quality Assessment**: FAIR compliance scoring
- ✅ **Semantic Search**: BERT + TF-IDF powered search
- ✅ **Data Cleaning**: Automatic data preprocessing
- ✅ **AI Standards**: Ethics and bias assessment
- ✅ **Report Generation**: PDF quality reports

### Advanced Features:
- ✅ **Schema.org Compliance**: Automatic metadata generation
- ✅ **FAIR Principles**: Comprehensive assessment
- ✅ **Bias Detection**: AI fairness evaluation
- ✅ **Data Visualization**: Quality charts and overviews
- ✅ **Community Discovery**: Featured datasets section

## 🔧 Configuration

### Environment Variables (.env):
```env
# Flask Configuration
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=dataset_metadata_manager

# Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216

# Features
ENABLE_SEMANTIC_SEARCH=true
DEBUG=true
```

### Performance Tuning:
```bash
# Install optional performance packages
pip install faiss-cpu  # Faster similarity search
pip install accelerate  # Faster BERT inference
```

## 📚 Documentation

- **[LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)** - Detailed setup instructions
- **[SEMANTIC_SEARCH_SETUP.md](SEMANTIC_SEARCH_SETUP.md)** - Advanced search configuration
- **[README.md](README.md)** - Project overview and features

## 🆘 Getting Help

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Review the detailed setup guide**: [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)
3. **Check application logs** in the terminal
4. **Verify MongoDB is running** and accessible
5. **Ensure all dependencies** are installed correctly

## 🎉 Success Indicators

You'll know everything is working when you see:

```
 * Running on http://127.0.0.1:5001
 * Debug mode: on
 * MongoDB connected: mongodb://localhost:27017/dataset_metadata_manager
 * File upload enabled: uploads/
 * Semantic search: Initializing...
 * Application ready!
```

## 🔄 Updating

To update to the latest version:
```bash
git pull origin main
pip install -r semantic_search_requirements.txt
python run.py
```

---

**🎊 You're all set! Start uploading datasets and exploring the powerful features of AIMetaHarvest!**

## 📞 Support

For additional help:
- Check the comprehensive [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)
- Review error messages in the terminal
- Ensure all prerequisites are properly installed
