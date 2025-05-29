# 🚀 AIMetaHarvest Local Setup Guide

This comprehensive guide will help you set up and run the AIMetaHarvest application locally on your machine.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended for semantic search)
- **Storage**: At least 2GB free space
- **Internet**: Required for initial setup and model downloads

### Required Software
1. **Python 3.8+** - [Download from python.org](https://www.python.org/downloads/)
2. **MongoDB** - [Download MongoDB Community Server](https://www.mongodb.com/try/download/community)
3. **Git** - [Download from git-scm.com](https://git-scm.com/downloads)

## 🔧 Step 1: Install MongoDB

### Windows:
1. Download MongoDB Community Server from the official website
2. Run the installer and follow the setup wizard
3. Choose "Complete" installation
4. Install MongoDB as a Windows Service
5. Start MongoDB service:
   ```cmd
   net start MongoDB
   ```

### macOS:
```bash
# Using Homebrew (recommended)
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
brew services start mongodb/brew/mongodb-community
```

### Linux (Ubuntu/Debian):
```bash
# Import MongoDB public key
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Update package list and install
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod
```

### Verify MongoDB Installation:
```bash
# Check if MongoDB is running
mongo --eval "db.adminCommand('ismaster')"
```

## 📥 Step 2: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/dayotunde25/AIMetaHarvest.git

# Navigate to the project directory
cd AIMetaHarvest
```

## 🐍 Step 3: Set Up Python Environment

### Create Virtual Environment:

#### Windows:
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Verify Virtual Environment:
```bash
# Check Python version
python --version

# Check pip version
pip --version
```

## 📦 Step 4: Install Dependencies

### Install Core Dependencies:
```bash
# Upgrade pip
pip install --upgrade pip

# Install Flask and core dependencies
pip install Flask==2.3.3
pip install Flask-Login==0.6.3
pip install Flask-WTF==1.1.1
pip install WTForms==3.0.1
pip install mongoengine==0.27.0
pip install pymongo==4.5.0
pip install Werkzeug==2.3.7
pip install python-dotenv==1.0.0
```

### Install Data Processing Dependencies:
```bash
# Data processing and analysis
pip install pandas==2.1.1
pip install numpy==1.24.3
pip install openpyxl==3.1.2
pip install lxml==4.9.3
```

### Install Semantic Search Dependencies:
```bash
# Machine Learning and NLP
pip install scikit-learn==1.3.0
pip install sentence-transformers==2.2.2
pip install transformers==4.33.2
pip install torch==2.0.1
```

### Install Additional Dependencies:
```bash
# PDF generation and utilities
pip install reportlab==4.0.4
pip install Pillow==10.0.0
pip install python-dateutil==2.8.2
```

### Alternative: Install from Requirements File
If you prefer to install all dependencies at once:
```bash
# Create a requirements.txt file with all dependencies
pip install -r semantic_search_requirements.txt

# Install additional core dependencies
pip install Flask Flask-Login Flask-WTF mongoengine pandas numpy reportlab
```

## ⚙️ Step 5: Configure the Application

### Create Environment Configuration:
Create a `.env` file in the project root:

```bash
# Create .env file
touch .env  # Linux/macOS
# or create manually on Windows
```

Add the following configuration to `.env`:
```env
# Flask Configuration
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here-change-this-in-production

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=dataset_metadata_manager

# Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216  # 16MB max file size

# Semantic Search Configuration
ENABLE_SEMANTIC_SEARCH=true
CACHE_DIR=app/cache

# Debug Configuration
DEBUG=true
```

### Create Upload Directory:
```bash
# Create uploads directory
mkdir uploads

# Create cache directory for semantic search
mkdir -p app/cache/search
```

## 🗄️ Step 6: Initialize Database

### Start MongoDB (if not already running):

#### Windows:
```cmd
net start MongoDB
```

#### macOS:
```bash
brew services start mongodb/brew/mongodb-community
```

#### Linux:
```bash
sudo systemctl start mongod
```

### Verify Database Connection:
```bash
# Connect to MongoDB shell
mongo

# In MongoDB shell, create the database
use dataset_metadata_manager

# Exit MongoDB shell
exit
```

## 🚀 Step 7: Run the Application

### Start the Flask Application:
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run the application
python run.py
```

### Expected Output:
```
 * Running on http://127.0.0.1:5001
 * Debug mode: on
 * MongoDB connected: mongodb://localhost:27017/dataset_metadata_manager
 * File upload enabled: uploads/
 * Semantic search: Initializing...
 * Application ready!
```

## 🌐 Step 8: Access the Application

1. **Open your web browser**
2. **Navigate to**: `http://127.0.0.1:5001` or `http://localhost:5001`
3. **Default Admin Credentials**:
   - Username: `admin`
   - Password: `admin123`

## 🔧 Step 9: Initialize Semantic Search (Optional but Recommended)

### Method 1: Via Web Interface
1. Login as admin
2. Navigate to: `http://localhost:5001/admin/reindex-search`
3. Wait for indexing to complete

### Method 2: Via Python Console
```python
# Open Python console in project directory
python

# Run indexing
from app.services.dataset_service import get_dataset_service
service = get_dataset_service('uploads')
service.reindex_all_datasets()
```

## 🧪 Step 10: Test the Application

### Upload Test Dataset:
1. **Login** to the application
2. **Navigate** to "Upload Dataset"
3. **Upload** a sample CSV file (create one with sample data)
4. **Monitor** the processing progress
5. **Check** the dashboard for results

### Test Search Functionality:
1. **Navigate** to "Search Datasets"
2. **Try** semantic searches like:
   - "climate data"
   - "machine learning"
   - "financial information"
3. **Verify** results are returned and ranked properly

## 🛠️ Troubleshooting

### Common Issues and Solutions:

#### MongoDB Connection Issues:
```bash
# Check if MongoDB is running
# Windows:
net start MongoDB

# macOS:
brew services list | grep mongodb

# Linux:
sudo systemctl status mongod
```

#### Python Package Issues:
```bash
# Reinstall problematic packages
pip uninstall package-name
pip install package-name

# Clear pip cache
pip cache purge
```

#### Port Already in Use:
```bash
# Find process using port 5001
# Windows:
netstat -ano | findstr :5001

# macOS/Linux:
lsof -i :5001

# Kill the process or change port in run.py
```

#### Semantic Search Model Download Issues:
```bash
# Manual model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### AttributeError in AI Standards Service:
If you see `AttributeError: 'AIStandardsService' object has no attribute '_get_model_card_template'`:
```bash
# This indicates missing methods in the AI standards service
# The issue should be resolved in the latest version
# If it persists, try:
git pull origin main  # Get latest updates
python run.py         # Restart the application
```

#### File Upload Issues:
```bash
# Check upload directory permissions
# Linux/macOS:
chmod 755 uploads

# Windows: Right-click uploads folder → Properties → Security → Edit permissions
```

## 📊 Performance Optimization

### For Better Performance:
1. **Increase RAM allocation** for MongoDB
2. **Use SSD storage** for better I/O performance
3. **Install optional dependencies**:
   ```bash
   pip install faiss-cpu  # Faster similarity search
   pip install accelerate  # Faster BERT inference
   ```

### For Production Deployment:
1. **Change SECRET_KEY** in `.env`
2. **Set DEBUG=false**
3. **Use production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5001 run:app
   ```

## 🎉 Success!

If you've followed all steps correctly, you should now have:

✅ **AIMetaHarvest running locally**  
✅ **MongoDB database connected**  
✅ **Semantic search enabled**  
✅ **File upload working**  
✅ **Quality assessment active**  
✅ **FAIR compliance checking**  
✅ **AI standards assessment**  

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Review application logs** in the terminal
3. **Verify all dependencies** are installed correctly
4. **Check MongoDB connection** and service status
5. **Ensure all directories** have proper permissions

## 🐳 Docker Setup (Alternative)

For users who prefer containerized deployment:

### Prerequisites:
- Docker and Docker Compose installed

### Quick Docker Setup:
```bash
# Clone repository
git clone https://github.com/dayotunde25/AIMetaHarvest.git
cd AIMetaHarvest

# Start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### Access Application:
- **URL**: http://localhost:5001
- **Credentials**: admin / admin123

### Docker Management:
```bash
# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# View MongoDB logs
docker-compose logs mongodb
```

## 🔄 Updating the Application

### Standard Installation:
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart the application
python run.py
```

### Docker Installation:
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart containers
docker-compose down
docker-compose up --build -d
```

---

**🎊 Congratulations! You now have AIMetaHarvest running locally with full functionality including semantic search, quality assessment, and AI standards compliance!**
