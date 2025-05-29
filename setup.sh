#!/bin/bash

# AIMetaHarvest macOS/Linux Setup Script
# This script automates the setup process for macOS and Linux users

set -e  # Exit on any error

echo ""
echo "========================================"
echo "  AIMetaHarvest Local Setup (Unix/Linux)"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}🔧 $1${NC}"
}

# Check if Python 3 is installed
print_info "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION found"
    
    # Check if version is 3.8+
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required"
        echo "Please install Python 3.8+ from https://python.org/downloads/"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org/downloads/"
    exit 1
fi

# Check if MongoDB is installed and running
print_info "Checking MongoDB..."
if command -v mongo &> /dev/null; then
    if mongo --eval "db.adminCommand('ismaster')" &> /dev/null; then
        print_status "MongoDB is running"
    else
        print_warning "MongoDB is installed but not running"
        print_info "Attempting to start MongoDB..."
        
        # Try to start MongoDB based on the system
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew services start mongodb/brew/mongodb-community || {
                    print_warning "Could not start MongoDB with Homebrew"
                }
            fi
        else
            # Linux
            sudo systemctl start mongod || {
                print_warning "Could not start MongoDB service"
            }
        fi
        
        # Check again
        if mongo --eval "db.adminCommand('ismaster')" &> /dev/null; then
            print_status "MongoDB started successfully"
        else
            print_error "MongoDB is not running"
            echo "Please install and start MongoDB manually"
            echo "See LOCAL_SETUP_GUIDE.md for instructions"
            exit 1
        fi
    fi
else
    print_error "MongoDB is not installed"
    echo "Please install MongoDB following the guide in LOCAL_SETUP_GUIDE.md"
    exit 1
fi

# Create virtual environment
print_info "Creating virtual environment..."
if [ -d "venv" ]; then
    print_status "Virtual environment already exists"
else
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
print_info "Installing core Flask dependencies..."
pip install Flask==2.3.3 Flask-Login==0.6.3 Flask-WTF==1.1.1 WTForms==3.0.1
pip install mongoengine==0.27.0 pymongo==4.5.0 Werkzeug==2.3.7 python-dotenv==1.0.0

# Install data processing dependencies
print_info "Installing data processing dependencies..."
pip install pandas==2.1.1 numpy==1.24.3 openpyxl==3.1.2 lxml==4.9.3
pip install reportlab==4.0.4 Pillow==10.0.0 python-dateutil==2.8.2

# Install ML/NLP dependencies
print_info "Installing ML/NLP dependencies (this may take a while)..."
pip install scikit-learn==1.3.0
pip install sentence-transformers==2.2.2
pip install transformers==4.33.2
pip install torch==2.0.1

# Create directories
print_info "Creating directories..."
mkdir -p uploads
mkdir -p app/cache/search
print_status "Directories created"

# Create .env file
print_info "Creating configuration file..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Flask Configuration
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=dataset_metadata_manager

# Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216

# Semantic Search Configuration
ENABLE_SEMANTIC_SEARCH=true
CACHE_DIR=app/cache

# Debug Configuration
DEBUG=true
EOF
    print_status "Configuration file created"
else
    print_status "Configuration file already exists"
fi

# Test installation
print_info "Testing installation..."
python3 -c "
import flask, mongoengine, pandas, numpy
print('✅ Core dependencies imported successfully')
try:
    import sklearn, sentence_transformers
    print('✅ ML/NLP dependencies imported successfully')
except ImportError as e:
    print(f'⚠️  ML/NLP dependencies not available: {e}')
print('✅ Installation test passed')
" || {
    print_error "Installation test failed"
    exit 1
}

echo ""
echo "========================================"
echo "   Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the application: python run.py"
echo "3. Open browser to: http://localhost:5001"
echo "4. Login with: admin / admin123"
echo ""
echo "To activate virtual environment in future sessions:"
echo "    source venv/bin/activate"
echo ""
echo "For detailed instructions, see LOCAL_SETUP_GUIDE.md"
echo ""

# Make the script executable
chmod +x setup.sh
