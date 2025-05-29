@echo off
REM AIMetaHarvest Windows Setup Script
REM This script automates the setup process for Windows users

echo.
echo ========================================
echo   AIMetaHarvest Local Setup (Windows)
echo ========================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found!
python --version

REM Check if MongoDB is running
echo.
echo Checking MongoDB...
mongo --eval "db.adminCommand('ismaster')" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: MongoDB is not running
    echo Attempting to start MongoDB service...
    net start MongoDB >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Could not start MongoDB
        echo Please install MongoDB and start the service manually
        echo See LOCAL_SETUP_GUIDE.md for instructions
        pause
        exit /b 1
    )
)

echo MongoDB is running!

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created!
)

REM Activate virtual environment and install dependencies
echo.
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing core dependencies...
pip install Flask==2.3.3 Flask-Login==0.6.3 Flask-WTF==1.1.1 WTForms==3.0.1
pip install mongoengine==0.27.0 pymongo==4.5.0 Werkzeug==2.3.7 python-dotenv==1.0.0

echo Installing data processing dependencies...
pip install pandas==2.1.1 numpy==1.24.3 openpyxl==3.1.2 lxml==4.9.3
pip install reportlab==4.0.4 Pillow==10.0.0 python-dateutil==2.8.2

echo Installing ML/NLP dependencies (this may take a while)...
pip install scikit-learn==1.3.0
pip install sentence-transformers==2.2.2
pip install transformers==4.33.2
pip install torch==2.0.1

REM Create directories
echo.
echo Creating directories...
if not exist uploads mkdir uploads
if not exist app\cache mkdir app\cache
if not exist app\cache\search mkdir app\cache\search

REM Create .env file
echo.
echo Creating configuration file...
if not exist .env (
    echo # Flask Configuration > .env
    echo FLASK_APP=run.py >> .env
    echo FLASK_ENV=development >> .env
    echo SECRET_KEY=dev-secret-key-change-in-production >> .env
    echo. >> .env
    echo # MongoDB Configuration >> .env
    echo MONGODB_HOST=localhost >> .env
    echo MONGODB_PORT=27017 >> .env
    echo MONGODB_DB=dataset_metadata_manager >> .env
    echo. >> .env
    echo # Upload Configuration >> .env
    echo UPLOAD_FOLDER=uploads >> .env
    echo MAX_CONTENT_LENGTH=16777216 >> .env
    echo. >> .env
    echo # Semantic Search Configuration >> .env
    echo ENABLE_SEMANTIC_SEARCH=true >> .env
    echo CACHE_DIR=app/cache >> .env
    echo. >> .env
    echo # Debug Configuration >> .env
    echo DEBUG=true >> .env
    
    echo Configuration file created!
) else (
    echo Configuration file already exists
)

echo.
echo ========================================
echo   Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. The virtual environment is already activated
echo 2. Run the application: python run.py
echo 3. Open browser to: http://localhost:5001
echo 4. Login with: admin / admin123
echo.
echo To activate virtual environment in future sessions:
echo    venv\Scripts\activate
echo.
echo For detailed instructions, see LOCAL_SETUP_GUIDE.md
echo.
pause
