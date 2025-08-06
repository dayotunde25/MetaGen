@echo off
setlocal enabledelayedexpansion

echo Setting up AIMetaHarvest...

:: Check Python version
python -c "import sys; version=sys.version_info; exit(1 if version < (3,8) or version >= (3,11) else 0)" 2>nul
if errorlevel 1 (
    echo Error: Python version must be between 3.8 and 3.10
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

:: Create necessary directories
echo Creating directory structure...
mkdir uploads\temp 2>nul
mkdir app\cache\checkpoints 2>nul
mkdir app\cache\search 2>nul
mkdir app\cache\tasks 2>nul
mkdir instance 2>nul
mkdir logs 2>nul

:: Download NLP models
echo Downloading SpaCy models...
python -m spacy download en_core_web_md

echo Downloading FLAN-T5 model (this may take a while)...
python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; print('Downloading FLAN-T5 tokenizer...'); tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base'); print('Downloading FLAN-T5 model...'); model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'); print('FLAN-T5 model downloaded successfully')"

:: Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    (
        echo # Database Configuration
        echo MONGODB_URI=mongodb://localhost:27017/dataset_metadata_manager
        echo.
        echo # Redis Configuration
        echo REDIS_URL=redis://localhost:6379/0
        echo.
        echo # Flask Configuration
        echo SECRET_KEY=!RANDOM!!RANDOM!!RANDOM!!RANDOM!
        echo FLASK_ENV=development
        echo.
        echo # Upload Configuration
        echo MAX_CONTENT_LENGTH=5368709120
        echo UPLOAD_FOLDER=uploads
        echo.
        echo # Processing Configuration
        echo ENABLE_BACKGROUND_PROCESSING=true
        echo CELERY_BROKER_URL=redis://localhost:6379/0
        echo CELERY_RESULT_BACKEND=redis://localhost:6379/0
        echo.
        echo # Optional: Free AI API Keys
        echo # MISTRAL_API_KEY=your_key_here
        echo # GROQ_API_KEY=your_key_here
    ) > .env
)

echo Setup complete! Please ensure MongoDB and Redis are installed and running.
echo Run 'venv\Scripts\activate.bat' to activate the virtual environment
echo Then run 'python run.py' to start the application

pause
