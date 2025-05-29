# Dataset Metadata Manager

A comprehensive web platform for AI dataset management, empowering researchers and data scientists to efficiently source, validate, and explore structured metadata through an intelligent and user-friendly interface.

## Features

- **Dataset Management**: Upload, edit, and delete datasets with comprehensive metadata
- **Quality Assessment**: Automatic quality scoring based on completeness, consistency, and standards compliance
- **FAIR Compliance**: Assessment of datasets against Findable, Accessible, Interoperable, and Reusable principles
- **Health Reports**: Generate comprehensive HTML and PDF reports for dataset quality and compliance
- **Semantic Search**: Natural language processing for intelligent dataset discovery
- **Processing Queue**: Background processing of large datasets with progress tracking

## Installation

### Prerequisites

- Python 3.9+ installed on your system
- PostgreSQL database (optional, SQLite can be used for development)
- pip (Python package manager)
- Git

### Required Python Packages

The application requires the following primary Python packages:

```
Flask==2.3.3
Flask-Login==0.6.2
Flask-Migrate==4.0.5
Flask-SQLAlchemy==3.1.1
Flask-WTF==1.2.1
Werkzeug==2.3.7
WTForms==3.1.1
SQLAlchemy==2.0.21
psycopg2-binary==2.9.7
Jinja2==3.1.2
python-dotenv==1.0.0
email-validator==2.0.0

# NLP requirements
nltk==3.8.1
spacy==3.7.2
numpy==1.26.0
pandas==2.1.1

# Visualization
matplotlib==3.8.0

# PDF generation
pdfkit==1.0.0
wkhtmltopdf==0.2

# Production ready
gunicorn==21.2.0
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/dataset-metadata-manager.git
cd dataset-metadata-manager
```

### Step 2: Create a Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install NLP Models

```bash
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 5: Install wkhtmltopdf for PDF Generation

For PDF report generation, you need to install wkhtmltopdf:

- **Windows**: Download and install from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html)
- **macOS**: `brew install wkhtmltopdf`
- **Ubuntu/Debian**: `sudo apt-get install wkhtmltopdf`

### Step 6: Set Up Environment Variables

Create a `.env` file in the project root with the following content:

```
# Flask configuration
FLASK_APP=app:create_app
FLASK_ENV=development
SECRET_KEY=your-secret-key-change-this

# Database configuration
# SQLite (for development)
DATABASE_URL=sqlite:///app.db

# PostgreSQL (for production)
# DATABASE_URL=postgresql://username:password@localhost:5432/database_name

# Set to 'True' to enable debugging
DEBUG=True

# Upload folder for dataset files
UPLOAD_FOLDER=uploads
```

### Step 7: Initialize the Database

```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

### Step 8: Run the Application

```bash
flask run
```

Access the application at http://127.0.0.1:5000/

## Docker Deployment (Alternative)

If you prefer using Docker, you can use the provided Dockerfile:

```bash
# Build the Docker image
docker build -t dataset-metadata-manager .

# Run the container
docker run -p 5000:5000 -e DATABASE_URL=postgresql://username:password@host:5432/dbname dataset-metadata-manager
```

## Usage Guide

### Adding a Dataset

1. Navigate to the "Add Dataset" page
2. Fill out the dataset information form
3. Upload your dataset file or provide a URL
4. Submit the form to add the dataset to the system

### Generating a Health Report

1. Navigate to a dataset's detail page or quality assessment page
2. Click the "Generate Report" button
3. Choose between HTML or PDF format
4. View or download your comprehensive health report

### Searching Datasets

1. Use the search bar on the Datasets page
2. Enter keywords or phrases related to your desired dataset
3. Filter results by category, data type, or quality metrics
4. Sort results by relevance, date added, or quality score

## Project Structure

```
app/
├── __init__.py             # Application factory and configuration
├── config.py               # Configuration settings
├── forms.py                # WTForms definitions
├── models/                 # Database models
├── routes/                 # Route definitions
├── services/               # Business logic services
│   ├── dataset_service.py  # Dataset processing service
│   ├── metadata_service.py # Metadata generation service
│   ├── nlp_service.py      # Natural language processing service
│   ├── quality_service.py  # Quality assessment service
│   ├── report_generator.py # Report generation service
│   └── quality_scoring/    # Quality scoring algorithms
├── static/                 # Static assets (CSS, JS, images)
└── templates/              # Jinja2 HTML templates
    ├── auth/               # Authentication templates
    ├── datasets/           # Dataset management templates
    ├── main/               # Main page templates
    └── reports/            # Report templates
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.