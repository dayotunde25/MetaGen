# Dataset Metadata Management Platform

A web application for sourcing, processing, and generating structured metadata for AI research datasets with semantic search capabilities, focused on Education, Health, and Agriculture domains.

## Features

- Dataset management with support for various formats (CSV, JSON, XML)
- Automated metadata generation compliant with Schema.org standards
- FAIR compliance assessment (Findable, Accessible, Interoperable, Reusable)
- Semantic search capabilities powered by NLP
- Quality assessment of datasets and metadata
- Processing queue for handling large datasets
- User authentication and dataset ownership

## Technical Stack

- **Backend**: Flask, SQLAlchemy, Python 3.11
- **Database**: PostgreSQL
- **NLP**: NLTK, spaCy
- **Data Processing**: Pandas, NumPy
- **Frontend**: Jinja2 templates, Bootstrap

## Project Structure

- `app/` - Main application package
  - `__init__.py` - Application factory
  - `config.py` - Configuration settings
  - `models/` - SQLAlchemy database models
  - `routes/` - Web routes and API endpoints
  - `services/` - Business logic services
  - `templates/` - Jinja2 HTML templates
  - `static/` - CSS, JavaScript and other static files
- `app.py` - Application entry point
- `migrations/` - Database migrations

## Core Components

### Models

- `User` - User accounts and authentication
- `Dataset` - Dataset information and metadata
- `MetadataQuality` - Quality assessment and FAIR compliance
- `ProcessingQueue` - Processing queue for datasets

### Services

- `NLPService` - Natural language processing functions
- `DatasetService` - Dataset processing and management
- `MetadataService` - Metadata generation and validation

### Routes

- `auth` - User authentication routes
- `main` - Main application routes
- `datasets` - Dataset management routes
- `api` - API endpoints for AJAX and external access

## Setup and Installation

1. Ensure Python 3.11+ and PostgreSQL are installed
2. Clone the repository
3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set environment variables:
   ```
   export FLASK_APP=app.py
   export FLASK_CONFIG=development
   export DATABASE_URL=postgresql://username:password@localhost/dbname
   ```
6. Initialize the database:
   ```
   flask db upgrade
   ```
7. Run the application:
   ```
   flask run
   ```

## API Endpoints

- `GET /api/stats` - Get system statistics
- `GET /api/datasets` - Get all datasets
- `GET /api/datasets/:id` - Get a specific dataset
- `POST /api/datasets` - Create a new dataset
- `PUT /api/datasets/:id` - Update a dataset
- `DELETE /api/datasets/:id` - Delete a dataset
- `GET /api/search` - Search datasets
- `GET /api/processing-queue` - Get processing queue
- `POST /api/process-dataset/:id` - Process a dataset
- `GET /api/metadata-quality/:datasetId` - Get metadata quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.