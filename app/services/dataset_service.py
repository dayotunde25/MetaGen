import os
import json
import csv
import pandas as pd
import requests
from datetime import datetime
import tempfile
from urllib.parse import urlparse
from werkzeug.utils import secure_filename

from app.models import db
from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue, MetadataQuality
from app.services.nlp_service import nlp_service

class DatasetService:
    """Service for dataset operations including fetching, processing, and analysis"""
    
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
    
    def fetch_dataset(self, url):
        """Fetch a dataset from a URL and save it locally"""
        parsed_url = urlparse(url)
        filename = secure_filename(os.path.basename(parsed_url.path))
        
        if not filename:
            filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        if not ext:
            # Try to determine format from Content-Type
            response = requests.head(url)
            content_type = response.headers.get('Content-Type', '')
            
            if 'csv' in content_type:
                ext = '.csv'
            elif 'json' in content_type:
                ext = '.json'
            elif 'xml' in content_type:
                ext = '.xml'
            else:
                ext = '.dat'  # Generic data file
            
            filename = f"{filename}{ext}"
        
        # Create full path
        filepath = os.path.join(self.upload_folder, filename)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_size = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                file_size += len(chunk)
        
        # Determine format from extension
        format_type = ext[1:].lower()  # Remove the leading dot
        
        return {
            'path': filepath,
            'format': format_type,
            'size': file_size
        }
    
    def process_dataset(self, file_info, format_hint=None):
        """Process a dataset file to extract information and structure"""
        file_path = file_info['path']
        format_type = format_hint or file_info['format']
        
        try:
            if format_type == 'csv':
                return self._process_csv(file_path)
            elif format_type == 'json':
                return self._process_json(file_path)
            elif format_type == 'xml':
                return self._process_xml(file_path)
            else:
                return self._process_generic_file(file_path)
        except Exception as e:
            return {
                'error': str(e),
                'schema': None,
                'sample_data': None,
                'record_count': 0
            }
    
    def _process_csv(self, file_path):
        """Process a CSV file to extract schema and sample data"""
        df = pd.read_csv(file_path, nrows=100)  # Read just first 100 rows for analysis
        
        # Get column info
        schema = []
        for col in df.columns:
            data_type = str(df[col].dtype)
            schema.append({
                'name': col,
                'type': data_type,
                'example': str(df[col].iloc[0]) if not df[col].empty else None
            })
        
        # Get total row count
        with open(file_path, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header row
        
        return {
            'schema': schema,
            'sample_data': df.head(5).to_dict('records'),
            'record_count': row_count
        }
    
    def _process_json(self, file_path):
        """Process a JSON file to extract schema and sample data"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both array and object formats
        if isinstance(data, list):
            sample_data = data[:5]
            record_count = len(data)
            
            # Infer schema from first item
            schema = []
            if record_count > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    for key, value in first_item.items():
                        schema.append({
                            'name': key,
                            'type': type(value).__name__,
                            'example': str(value) if value is not None else None
                        })
        else:
            sample_data = data
            record_count = 1
            
            # Extract schema
            schema = []
            for key, value in data.items():
                schema.append({
                    'name': key,
                    'type': type(value).__name__,
                    'example': str(value) if value is not None else None
                })
        
        return {
            'schema': schema,
            'sample_data': sample_data,
            'record_count': record_count
        }
    
    def _process_xml(self, file_path):
        """Process an XML file (simplified)"""
        # For XML processing, we'd normally use a library like lxml
        # This is simplified for demonstration
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Count number of elements as a rough estimate
        tags = content.count('</')
        
        return {
            'schema': [{'name': 'xml_structure', 'type': 'complex', 'example': 'XML structure'}],
            'sample_data': {'content': content[:500] + '...' if len(content) > 500 else content},
            'record_count': tags
        }
    
    def _process_generic_file(self, file_path):
        """Process an unknown file format"""
        # For unknown formats, just get file stats
        stats = os.stat(file_path)
        
        with open(file_path, 'r', errors='ignore') as f:
            sample = f.read(1000)  # Read first 1000 chars
        
        return {
            'schema': [{'name': 'content', 'type': 'unknown', 'example': sample[:100]}],
            'sample_data': {'preview': sample},
            'record_count': 1  # Can't determine record count
        }
    
    def assess_quality(self, processed_data, dataset):
        """Assess the quality of the dataset and its metadata"""
        issues = []
        score = 70  # Start with a base score
        
        # Check for schema completeness
        schema = processed_data.get('schema', [])
        if not schema:
            issues.append("Schema could not be determined")
            score -= 15
        
        # Check for sufficient records
        record_count = processed_data.get('record_count', 0)
        if record_count < 10:
            issues.append("Dataset has very few records")
            score -= 10
        
        # Check for metadata completeness
        if not dataset.description:
            issues.append("Dataset is missing a description")
            score -= 10
        
        if not dataset.tags:
            issues.append("Dataset has no tags")
            score -= 5
        
        # Prevent negative scores
        score = max(0, score)
        
        return {
            'score': score,
            'issues': issues
        }
    
    def add_to_processing_queue(self, dataset_id, priority=1):
        """Add a dataset to the processing queue"""
        existing_queue = ProcessingQueue.query.filter_by(dataset_id=dataset_id).first()
        
        if existing_queue:
            # Update existing queue item
            existing_queue.status = 'queued'
            existing_queue.progress = 0
            existing_queue.priority = priority
            existing_queue.error_message = None
            existing_queue.started_at = None
            existing_queue.completed_at = None
            existing_queue.updated_at = datetime.utcnow()
            
            db.session.commit()
            return existing_queue
        else:
            # Create new queue item
            queue_item = ProcessingQueue(
                dataset_id=dataset_id,
                status='queued',
                priority=priority
            )
            
            db.session.add(queue_item)
            db.session.commit()
            return queue_item
    
    def process_queue_item(self, queue_item):
        """Process a dataset from the queue"""
        if not queue_item or queue_item.status == 'completed':
            return False
        
        # Mark as processing
        queue_item.status = 'processing'
        queue_item.started_at = datetime.utcnow()
        queue_item.progress = 10
        db.session.commit()
        
        try:
            # Get the dataset
            dataset = Dataset.query.get(queue_item.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset with ID {queue_item.dataset_id} not found")
            
            # Update dataset status
            dataset.status = 'processing'
            db.session.commit()
            
            # Fetch the dataset if needed
            if dataset.source_url and not dataset.file_path:
                queue_item.progress = 20
                db.session.commit()
                
                file_info = self.fetch_dataset(dataset.source_url)
                dataset.file_path = file_info['path']
                dataset.format = file_info.get('format', dataset.format)
                dataset.size = f"{file_info.get('size', 0) / (1024 * 1024):.2f} MB"
                db.session.commit()
            
            # Process the dataset
            queue_item.progress = 40
            db.session.commit()
            
            processed_data = self.process_dataset({
                'path': dataset.file_path,
                'format': dataset.format
            })
            
            # Update record count if available
            if 'record_count' in processed_data:
                dataset.record_count = processed_data['record_count']
            
            queue_item.progress = 60
            db.session.commit()
            
            # Assess quality
            quality_assessment = self.assess_quality(processed_data, dataset)
            
            # Create or update metadata quality record
            metadata_quality = MetadataQuality.query.filter_by(dataset_id=dataset.id).first()
            
            if not metadata_quality:
                metadata_quality = MetadataQuality(
                    dataset_id=dataset.id,
                    quality_score=quality_assessment['score'],
                    completeness=quality_assessment['score'],  # Simplified
                    consistency=quality_assessment['score'],  # Simplified
                    findable_score=65,  # Default values
                    accessible_score=70,
                    interoperable_score=60,
                    reusable_score=65,
                )
                db.session.add(metadata_quality)
            else:
                metadata_quality.quality_score = quality_assessment['score']
                metadata_quality.completeness = quality_assessment['score']  # Simplified
                metadata_quality.consistency = quality_assessment['score']  # Simplified
                metadata_quality.issues_list = quality_assessment['issues']
                
            queue_item.progress = 80
            db.session.commit()
            
            # Generate and suggest tags if needed
            if not dataset.tags:
                suggested_tags = nlp_service.suggest_tags(dataset.description or dataset.title)
                dataset.tags_list = suggested_tags
            
            # Mark as completed
            queue_item.progress = 100
            queue_item.status = 'completed'
            queue_item.completed_at = datetime.utcnow()
            
            dataset.status = 'completed'
            
            db.session.commit()
            return True
            
        except Exception as e:
            # Handle error
            queue_item.status = 'failed'
            queue_item.error_message = str(e)
            
            dataset = Dataset.query.get(queue_item.dataset_id)
            if dataset:
                dataset.status = 'failed'
            
            db.session.commit()
            return False
    
    def search_datasets(self, query, filters=None, sort_by='created_at', limit=10, offset=0):
        """Search datasets using NLP-based semantic search"""
        # Start with a base query
        base_query = Dataset.query
        
        # Apply filters if provided
        if filters:
            if 'category' in filters and filters['category']:
                base_query = base_query.filter(Dataset.category == filters['category'])
            
            if 'data_type' in filters and filters['data_type']:
                base_query = base_query.filter(Dataset.data_type == filters['data_type'])
            
            if 'status' in filters and filters['status']:
                base_query = base_query.filter(Dataset.status == filters['status'])
        
        # Get all potential datasets
        all_datasets = base_query.all()
        
        if not query:
            # No query, just apply sorting
            if sort_by == 'title':
                results = sorted(all_datasets, key=lambda d: d.title)
            elif sort_by == 'updated_at':
                results = sorted(all_datasets, key=lambda d: d.updated_at or d.created_at, reverse=True)
            else:  # Default to created_at
                results = sorted(all_datasets, key=lambda d: d.created_at, reverse=True)
            
            # Apply pagination
            return results[offset:offset+limit]
        
        # Prepare text for semantic search
        documents = [f"{d.title} {d.description or ''}" for d in all_datasets]
        document_ids = [d.id for d in all_datasets]
        
        # Perform semantic search
        search_results = nlp_service.semantic_search(query, documents, document_ids)
        
        # Get datasets by ID with similarity scores
        results = []
        for result in search_results:
            dataset = next((d for d in all_datasets if d.id == result['id']), None)
            if dataset:
                # Attach similarity score
                dataset.similarity = result['similarity']
                results.append(dataset)
        
        # Apply pagination
        return results[offset:offset+limit]

# Initialize singleton service
dataset_service = None

def get_dataset_service(upload_folder):
    global dataset_service
    if dataset_service is None:
        dataset_service = DatasetService(upload_folder)
    return dataset_service