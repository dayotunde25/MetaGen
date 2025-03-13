import os
import json
import urllib.request
import pandas as pd
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from werkzeug.utils import secure_filename
import threading

from app.models import db
from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue, MetadataQuality
from app.services.nlp_service import nlp_service
from app.services.metadata_service import metadata_service

class DatasetService:
    """Service for dataset operations including fetching, processing, and analysis"""
    
    def __init__(self, upload_folder):
        """Initialize dataset service"""
        self.upload_folder = upload_folder
        self.tempdir = os.path.join(upload_folder, 'temp')
        
        # Create directories if they don't exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(self.tempdir, exist_ok=True)
    
    def fetch_dataset(self, url):
        """Fetch a dataset from a URL and save it locally"""
        if not url:
            return None
        
        try:
            # Create a unique filename
            filename = secure_filename(os.path.basename(url))
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            local_path = os.path.join(self.tempdir, f"{timestamp}_{filename}")
            
            # Download file
            urllib.request.urlretrieve(url, local_path)
            
            # Determine format from extension
            format = os.path.splitext(filename)[1][1:].lower() or 'unknown'
            
            # Get file size
            size = os.path.getsize(local_path)
            
            return {
                'path': local_path,
                'format': format,
                'size': size
            }
        except Exception as e:
            print(f"Error fetching dataset: {str(e)}")
            return None
    
    def process_dataset(self, file_info, format_hint=None):
        """Process a dataset file to extract information and structure"""
        if not file_info or not os.path.exists(file_info['path']):
            return None
        
        try:
            # Determine format
            format = format_hint or file_info['format']
            
            # Process according to format
            if format in ['csv', 'tsv', 'txt']:
                return self._process_csv(file_info['path'])
            elif format in ['json']:
                return self._process_json(file_info['path'])
            elif format in ['xml']:
                return self._process_xml(file_info['path'])
            else:
                return self._process_generic_file(file_info['path'])
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            return None
    
    def _process_csv(self, file_path):
        """Process a CSV file to extract schema and sample data"""
        try:
            # Read a few lines to determine structure
            with open(file_path, 'r', encoding='utf-8') as file:
                # Try to detect delimiter
                dialect = csv.Sniffer().sniff(file.read(1024))
                file.seek(0)
                
                # Read with pandas
                df = pd.read_csv(file_path, sep=dialect.delimiter, nrows=100)
                
                # Extract schema
                schema = {}
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    schema[col] = {
                        'type': dtype,
                        'sample_values': df[col].dropna().head(5).tolist()
                    }
                
                return {
                    'format': 'csv',
                    'columns': list(df.columns),
                    'schema': schema,
                    'record_count': len(df),
                    'sample_data': df.head(10).to_dict(orient='records')
                }
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return {
                'format': 'csv',
                'error': str(e),
                'summary': 'Failed to process CSV file'
            }
    
    def _process_json(self, file_path):
        """Process a JSON file to extract schema and sample data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # If it's a list of records
            if isinstance(data, list) and len(data) > 0:
                # Extract a sample
                sample = data[:10]
                
                # Extract schema from first item
                first_item = data[0]
                schema = {}
                
                if isinstance(first_item, dict):
                    for key, value in first_item.items():
                        schema[key] = {
                            'type': type(value).__name__,
                            'sample_values': [item.get(key) for item in sample[:5] if item.get(key) is not None]
                        }
                
                return {
                    'format': 'json',
                    'record_count': len(data),
                    'is_array': True,
                    'schema': schema,
                    'sample_data': sample
                }
            else:
                # Treat as a single object
                return {
                    'format': 'json',
                    'is_array': False,
                    'structure': self._get_json_structure(data),
                    'sample_data': data
                }
        except Exception as e:
            print(f"Error processing JSON: {str(e)}")
            return {
                'format': 'json',
                'error': str(e),
                'summary': 'Failed to process JSON file'
            }
    
    def _get_json_structure(self, data, max_depth=3, current_depth=0):
        """Recursively analyze JSON structure"""
        if current_depth >= max_depth:
            return {'type': type(data).__name__}
        
        if isinstance(data, dict):
            structure = {'type': 'object', 'properties': {}}
            for key, value in data.items():
                structure['properties'][key] = self._get_json_structure(
                    value, max_depth, current_depth + 1)
            return structure
        elif isinstance(data, list):
            if len(data) == 0:
                return {'type': 'array', 'items': {'type': 'unknown'}}
            elif len(data) > 0:
                return {
                    'type': 'array',
                    'items': self._get_json_structure(data[0], max_depth, current_depth + 1),
                    'count': len(data)
                }
        else:
            return {
                'type': type(data).__name__,
                'sample': str(data)[:100] if isinstance(data, str) else data
            }
    
    def _process_xml(self, file_path):
        """Process an XML file (simplified)"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract basic structure
            structure = {
                'root_tag': root.tag,
                'attributes': root.attrib,
                'children': []
            }
            
            # Extract first level children
            for child in root[:10]:  # Limit to first 10 children
                child_info = {
                    'tag': child.tag,
                    'attributes': child.attrib,
                    'text': child.text.strip() if child.text else ''
                }
                structure['children'].append(child_info)
            
            return {
                'format': 'xml',
                'structure': structure
            }
        except Exception as e:
            print(f"Error processing XML: {str(e)}")
            return {
                'format': 'xml',
                'error': str(e),
                'summary': 'Failed to process XML file'
            }
    
    def _process_generic_file(self, file_path):
        """Process an unknown file format"""
        try:
            # Get file size
            size = os.path.getsize(file_path)
            
            # Read first few lines
            preview = ''
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    preview = ''.join(file.readline() for _ in range(10))
            except:
                # If we can't read as text, treat as binary
                with open(file_path, 'rb') as file:
                    binary_preview = file.read(100)
                    preview = f"Binary data: {binary_preview}"
            
            return {
                'format': 'unknown',
                'size': size,
                'preview': preview[:500]  # Limit preview size
            }
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return {
                'format': 'unknown',
                'error': str(e),
                'summary': 'Failed to process file'
            }
    
    def assess_quality(self, processed_data, dataset):
        """Assess the quality of the dataset and its metadata"""
        # Simple quality assessment
        quality_score = 0
        issues = []
        
        # Check for basic metadata
        if not dataset.description:
            issues.append("Missing dataset description")
        else:
            quality_score += 20
        
        if not dataset.tags:
            issues.append("Missing dataset tags")
        else:
            quality_score += 10
        
        # Check for source information
        if not dataset.source:
            issues.append("Missing dataset source")
        else:
            quality_score += 10
        
        if not dataset.source_url:
            issues.append("Missing dataset source URL")
        else:
            quality_score += 10
        
        # Check for categorical information
        if not dataset.category:
            issues.append("Missing dataset category")
        else:
            quality_score += 10
        
        if not dataset.data_type:
            issues.append("Missing dataset data type")
        else:
            quality_score += 10
        
        # Check for processed data quality
        if processed_data:
            if 'error' in processed_data:
                issues.append(f"Processing error: {processed_data['error']}")
            else:
                quality_score += 30
        else:
            issues.append("No processed data available")
        
        return {
            'score': quality_score,
            'issues': issues
        }
    
    def add_to_processing_queue(self, dataset_id, priority=1):
        """Add a dataset to the processing queue"""
        # Check if already in queue
        existing = ProcessingQueue.query.filter_by(dataset_id=dataset_id).first()
        if existing:
            # If not already completed or failed, just return it
            if existing.status not in ['completed', 'failed']:
                return existing
            
            # Otherwise update it to re-process
            existing.status = 'queued'
            existing.progress = 0
            existing.started_at = None
            existing.completed_at = None
            existing.error_message = None
            db.session.commit()
            return existing
        
        # Create new queue item
        queue_item = ProcessingQueue(
            dataset_id=dataset_id,
            status='queued',
            priority=priority
        )
        
        db.session.add(queue_item)
        db.session.commit()
        
        # Start processing thread
        threading.Thread(
            target=self.process_queue_item,
            args=(queue_item.id,),
            daemon=True
        ).start()
        
        return queue_item
    
    def process_queue_item(self, queue_id):
        """Process a dataset from the queue"""
        # Get queue item
        queue_item = ProcessingQueue.query.get(queue_id)
        if not queue_item:
            return
        
        # Get dataset
        dataset = Dataset.query.get(queue_item.dataset_id)
        if not dataset:
            return
        
        try:
            # Update status
            queue_item.status = 'processing'
            queue_item.progress = 10
            queue_item.started_at = datetime.utcnow()
            db.session.commit()
            
            # Fetch dataset if URL is provided
            file_info = None
            if dataset.source_url:
                queue_item.progress = 20
                db.session.commit()
                
                file_info = self.fetch_dataset(dataset.source_url)
                
                # Update progress
                queue_item.progress = 40
                db.session.commit()
            
            # Process dataset if we have a file
            processed_data = None
            if file_info:
                processed_data = self.process_dataset(file_info, dataset.format)
                
                # Update dataset with file info
                dataset.file_path = file_info['path']
                dataset.format = file_info['format']
                dataset.size = f"{file_info['size'] / 1024 / 1024:.2f} MB"
                
                if processed_data and 'record_count' in processed_data:
                    dataset.record_count = processed_data['record_count']
                
                # Update progress
                queue_item.progress = 70
                db.session.commit()
            
            # Generate metadata
            if processed_data:
                # Assess quality
                quality_result = self.assess_quality(processed_data, dataset)
                
                # Generate metadata
                metadata = metadata_service.generate_metadata(dataset, processed_data)
                
                # Save metadata quality
                existing_quality = MetadataQuality.query.filter_by(dataset_id=dataset.id).first()
                
                if existing_quality:
                    # Update existing
                    existing_quality.quality_score = quality_result['score']
                    existing_quality.issues_list = quality_result['issues']
                    
                    # Update FAIR scores
                    if metadata.get('fair_scores'):
                        existing_quality.findable_score = metadata['fair_scores'].get('findable', 0)
                        existing_quality.accessible_score = metadata['fair_scores'].get('accessible', 0)
                        existing_quality.interoperable_score = metadata['fair_scores'].get('interoperable', 0)
                        existing_quality.reusable_score = metadata['fair_scores'].get('reusable', 0)
                        existing_quality.fair_compliant = metadata.get('fair_compliant', False)
                    
                    # Update schema.org compliance
                    if metadata.get('schema_org'):
                        existing_quality.schema_org_json = metadata['schema_org']
                        existing_quality.schema_org_compliant = metadata.get('schema_org_compliant', False)
                    
                    # Update recommendations
                    if metadata.get('recommendations'):
                        existing_quality.recommendations_list = metadata['recommendations']
                else:
                    # Create new metadata quality
                    metadata_quality = MetadataQuality(
                        dataset_id=dataset.id,
                        quality_score=quality_result['score'],
                        completeness=metadata.get('completeness', 0),
                        consistency=metadata.get('consistency', 0)
                    )
                    
                    # Set issues
                    metadata_quality.issues_list = quality_result['issues']
                    
                    # Set FAIR scores
                    if metadata.get('fair_scores'):
                        metadata_quality.findable_score = metadata['fair_scores'].get('findable', 0)
                        metadata_quality.accessible_score = metadata['fair_scores'].get('accessible', 0)
                        metadata_quality.interoperable_score = metadata['fair_scores'].get('interoperable', 0)
                        metadata_quality.reusable_score = metadata['fair_scores'].get('reusable', 0)
                        metadata_quality.fair_compliant = metadata.get('fair_compliant', False)
                    
                    # Set schema.org compliance
                    if metadata.get('schema_org'):
                        metadata_quality.schema_org_json = metadata['schema_org']
                        metadata_quality.schema_org_compliant = metadata.get('schema_org_compliant', False)
                    
                    # Set recommendations
                    if metadata.get('recommendations'):
                        metadata_quality.recommendations_list = metadata['recommendations']
                    
                    db.session.add(metadata_quality)
                
                # Update progress
                queue_item.progress = 90
                db.session.commit()
            
            # Update dataset status
            dataset.status = 'completed'
            db.session.commit()
            
            # Complete the queue item
            queue_item.status = 'completed'
            queue_item.progress = 100
            queue_item.completed_at = datetime.utcnow()
            db.session.commit()
            
        except Exception as e:
            # Handle errors
            print(f"Error processing dataset {dataset.id}: {str(e)}")
            
            # Update queue item
            queue_item.status = 'failed'
            queue_item.error_message = str(e)
            queue_item.completed_at = datetime.utcnow()
            db.session.commit()
            
            # Update dataset status
            dataset.status = 'failed'
            db.session.commit()
    
    def search_datasets(self, query, filters=None, sort_by='created_at', limit=10, offset=0):
        """Search datasets using NLP-based semantic search"""
        # Get all datasets initially
        datasets_query = Dataset.query
        
        # Apply filters if provided
        if filters:
            if 'category' in filters and filters['category']:
                datasets_query = datasets_query.filter_by(category=filters['category'])
            if 'data_type' in filters and filters['data_type']:
                datasets_query = datasets_query.filter_by(data_type=filters['data_type'])
        
        # Get datasets
        datasets = datasets_query.all()
        
        # If no query, just return sorted results
        if not query:
            # Apply sort
            if sort_by == 'created_at':
                sorted_datasets = sorted(datasets, 
                                         key=lambda x: x.created_at if x.created_at else datetime.min, 
                                         reverse=True)
            elif sort_by == 'title':
                sorted_datasets = sorted(datasets, key=lambda x: x.title.lower())
            else:
                sorted_datasets = datasets
            
            # Apply limit and offset
            return sorted_datasets[offset:offset+limit]
        
        # Prepare texts for semantic search
        texts = []
        for dataset in datasets:
            # Combine dataset fields for search
            text = f"{dataset.title} {dataset.description or ''} {dataset.tags or ''}"
            texts.append(text)
        
        # Perform semantic search
        search_results = nlp_service.semantic_search(query, texts, [d.id for d in datasets])
        
        # Return datasets sorted by search score
        ranked_datasets = []
        for dataset_id, score in search_results:
            dataset = Dataset.query.get(dataset_id)
            if dataset:
                # Add similarity score attribute
                dataset.similarity = score
                ranked_datasets.append(dataset)
        
        # Apply limit and offset
        return ranked_datasets[offset:offset+limit]


# Global service instance factory
_dataset_service = None

def get_dataset_service(upload_folder):
    """Get or create the dataset service instance"""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService(upload_folder)
    return _dataset_service