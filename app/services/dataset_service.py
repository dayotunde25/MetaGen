import os
import json
import urllib.request
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue

logger = logging.getLogger(__name__)

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
        """Add a dataset to the processing queue (simplified for MongoDB)"""
        try:
            # Check if already in queue
            existing = ProcessingQueue.objects(dataset=str(dataset_id)).first()
            if existing:
                # If not already completed or failed, just return it
                if existing.status not in ['completed', 'failed']:
                    return existing

                # Otherwise update it to re-process
                existing.update(
                    status='pending',
                    progress=0,
                    started_at=None,
                    completed_at=None,
                    error=None
                )
                return existing

            # Create new queue item
            queue_item = ProcessingQueue.create(
                dataset=str(dataset_id),
                status='pending',
                priority=priority
            )

            return queue_item
        except Exception as e:
            print(f"Error adding to processing queue: {e}")
            return None

    # Note: Complex processing methods removed for simplification
    # The manual processing in the routes handles basic quality assessment

    def index_dataset_for_search(self, dataset_id: str) -> bool:
        """
        Index a single dataset for semantic search.

        Args:
            dataset_id: ID of the dataset to index

        Returns:
            True if indexing successful, False otherwise
        """
        try:
            from app.services.semantic_search_service import get_semantic_search_service
            from app.models.dataset import Dataset

            # Get the dataset
            dataset = Dataset.objects(id=dataset_id).first()
            if not dataset:
                logger.warning(f"Dataset {dataset_id} not found for indexing")
                return False

            # Get semantic search service
            semantic_search_service = get_semantic_search_service()

            # Index the dataset
            success = semantic_search_service.index_datasets([dataset])

            if success:
                logger.info(f"Successfully indexed dataset {dataset_id} for semantic search")
            else:
                logger.warning(f"Failed to index dataset {dataset_id} for semantic search")

            return success

        except Exception as e:
            logger.error(f"Error indexing dataset {dataset_id} for search: {e}")
            return False

    def reindex_all_datasets(self) -> bool:
        """
        Reindex all datasets for semantic search.

        Returns:
            True if indexing successful, False otherwise
        """
        try:
            from app.services.semantic_search_service import get_semantic_search_service
            from app.models.dataset import Dataset

            # Get all datasets
            datasets = list(Dataset.objects())

            if not datasets:
                logger.info("No datasets found to index")
                return True

            # Get semantic search service
            semantic_search_service = get_semantic_search_service()

            # Index all datasets
            success = semantic_search_service.index_datasets(datasets)

            if success:
                logger.info(f"Successfully indexed {len(datasets)} datasets for semantic search")
            else:
                logger.warning("Failed to index datasets for semantic search")

            return success

        except Exception as e:
            logger.error(f"Error reindexing all datasets for search: {e}")
            return False

    # Note: Search functionality removed - uses SQLAlchemy which is not compatible with MongoDB


# Global service instance factory
_dataset_service = None

def get_dataset_service(upload_folder):
    """Get or create the dataset service instance"""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService(upload_folder)
    return _dataset_service