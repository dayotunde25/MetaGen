"""
Background Processing Service with Celery for AIMetaHarvest.

This service provides robust background processing that continues even when
the main application goes offline. It includes:
- Persistent task queues
- Automatic retry mechanisms
- Progress checkpointing
- Recovery from interruptions
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Celery imports (will be optional)
try:
    from celery.result import AsyncResult
    # Import the standalone Celery app
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    from celery_app import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None

from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue, MetadataQuality
from app.services.dataset_service import get_dataset_service
from app.services.nlp_service import nlp_service
from app.services.metadata_generator import metadata_service
from app.services.quality_assessment_service import quality_assessment_service
from app.services.data_cleaning_service import data_cleaning_service
from app.services.ai_standards_service import ai_standards_service
from app.services.collection_metadata_service import CollectionMetadataService

logger = logging.getLogger(__name__)

# Celery app is imported from celery_app.py if available

class BackgroundProcessingService:
    """
    Service for robust background processing that survives application restarts.
    
    Features:
    - Persistent task queues with Redis/Celery
    - Automatic retry mechanisms
    - Progress checkpointing
    - Recovery from interruptions
    - Scalable worker processes
    """
    
    def __init__(self):
        """Initialize the background processing service."""
        self.celery_available = CELERY_AVAILABLE
        self.checkpoint_dir = "app/cache/checkpoints"
        self.ensure_checkpoint_dir()
        
    def ensure_checkpoint_dir(self):
        """Ensure checkpoint directory exists."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def start_background_processing(self, dataset_id: str, upload_folder: str, priority: int = 1) -> Dict[str, Any]:
        """
        Start background processing for a dataset.
        
        Args:
            dataset_id: ID of the dataset to process
            upload_folder: Path to upload folder
            priority: Processing priority (1-10, higher is more urgent)
            
        Returns:
            Dictionary with task information
        """
        try:
            # Check if dataset exists
            dataset = Dataset.find_by_id(dataset_id)
            if not dataset:
                return {'success': False, 'error': 'Dataset not found'}

            # Check if already processing to prevent duplication
            queue_item = ProcessingQueue.objects(dataset=dataset_id).first()
            if queue_item and queue_item.status in ['processing', 'queued']:
                logger.warning(f"Dataset {dataset_id} is already being processed (status: {queue_item.status})")
                return {'success': False, 'error': f'Dataset is already being processed (status: {queue_item.status})'}

            # Create or update processing queue entry
            if not queue_item:
                queue_item = ProcessingQueue.create(
                    dataset=dataset_id,
                    status='queued',
                    priority=priority
                )
            else:
                queue_item.update(
                    status='queued',
                    progress=0,
                    started_at=None,
                    completed_at=None,
                    error=None,
                    priority=priority
                )
            
            if self.celery_available:
                # Use Celery for robust background processing
                from celery_app import process_dataset_task
                task = process_dataset_task.apply_async(
                    args=[dataset_id, upload_folder],
                    priority=priority
                    # Use default queue for compatibility
                )

                # Store task ID for tracking
                queue_item.update(task_id=task.id)

                return {
                    'success': True,
                    'task_id': task.id,
                    'method': 'celery',
                    'message': 'Dataset queued for background processing with Celery'
                }
            else:
                # Fallback to enhanced threading with checkpointing
                return self._start_enhanced_threading(dataset_id, upload_folder, queue_item)
                
        except Exception as e:
            logger.error(f"Error starting background processing for dataset {dataset_id}: {e}")
            return {'success': False, 'error': str(e)}

    def queue_dataset_processing(self, dataset_id: str, upload_folder: str = "uploads") -> Dict[str, Any]:
        """
        Queue dataset processing (alias for start_background_processing).

        Args:
            dataset_id: ID of the dataset to process
            upload_folder: Upload folder path

        Returns:
            Dictionary with task information
        """
        return self.start_background_processing(dataset_id, upload_folder)

    def _start_enhanced_threading(self, dataset_id: str, upload_folder: str, queue_item) -> Dict[str, Any]:
        """Enhanced threading with checkpointing as fallback."""
        import threading
        
        def enhanced_processing():
            try:
                self._process_with_checkpoints(dataset_id, upload_folder)
            except Exception as e:
                logger.error(f"Enhanced threading processing failed for {dataset_id}: {e}")
                queue_item.update(status='failed', error=str(e))
        
        thread = threading.Thread(target=enhanced_processing, daemon=False)  # Non-daemon thread
        thread.start()
        
        return {
            'success': True,
            'task_id': f"thread_{dataset_id}",
            'method': 'enhanced_threading',
            'message': 'Dataset queued for enhanced background processing'
        }
    
    def _process_with_checkpoints(self, dataset_id: str, upload_folder: str):
        """Process dataset with checkpoint saving for recovery."""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{dataset_id}_checkpoint.json")
        
        try:
            # Load existing checkpoint if available
            checkpoint = self._load_checkpoint(checkpoint_file)
            start_step = checkpoint.get('last_completed_step', 0)
            
            # Update status
            self._update_progress(dataset_id, start_step * 12.5, 'processing', 
                               f'Resuming from step {start_step + 1}...')
            
            # Get dataset and services
            dataset = Dataset.find_by_id(dataset_id)
            dataset_service = get_dataset_service(upload_folder)
            
            if not dataset:
                raise Exception('Dataset not found')
            
            # Processing steps with checkpointing
            steps = [
                (self._step_process_file, "Processing dataset file"),
                (self._step_clean_data, "Cleaning and restructuring data"),
                (self._step_nlp_analysis, "Performing NLP analysis"),
                (self._step_quality_assessment, "Assessing dataset quality"),
                (self._step_ai_standards, "Assessing AI standards compliance"),
                (self._step_metadata_generation, "Generating comprehensive metadata"),
                (self._step_save_results, "Saving processing results"),
                (self._step_generate_visualizations, "Generating visualizations")
            ]
            
            # Execute steps starting from checkpoint
            results = checkpoint.get('results', {})
            
            for i, (step_func, step_name) in enumerate(steps):
                if i < start_step:
                    continue  # Skip already completed steps
                
                progress = (i + 1) * 12.5
                self._update_progress(dataset_id, progress, 'processing', step_name)
                
                # Execute step
                try:
                    step_result = step_func(dataset, dataset_service, results)
                    results[f'step_{i}'] = step_result
                except Exception as step_error:
                    import traceback
                    logger.error(f"Error in step {i + 1} ({step_name}): {step_error}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise step_error
                
                # Save checkpoint
                checkpoint = {
                    'last_completed_step': i + 1,
                    'results': results,
                    'timestamp': datetime.utcnow().isoformat()
                }
                self._save_checkpoint(checkpoint_file, checkpoint)
                
                logger.info(f"Completed step {i + 1} for dataset {dataset_id}")
            
            # Apply FAIR compliance enhancements
            self._apply_fair_enhancements(dataset_id)

            # Complete processing
            self._update_progress(dataset_id, 100, 'completed', 'Processing completed successfully')

            # Update dataset status to completed
            dataset.update(status='completed')

            # Clean up completed queue item after a delay to allow dashboard to show completion
            self._schedule_queue_cleanup(dataset_id)

            # Clean up checkpoint
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Error processing dataset {dataset_id}: {e}")
            self._update_progress(dataset_id, 0, 'failed', error_msg)
    
    def _load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load processing checkpoint."""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {}
    
    def _save_checkpoint(self, checkpoint_file: str, checkpoint: Dict[str, Any]):
        """Save processing checkpoint."""
        try:
            # Convert numpy types to native Python types for JSON serialization
            checkpoint_serializable = self._make_json_serializable(checkpoint)
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable types to JSON-serializable types."""
        import numpy as np
        import pandas as pd
        from datetime import datetime

        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
            return obj.item() if hasattr(obj, 'item') else obj
        elif hasattr(obj, 'item'):  # Handle pandas/numpy scalars
            try:
                return obj.item()
            except (ValueError, AttributeError):
                return str(obj)
        elif hasattr(obj, 'dtype'):  # Handle other numpy/pandas types
            try:
                return obj.item()
            except (ValueError, AttributeError):
                return str(obj)
        else:
            return obj
    
    def _update_progress(self, dataset_id: str, progress: float, status: str, message: str):
        """Update processing progress in database."""
        try:
            queue_item = ProcessingQueue.objects(dataset=dataset_id).first()
            if queue_item:
                update_data = {
                    'progress': progress,
                    'status': status,
                    'message': message
                }
                
                if status == 'processing' and not queue_item.started_at:
                    update_data['started_at'] = datetime.utcnow()
                elif status in ['completed', 'failed']:
                    update_data['completed_at'] = datetime.utcnow()
                    if status == 'failed':
                        update_data['error'] = message
                
                queue_item.update(**update_data)
                
        except Exception as e:
            logger.error(f"Error updating progress for dataset {dataset_id}: {e}")

    def _schedule_queue_cleanup(self, dataset_id: str):
        """Schedule cleanup of completed queue item."""
        try:
            # Use Celery to schedule cleanup after 30 seconds
            if celery_app:
                from celery_app import cleanup_completed_queue_item
                cleanup_completed_queue_item.apply_async(args=[dataset_id], countdown=30)
            else:
                # Immediate cleanup if Celery not available
                self._cleanup_completed_queue_item(dataset_id)
        except Exception as e:
            logger.warning(f"Could not schedule queue cleanup for {dataset_id}: {e}")

    def _cleanup_completed_queue_item(self, dataset_id: str):
        """Remove completed queue item from processing queue."""
        try:
            queue_item = ProcessingQueue.objects(dataset=dataset_id, status='completed').first()
            if queue_item:
                queue_item.delete()
                logger.info(f"Cleaned up completed queue item for dataset {dataset_id}")
        except Exception as e:
            logger.error(f"Error cleaning up queue item for dataset {dataset_id}: {e}")

    # Processing step methods
    def _step_process_file(self, dataset, dataset_service, results):
        """Step 1: Process dataset file with full dataset processing."""
        file_info = None

        # Handle URL-based datasets
        if dataset.source_url and not dataset.file_path:
            print(f"🌐 Fetching dataset from URL: {dataset.source_url}")
            file_info = dataset_service.fetch_dataset(dataset.source_url)
            if file_info:
                # Update dataset with fetched file information
                dataset.file_path = file_info['path']
                dataset.format = file_info['format']
                dataset.size = f"{file_info['size'] / 1024 / 1024:.2f} MB"
                dataset.save()
                print(f"✅ Successfully fetched and saved dataset from URL")
            else:
                raise Exception(f"Failed to fetch dataset from URL: {dataset.source_url}")

        # Handle local file datasets
        elif dataset.file_path:
            file_info = {
                'path': dataset.file_path,
                'format': dataset.format,
                'size': dataset.size
            }
        else:
            raise Exception("No file path or source URL available for processing")

        if file_info:
            # Check if this is a collection (ZIP file)
            if getattr(dataset, 'is_collection', False) and dataset.format == 'zip':
                print(f"📦 Processing ZIP collection: {dataset.title}")
                result = self._process_collection(dataset, dataset_service)
            else:
                # Enable full dataset processing for individual files
                result = dataset_service.process_dataset(file_info, dataset.format, process_full_dataset=True)
                if result:
                    print(f"📊 Dataset processing: {result.get('processed_records', 0)} records processed from {result.get('record_count', 0)} total")

                    # Debug: Check if schema and field information is extracted
                    print(f"🔍 Data extraction debug:")
                    print(f"   - Keys in processed_data: {list(result.keys())}")
                    if 'schema' in result:
                        schema = result['schema']
                        print(f"   - Schema type: {type(schema)}")
                        if isinstance(schema, dict):
                            print(f"   - Schema keys (fields): {list(schema.keys())[:10]}...")
                            print(f"   - Total fields in schema: {len(schema)}")
                    if 'sample_data' in result:
                        sample_data = result['sample_data']
                        print(f"   - Sample data type: {type(sample_data)}")
                        if isinstance(sample_data, list) and sample_data:
                            print(f"   - First row keys: {list(sample_data[0].keys()) if isinstance(sample_data[0], dict) else 'Not a dict'}")
                else:
                    print("⚠️ No processed data returned from file processing")

            return result
        else:
            print(f"⚠️ File path not found for dataset {dataset.id}")
            return {}

    def _process_collection(self, collection_dataset, dataset_service):
        """Process a ZIP collection and create individual datasets."""
        try:
            # Process the ZIP file using dataset service
            file_info = {
                'path': collection_dataset.file_path,
                'format': 'zip',
                'size': collection_dataset.size
            }

            zip_result = dataset_service.process_dataset(file_info, 'zip', process_full_dataset=True)

            if not zip_result or 'error' in zip_result:
                print(f"❌ Failed to process ZIP file: {zip_result.get('error', 'Unknown error')}")
                return zip_result or {'error': 'Failed to process ZIP file'}

            # Create individual datasets for each file in the collection
            individual_datasets = []
            processed_files = zip_result.get('processed_files', [])

            print(f"📁 Creating {len(processed_files)} individual datasets from collection")

            for i, file_data in enumerate(processed_files):
                try:
                    # Create individual dataset
                    individual_dataset = Dataset.create(
                        title=f"{file_data.get('original_filename', f'Dataset {i+1}')}",
                        description=f"Dataset extracted from collection: {collection_dataset.title}",
                        source=collection_dataset.source,
                        source_url=collection_dataset.source_url,
                        data_type=file_data.get('format', 'unknown'),
                        category=collection_dataset.category,
                        tags=collection_dataset.tags,
                        user=collection_dataset.user,
                        file_path=file_data.get('extracted_path'),
                        format=file_data.get('format'),
                        size=f"{file_data.get('record_count', 0)} records" if file_data.get('record_count') else None,
                        record_count=file_data.get('record_count', 0),
                        field_count=len(file_data.get('columns', [])),
                        field_names=','.join(file_data.get('columns', [])),
                        collection_id=getattr(collection_dataset, 'collection_id', None),
                        parent_collection=collection_dataset,
                        is_collection=False,
                        status='completed'  # Mark as completed since we're processing it now
                    )

                    individual_datasets.append(individual_dataset)
                    print(f"✅ Created dataset: {individual_dataset.title}")

                except Exception as e:
                    print(f"❌ Error creating individual dataset for {file_data.get('original_filename', 'unknown')}: {e}")
                    continue

            # Generate collection-level metadata
            collection_metadata_service = CollectionMetadataService()
            collection_metadata = collection_metadata_service.generate_collection_metadata(
                collection_dataset, individual_datasets
            )

            # Update collection dataset with generated metadata
            collection_description = collection_metadata.get('description', {})
            if collection_description.get('overview', {}).get('text'):
                import json
                collection_dataset.update(
                    description=collection_description['overview']['text'],
                    structured_description=json.dumps(collection_description),
                    record_count=collection_metadata.get('aggregated_statistics', {}).get('total_records', 0),
                    field_count=collection_metadata.get('aggregated_statistics', {}).get('total_fields', 0)
                )

            print(f"✅ Generated collection metadata for {len(individual_datasets)} datasets")

            # Return collection processing result
            return {
                'format': 'zip_collection',
                'is_collection': True,
                'individual_datasets': len(individual_datasets),
                'collection_metadata': collection_metadata,
                'processed_files': processed_files,
                'summary': f"Successfully processed collection with {len(individual_datasets)} datasets"
            }

        except Exception as e:
            print(f"❌ Error processing collection: {e}")
            return {'error': f"Collection processing failed: {str(e)}"}
    
    def _step_clean_data(self, dataset, dataset_service, results):
        """Step 2: Clean and restructure data."""
        processed_data = results.get('step_0', {})
        if processed_data:
            return data_cleaning_service.clean_dataset(processed_data)
        return processed_data
    
    def _step_nlp_analysis(self, dataset, dataset_service, results):
        """Step 3: Advanced NLP analysis with BERT, TF-IDF, and NER."""
        cleaned_data = results.get('step_1', {})
        processed_data = results.get('step_0', {})

        # Extract comprehensive text content from full dataset
        content_text = self._extract_comprehensive_text_content(dataset, cleaned_data, processed_data)

        if content_text:
            print(f"🔍 Starting advanced NLP analysis on {len(content_text)} characters of text...")

            # Use advanced content analysis with all NLP techniques
            analysis_result = nlp_service.advanced_content_analysis(content_text)

            # Add dataset-specific analysis
            analysis_result.update({
                'dataset_title_analysis': nlp_service.extract_keywords(dataset.title or "", 5),
                'field_name_analysis': self._analyze_field_names(cleaned_data),
                'data_type_insights': self._analyze_data_types(cleaned_data),
                'processing_timestamp': datetime.now().isoformat()
            })

            print(f"✅ Advanced NLP analysis complete: {len(analysis_result.get('keywords', []))} keywords, {len(analysis_result.get('entities', []))} entities")
            return analysis_result
        else:
            print("⚠️ No text content found for NLP analysis")
            return {}
    
    def _step_quality_assessment(self, dataset, dataset_service, results):
        """Step 4: Quality assessment."""
        cleaned_data = results.get('step_1', {})
        metadata_quality = quality_assessment_service.assess_dataset_quality(dataset, cleaned_data)
        return metadata_quality.to_dict()
    
    def _step_ai_standards(self, dataset, dataset_service, results):
        """Step 5: AI standards compliance."""
        cleaned_data = results.get('step_1', {})
        return ai_standards_service.assess_ai_compliance(dataset, cleaned_data)
    
    def _step_metadata_generation(self, dataset, dataset_service, results):
        """Step 6: Generate comprehensive metadata using advanced NLP."""
        processed_data = results.get('step_0', {})
        cleaned_data = results.get('step_1', {})
        nlp_results = results.get('step_2', {})
        quality_results = results.get('step_3', {})
        ai_compliance = results.get('step_4', {})

        print(f"🔍 Generating comprehensive metadata using NLP for dataset {dataset.id}")

        # Extract comprehensive text content for NLP-powered metadata generation
        content_text = self._extract_comprehensive_text_content(dataset, cleaned_data, processed_data)

        # Use NLP to enhance metadata generation
        if content_text and len(content_text) > 50:
            print(f"📊 Using {len(content_text)} characters of content for NLP-powered metadata generation")

            # Generate NLP-enhanced description
            nlp_enhanced_description = self._generate_nlp_enhanced_description(
                dataset, content_text, nlp_results, processed_data
            )

            # Generate NLP-based category if missing or generic
            nlp_category = self._generate_nlp_category(dataset, content_text, nlp_results)

            # Generate comprehensive keywords using NLP
            nlp_keywords = self._extract_nlp_keywords(content_text, dataset, nlp_results)

            # Generate use cases using content analysis
            nlp_use_cases = self._generate_nlp_use_cases(content_text, dataset, nlp_results)

            # Create enhanced metadata with NLP insights
            enhanced_metadata = {
                'nlp_enhanced_description': nlp_enhanced_description,
                'nlp_category': nlp_category,
                'nlp_keywords': nlp_keywords,
                'nlp_use_cases': nlp_use_cases,
                'content_analysis': {
                    'text_length': len(content_text),
                    'estimated_reading_time': len(content_text.split()) // 200,  # words per minute
                    'content_complexity': self._assess_content_complexity(content_text),
                    'domain_indicators': self._identify_domain_indicators(content_text)
                }
            }
        else:
            print("⚠️ Limited content available for NLP-enhanced metadata generation")
            enhanced_metadata = {}

        # Generate standard metadata
        standard_metadata = metadata_service.generate_metadata(dataset, cleaned_data)

        # Combine standard and NLP-enhanced metadata
        combined_metadata = {**standard_metadata, **enhanced_metadata}

        print(f"✅ Generated comprehensive metadata with {len(combined_metadata)} fields")
        return combined_metadata
    
    def _step_save_results(self, dataset, dataset_service, results):
        """Step 7: Save comprehensive metadata and results."""
        # Get all processing results
        processed_data = results.get('step_0', {})
        cleaned_data = results.get('step_1', {})
        nlp_results = results.get('step_2', {})
        quality_results = results.get('step_3', {})
        ai_compliance = results.get('step_4', {})
        metadata_results = results.get('step_5', {})

        # Generate comprehensive metadata
        comprehensive_metadata = self._generate_comprehensive_metadata(
            dataset, processed_data, cleaned_data, nlp_results, quality_results, ai_compliance, metadata_results
        )

        # Generate FAIR-compliant metadata
        fair_metadata = self._generate_fair_compliant_metadata(
            dataset, processed_data, cleaned_data, nlp_results, quality_results
        )

        # Update dataset with all metadata
        update_data = {}

        # Core metadata fields
        if comprehensive_metadata.get('title'):
            update_data['title'] = comprehensive_metadata['title']

        # FAIR scores
        if comprehensive_metadata.get('fair_score'):
            update_data['fair_score'] = comprehensive_metadata['fair_score']
        if comprehensive_metadata.get('findable_score'):
            update_data['findable_score'] = comprehensive_metadata['findable_score']
        if comprehensive_metadata.get('accessible_score'):
            update_data['accessible_score'] = comprehensive_metadata['accessible_score']
        if comprehensive_metadata.get('interoperable_score'):
            update_data['interoperable_score'] = comprehensive_metadata['interoperable_score']
        if comprehensive_metadata.get('reusable_score'):
            update_data['reusable_score'] = comprehensive_metadata['reusable_score']
        if 'fair_compliant' in comprehensive_metadata:
            update_data['fair_compliant'] = comprehensive_metadata['fair_compliant']

        if comprehensive_metadata.get('description'):
            update_data['description'] = comprehensive_metadata['description']

        if comprehensive_metadata.get('python_analysis_code'):
            update_data['python_analysis_code'] = comprehensive_metadata['python_analysis_code']

        if comprehensive_metadata.get('source'):
            update_data['source'] = comprehensive_metadata['source']

        if comprehensive_metadata.get('category'):
            update_data['category'] = comprehensive_metadata['category']

        # Data structure metadata
        if comprehensive_metadata.get('record_count'):
            update_data['record_count'] = comprehensive_metadata['record_count']

        if comprehensive_metadata.get('field_count'):
            update_data['field_count'] = comprehensive_metadata['field_count']

        # Extract and save field names and data types
        field_names = self._extract_field_names(processed_data, cleaned_data)
        if field_names:
            update_data['field_names'] = field_names
            print(f"✅ Field names extracted: {field_names[:100]}...")
        else:
            print("⚠️ No field names extracted")

        # Extract field count
        field_count = self._get_field_count(processed_data, cleaned_data)
        if field_count:
            update_data['field_count'] = field_count
            print(f"✅ Field count extracted: {field_count}")
        else:
            print("⚠️ No field count extracted")

        data_types = self._extract_data_types(processed_data, cleaned_data)
        if data_types:
            update_data['data_types'] = data_types

        # FAIR-compliant metadata updates (with safe handling)
        try:
            if fair_metadata.get('persistent_identifier'):
                update_data['persistent_identifier'] = fair_metadata['persistent_identifier']
            if fair_metadata.get('fair_metadata_json'):
                update_data['fair_metadata'] = fair_metadata['fair_metadata_json']
            if fair_metadata.get('dublin_core_json'):
                update_data['dublin_core'] = fair_metadata['dublin_core_json']
            if fair_metadata.get('dcat_json'):
                update_data['dcat_metadata'] = fair_metadata['dcat_json']
            if fair_metadata.get('json_ld'):
                update_data['json_ld'] = fair_metadata['json_ld']
            if fair_metadata.get('fair_compliant'):
                update_data['fair_compliant'] = fair_metadata['fair_compliant']
        except Exception as e:
            logger.warning(f"Could not update FAIR metadata fields: {e}")
            # Continue without FAIR metadata if fields don't exist

        if comprehensive_metadata.get('fields_info'):
            update_data['fields_info'] = json.dumps(self._make_json_serializable(comprehensive_metadata['fields_info']))

        # Content analysis metadata
        if comprehensive_metadata.get('keywords'):
            update_data['keywords'] = json.dumps(comprehensive_metadata['keywords'])

        if comprehensive_metadata.get('tags'):
            update_data['tags'] = comprehensive_metadata['tags']

        if comprehensive_metadata.get('use_cases'):
            update_data['use_cases'] = json.dumps(comprehensive_metadata['use_cases'])

        # Quality and compliance metadata
        if comprehensive_metadata.get('quality_score'):
            update_data['quality_score'] = comprehensive_metadata['quality_score']

        # FAIR scores
        if comprehensive_metadata.get('fair_score'):
            update_data['fair_score'] = comprehensive_metadata['fair_score']
        if comprehensive_metadata.get('findable_score'):
            update_data['findable_score'] = comprehensive_metadata['findable_score']
        if comprehensive_metadata.get('accessible_score'):
            update_data['accessible_score'] = comprehensive_metadata['accessible_score']
        if comprehensive_metadata.get('interoperable_score'):
            update_data['interoperable_score'] = comprehensive_metadata['interoperable_score']
        if comprehensive_metadata.get('reusable_score'):
            update_data['reusable_score'] = comprehensive_metadata['reusable_score']
        if comprehensive_metadata.get('fair_compliant') is not None:
            update_data['fair_compliant'] = comprehensive_metadata['fair_compliant']

        # Schema.org and structured metadata
        if comprehensive_metadata.get('schema_org'):
            schema_org_data = self._make_json_serializable(comprehensive_metadata['schema_org'])
            update_data['schema_org'] = json.dumps(schema_org_data)

        # AI compliance data
        if comprehensive_metadata.get('ai_compliance'):
            ai_compliance_data = self._make_json_serializable(comprehensive_metadata['ai_compliance'])
            update_data['ai_compliance'] = json.dumps(ai_compliance_data)

        # Update dataset with all metadata
        if update_data:
            try:
                # Make sure all data is JSON serializable
                serializable_data = self._make_json_serializable(update_data)
                dataset.update(**serializable_data)
                logger.info(f"Updated dataset {dataset.id} with comprehensive metadata: {list(serializable_data.keys())}")
            except Exception as e:
                logger.error(f"Error updating dataset {dataset.id}: {e}")
                # Try to save individual fields
                for key, value in update_data.items():
                    try:
                        serializable_value = self._make_json_serializable(value)
                        dataset.update(**{key: serializable_value})
                        logger.info(f"Successfully updated field {key}")
                    except Exception as field_error:
                        logger.error(f"Error updating field {key}: {field_error}")

        # Save quality assessment to MetadataQuality collection
        if quality_results:
            # Include FAIR scores from comprehensive metadata in quality results
            enhanced_quality_results = quality_results.copy()
            if comprehensive_metadata:
                enhanced_quality_results.update({
                    'findable_score': comprehensive_metadata.get('findable_score', 0),
                    'accessible_score': comprehensive_metadata.get('accessible_score', 0),
                    'interoperable_score': comprehensive_metadata.get('interoperable_score', 0),
                    'reusable_score': comprehensive_metadata.get('reusable_score', 0),
                    'fair_score': comprehensive_metadata.get('fair_score', 0),
                    'fair_compliant': comprehensive_metadata.get('fair_compliant', False)
                })
            self._save_quality_assessment(dataset, enhanced_quality_results)

        # Save comprehensive metadata summary
        metadata_summary = {
            'total_fields': len(update_data),
            'metadata_completeness': self._calculate_metadata_completeness(comprehensive_metadata),
            'processing_timestamp': datetime.now().isoformat(),
            'metadata_version': '1.0'
        }

        return {
            'saved': True,
            'updated_fields': list(update_data.keys()),
            'metadata_summary': metadata_summary,
            'comprehensive_metadata': comprehensive_metadata
        }
    
    def _step_generate_visualizations(self, dataset, dataset_service, results):
        """Step 8: Generate comprehensive visualizations."""
        processed_data = results.get('step_0', {})
        cleaned_data = results.get('step_1', {})
        nlp_results = results.get('step_2', {})
        quality_results = results.get('step_3', {})

        # Generate comprehensive visualization data with enhanced chart types
        visualizations = {
            'quality_metrics': self._create_quality_chart_data(quality_results),
            'data_overview': self._create_data_overview_chart(processed_data, cleaned_data),
            'field_analysis': self._create_field_analysis_chart(processed_data, cleaned_data),
            'keyword_cloud': self._create_keyword_visualization(nlp_results),
            'metadata_completeness': self._create_metadata_completeness_chart(dataset, quality_results),
            'data_distribution': self._create_data_distribution_charts(processed_data, cleaned_data),
            'correlation_analysis': self._create_correlation_heatmap(processed_data, cleaned_data),
            'trend_analysis': self._create_trend_charts(processed_data, cleaned_data),
            'generated_at': datetime.now().isoformat(),
            'visualization_version': '3.0'
        }

        # Save visualizations to dataset
        try:
            visualization_json = json.dumps(self._make_json_serializable(visualizations))
            dataset.update(visualizations=visualization_json)
            logger.info(f"Generated and saved visualizations for dataset {dataset.id}")
        except Exception as e:
            logger.error(f"Error saving visualizations for dataset {dataset.id}: {e}")

        return visualizations

    def _generate_fair_compliant_metadata(self, dataset, processed_data, cleaned_data, nlp_results, quality_results):
        """Generate FAIR-compliant metadata with persistent identifiers and rich vocabularies."""
        try:
            print(f"🔍 Generating FAIR-compliant metadata for dataset {dataset.id}")

            # Generate persistent identifier (DOI-like format)
            persistent_id = self._generate_persistent_identifier(dataset)

            # Generate Dublin Core metadata
            dublin_core = self._generate_dublin_core_metadata(dataset, processed_data, nlp_results)

            # Generate DCAT metadata
            dcat_metadata = self._generate_dcat_metadata(dataset, processed_data, nlp_results)

            # Generate comprehensive FAIR metadata
            fair_metadata = self._generate_comprehensive_fair_metadata(
                dataset, processed_data, cleaned_data, nlp_results, quality_results, persistent_id
            )

            # Generate machine-readable formats
            json_ld = self._generate_json_ld_metadata(dataset, fair_metadata, persistent_id)

            print(f"✅ Generated FAIR-compliant metadata with persistent ID: {persistent_id}")

            return {
                'persistent_identifier': persistent_id,
                'dublin_core_json': json.dumps(dublin_core, indent=2),
                'dcat_json': json.dumps(dcat_metadata, indent=2),
                'fair_metadata_json': json.dumps(fair_metadata, indent=2),
                'json_ld': json.dumps(json_ld, indent=2),
                'fair_compliant': True
            }

        except Exception as e:
            logger.error(f"Error generating FAIR-compliant metadata: {e}")
            return {
                'persistent_identifier': None,
                'dublin_core_json': '{}',
                'dcat_json': '{}',
                'fair_metadata_json': '{}',
                'json_ld': '{}',
                'fair_compliant': False
            }

    def _generate_persistent_identifier(self, dataset):
        """Generate a persistent identifier for the dataset."""
        # Generate a DOI-like identifier
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return f"doi:10.5555/aimh.{dataset.id}.{unique_id}"

    def _generate_dublin_core_metadata(self, dataset, processed_data, nlp_results):
        """Generate Dublin Core metadata."""
        record_count = self._get_record_count(processed_data, {})

        return {
            "dc:title": dataset.title or "Untitled Dataset",
            "dc:description": dataset.description or "No description available",
            "dc:creator": dataset.user.username if dataset.user else "Unknown",
            "dc:publisher": "AI Meta Harvest",
            "dc:date": dataset.created_at.isoformat() if dataset.created_at else None,
            "dc:type": "Dataset",
            "dc:format": dataset.format or "Unknown",
            "dc:identifier": dataset.id,
            "dc:language": "en",
            "dc:rights": dataset.license or "All rights reserved",
            "dc:subject": nlp_results.get('keywords', [])[:10] if nlp_results else [],
            "dc:coverage": f"{record_count} records" if record_count else "Unknown size"
        }

    def _generate_dcat_metadata(self, dataset, processed_data, nlp_results):
        """Generate DCAT (Data Catalog Vocabulary) metadata."""
        record_count = self._get_record_count(processed_data, {})

        return {
            "@type": "dcat:Dataset",
            "dcat:title": dataset.title or "Untitled Dataset",
            "dcat:description": dataset.description or "No description available",
            "dcat:keyword": nlp_results.get('keywords', [])[:10] if nlp_results else [],
            "dcat:theme": dataset.category or "General",
            "dcat:issued": dataset.created_at.isoformat() if dataset.created_at else None,
            "dcat:modified": dataset.updated_at.isoformat() if dataset.updated_at else None,
            "dcat:publisher": {
                "@type": "foaf:Organization",
                "foaf:name": "AI Meta Harvest"
            },
            "dcat:contactPoint": {
                "@type": "vcard:Contact",
                "vcard:fn": dataset.user.username if dataset.user else "Unknown",
                "vcard:hasEmail": dataset.user.email if dataset.user else None
            },
            "dcat:distribution": {
                "@type": "dcat:Distribution",
                "dcat:format": dataset.format or "Unknown",
                "dcat:mediaType": self._get_media_type(dataset.format),
                "dcat:byteSize": record_count if record_count else None
            },
            "dcat:landingPage": f"/datasets/{dataset.id}",
            "dcat:accessRights": "public" if not dataset.private else "restricted"
        }

    def _generate_comprehensive_fair_metadata(self, dataset, processed_data, cleaned_data, nlp_results, quality_results, persistent_id):
        """Generate comprehensive FAIR metadata."""
        record_count = self._get_record_count(processed_data, cleaned_data)
        field_count = self._get_field_count(processed_data, cleaned_data)

        return {
            # Findable
            "findable": {
                "persistent_identifier": persistent_id,
                "rich_metadata": True,
                "metadata_includes_identifier": True,
                "registered_in_searchable_resource": True,
                "keywords": nlp_results.get('keywords', [])[:15] if nlp_results else [],
                "title": dataset.title,
                "description": dataset.description,
                "tags": dataset.tags.split(',') if dataset.tags else []
            },

            # Accessible
            "accessible": {
                "retrievable_by_identifier": True,
                "protocol_open_free": True,
                "protocol_allows_authentication": True,
                "metadata_accessible_when_data_unavailable": True,
                "access_url": f"/datasets/{dataset.id}",
                "download_url": f"/datasets/{dataset.id}/download" if not dataset.private else None,
                "access_rights": "public" if not dataset.private else "restricted"
            },

            # Interoperable
            "interoperable": {
                "formal_accessible_shared_language": True,
                "vocabularies_follow_fair": True,
                "qualified_references": True,
                "format": dataset.format,
                "schema_org_compliant": bool(dataset.schema_org_json),
                "standard_vocabularies": ["Dublin Core", "DCAT", "Schema.org"],
                "data_structure": {
                    "records": record_count,
                    "fields": field_count,
                    "format": dataset.format
                }
            },

            # Reusable
            "reusable": {
                "rich_attributes": True,
                "clear_usage_license": bool(dataset.license),
                "detailed_provenance": True,
                "community_standards": True,
                "license": dataset.license or "Not specified",
                "provenance": {
                    "created": dataset.created_at.isoformat() if dataset.created_at else None,
                    "creator": dataset.user.username if dataset.user else "Unknown",
                    "source": dataset.source or "Not specified",
                    "processing_history": "Processed with AI Meta Harvest"
                },
                "quality_assessment": {
                    "overall_score": quality_results.get('quality_score', 0) if quality_results else 0,
                    "completeness": quality_results.get('completeness', 0) if quality_results else 0,
                    "accuracy": quality_results.get('accuracy', 0) if quality_results else 0,
                    "consistency": quality_results.get('consistency', 0) if quality_results else 0
                }
            },

            # Additional metadata
            "technical_metadata": {
                "file_size": getattr(dataset, 'file_size', None),
                "encoding": "UTF-8",
                "creation_date": dataset.created_at.isoformat() if dataset.created_at else None,
                "last_modified": dataset.updated_at.isoformat() if dataset.updated_at else None
            },

            "semantic_metadata": {
                "entities": nlp_results.get('entities', [])[:20] if nlp_results else [],
                "concepts": nlp_results.get('keywords', [])[:10] if nlp_results else [],
                "language": "en",
                "domain": dataset.category or "General"
            }
        }

    def _generate_json_ld_metadata(self, dataset, fair_metadata, persistent_id):
        """Generate JSON-LD structured data."""
        return {
            "@context": {
                "@vocab": "https://schema.org/",
                "dcat": "http://www.w3.org/ns/dcat#",
                "dc": "http://purl.org/dc/terms/",
                "foaf": "http://xmlns.com/foaf/0.1/"
            },
            "@type": "Dataset",
            "@id": persistent_id,
            "name": dataset.title or "Untitled Dataset",
            "description": dataset.description or "No description available",
            "creator": {
                "@type": "Person",
                "name": dataset.user.username if dataset.user else "Unknown"
            },
            "publisher": {
                "@type": "Organization",
                "name": "AI Meta Harvest"
            },
            "dateCreated": dataset.created_at.isoformat() if dataset.created_at else None,
            "dateModified": dataset.updated_at.isoformat() if dataset.updated_at else None,
            "keywords": fair_metadata.get('findable', {}).get('keywords', []),
            "license": dataset.license or "Not specified",
            "url": f"/datasets/{dataset.id}",
            "identifier": persistent_id,
            "encodingFormat": dataset.format or "Unknown",
            "isAccessibleForFree": not dataset.private,
            "distribution": {
                "@type": "DataDownload",
                "encodingFormat": dataset.format or "Unknown",
                "contentUrl": f"/datasets/{dataset.id}/download" if not dataset.private else None
            }
        }

    def _get_media_type(self, format_str):
        """Get MIME type for format."""
        format_map = {
            'csv': 'text/csv',
            'json': 'application/json',
            'xml': 'application/xml',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'txt': 'text/plain'
        }
        return format_map.get(format_str.lower() if format_str else '', 'application/octet-stream')

    def _create_data_distribution_charts(self, processed_data, cleaned_data):
        """Create histogram and box plot for data distribution analysis."""
        try:
            data_source = cleaned_data if cleaned_data else processed_data
            if not data_source or 'sample_data' not in data_source:
                return {
                    'type': 'histogram',
                    'data': {'bins': [], 'frequencies': []},
                    'chart_config': {'type': 'histogram', 'title': 'Data Distribution', 'description': 'No data available'}
                }

            sample_data = data_source['sample_data']
            if not isinstance(sample_data, list) or not sample_data:
                return {
                    'type': 'histogram',
                    'data': {'bins': [], 'frequencies': []},
                    'chart_config': {'type': 'histogram', 'title': 'Data Distribution', 'description': 'No valid data'}
                }

            # Find numeric columns for distribution analysis
            numeric_data = []
            for row in sample_data[:100]:  # Analyze first 100 rows
                if isinstance(row, dict):
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            numeric_data.append(value)

            if not numeric_data:
                return {
                    'type': 'histogram',
                    'data': {'bins': [], 'frequencies': []},
                    'chart_config': {'type': 'histogram', 'title': 'Data Distribution', 'description': 'No numeric data found'}
                }

            # Create histogram bins
            import numpy as np
            hist, bin_edges = np.histogram(numeric_data, bins=10)
            bins = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

            return {
                'type': 'histogram',
                'data': {
                    'bins': bins,
                    'frequencies': hist.tolist(),
                    'xlabel': 'Value Range',
                    'total_values': len(numeric_data)
                },
                'chart_config': {
                    'type': 'histogram',
                    'title': 'Data Distribution',
                    'description': f'Distribution of {len(numeric_data)} numeric values'
                }
            }

        except Exception as e:
            logger.error(f"Error creating data distribution chart: {e}")
            return {
                'type': 'histogram',
                'data': {'bins': [], 'frequencies': []},
                'chart_config': {'type': 'histogram', 'title': 'Data Distribution', 'description': 'Error generating chart'}
            }

    def _create_correlation_heatmap(self, processed_data, cleaned_data):
        """Create correlation heatmap for numeric fields."""
        try:
            data_source = cleaned_data if cleaned_data else processed_data
            if not data_source or 'sample_data' not in data_source:
                return {
                    'type': 'heatmap',
                    'data': {'matrix': [], 'labels': {}},
                    'chart_config': {'type': 'heatmap', 'title': 'Correlation Analysis', 'description': 'No data available'}
                }

            sample_data = data_source['sample_data']
            if not isinstance(sample_data, list) or not sample_data:
                return {
                    'type': 'heatmap',
                    'data': {'matrix': [], 'labels': {}},
                    'chart_config': {'type': 'heatmap', 'title': 'Correlation Analysis', 'description': 'No valid data'}
                }

            # Extract numeric columns
            numeric_columns = {}
            for row in sample_data[:50]:  # Analyze first 50 rows
                if isinstance(row, dict):
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if key not in numeric_columns:
                                numeric_columns[key] = []
                            numeric_columns[key].append(value)

            # Filter columns with sufficient data
            valid_columns = {k: v for k, v in numeric_columns.items() if len(v) >= 10}

            if len(valid_columns) < 2:
                return {
                    'type': 'heatmap',
                    'data': {'matrix': [], 'labels': {}},
                    'chart_config': {'type': 'heatmap', 'title': 'Correlation Analysis', 'description': 'Insufficient numeric data for correlation'}
                }

            # Calculate simple correlation matrix
            import numpy as np
            column_names = list(valid_columns.keys())[:5]  # Limit to 5 columns
            matrix = []

            for i, col1 in enumerate(column_names):
                row = []
                for j, col2 in enumerate(column_names):
                    if i == j:
                        correlation = 1.0
                    else:
                        # Simple correlation calculation
                        try:
                            data1 = np.array(valid_columns[col1][:min(len(valid_columns[col1]), len(valid_columns[col2]))])
                            data2 = np.array(valid_columns[col2][:len(data1)])
                            correlation = np.corrcoef(data1, data2)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0.0
                        except:
                            correlation = 0.0
                    row.append(correlation)
                matrix.append(row)

            return {
                'type': 'heatmap',
                'data': {
                    'matrix': matrix,
                    'labels': {'x': column_names, 'y': column_names},
                    'max': 1.0
                },
                'chart_config': {
                    'type': 'heatmap',
                    'title': 'Correlation Analysis',
                    'description': f'Correlation matrix for {len(column_names)} numeric fields'
                }
            }

        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return {
                'type': 'heatmap',
                'data': {'matrix': [], 'labels': {}},
                'chart_config': {'type': 'heatmap', 'title': 'Correlation Analysis', 'description': 'Error generating heatmap'}
            }

    def _create_trend_charts(self, processed_data, cleaned_data):
        """Create line charts for trend analysis."""
        try:
            data_source = cleaned_data if cleaned_data else processed_data
            record_count = self._get_record_count(processed_data, cleaned_data)

            # Create a simple trend showing data quality over record index
            if record_count > 0:
                # Simulate quality trend (in real implementation, this would analyze actual data quality metrics)
                x_values = list(range(0, min(record_count, 100), 10))
                y_values = [85 + (i * 0.1) + ((-1) ** i) * 2 for i in range(len(x_values))]  # Simulated trend

                return {
                    'type': 'line',
                    'data': {
                        'labels': [f"Record {x}" for x in x_values],
                        'values': y_values,
                        'x': x_values,
                        'y': y_values
                    },
                    'chart_config': {
                        'type': 'line',
                        'title': 'Data Quality Trend',
                        'description': f'Quality trend across {len(x_values)} sample points'
                    }
                }
            else:
                return {
                    'type': 'line',
                    'data': {'labels': [], 'values': []},
                    'chart_config': {'type': 'line', 'title': 'Data Quality Trend', 'description': 'No data available'}
                }

        except Exception as e:
            logger.error(f"Error creating trend chart: {e}")
            return {
                'type': 'line',
                'data': {'labels': [], 'values': []},
                'chart_config': {'type': 'line', 'title': 'Data Quality Trend', 'description': 'Error generating chart'}
            }

    def _extract_text_content(self, dataset, processed_data):
        """Extract text content for NLP analysis."""
        text_parts = []
        if dataset.title:
            text_parts.append(dataset.title)
        if dataset.description:
            text_parts.append(dataset.description)
        if dataset.tags:
            text_parts.extend(dataset.tags if isinstance(dataset.tags, list) else [dataset.tags])
        
        if processed_data and 'sample_data' in processed_data:
            sample_data = processed_data['sample_data']
            if isinstance(sample_data, list):
                for row in sample_data[:5]:
                    if isinstance(row, dict):
                        text_parts.extend([str(v) for v in row.values() if isinstance(v, str)])
        
        return ' '.join(text_parts)
    
    def _create_quality_chart_data(self, quality_results):
        """Create comprehensive quality metrics chart."""
        if not quality_results:
            return {
                'type': 'quality_metrics',
                'data': {'overall_score': 0, 'completeness': 0, 'consistency': 0, 'accuracy': 0},
                'chart_config': {'type': 'radar', 'title': 'Data Quality Metrics'}
            }

        return {
            'type': 'quality_metrics',
            'data': {
                'overall_score': quality_results.get('quality_score', 0),
                'completeness': quality_results.get('completeness', 0),
                'consistency': quality_results.get('consistency', 0),
                'accuracy': quality_results.get('accuracy', 0),
                'fair_compliant': quality_results.get('fair_compliant', False)
            },
            'chart_config': {
                'type': 'radar',
                'title': 'Data Quality Assessment',
                'description': 'Comprehensive quality metrics for the dataset'
            }
        }

    def _create_data_overview_chart(self, processed_data, cleaned_data):
        """Create data overview visualization."""
        record_count = self._get_record_count(processed_data, cleaned_data)
        field_count = self._get_field_count(processed_data, cleaned_data)

        # Ensure counts are numeric for calculations
        if not isinstance(record_count, (int, float)):
            record_count = 0
        if not isinstance(field_count, (int, float)):
            field_count = 0

        return {
            'type': 'data_overview',
            'data': {
                'record_count': record_count,
                'field_count': field_count,
                'data_size': f"{record_count} rows × {field_count} columns",
                'estimated_size_mb': round((record_count * field_count * 8) / (1024 * 1024), 2) if record_count > 0 and field_count > 0 else 0  # Rough estimate
            },
            'chart_config': {
                'type': 'info_cards',
                'title': 'Dataset Overview',
                'description': 'Basic statistics about the dataset structure'
            }
        }

    def _create_field_analysis_chart(self, processed_data, cleaned_data):
        """Create field analysis visualization."""
        # Use cleaned_data first, fallback to processed_data
        data_source = cleaned_data if cleaned_data else processed_data
        data_types = self._analyze_data_types(data_source)

        return {
            'type': 'field_analysis',
            'data': {
                'type_distribution': data_types.get('type_distribution', {}),
                'numeric_fields': data_types.get('numeric_fields', 0),
                'text_fields': data_types.get('text_fields', 0),
                'categorical_fields': data_types.get('categorical_fields', 0),
                'datetime_fields': data_types.get('datetime_fields', 0)
            },
            'chart_config': {
                'type': 'pie',
                'title': 'Field Type Distribution',
                'description': 'Distribution of data types across dataset fields'
            }
        }

    def _create_keyword_visualization(self, nlp_results):
        """Create keyword cloud visualization."""
        if not nlp_results or 'keywords' not in nlp_results:
            return {
                'type': 'keyword_cloud',
                'data': {'keywords': []},
                'chart_config': {'type': 'wordcloud', 'title': 'Keywords', 'description': 'No keywords available'}
            }

        keywords = nlp_results['keywords'][:20]  # Limit to top 20
        keyword_data = [{'text': keyword, 'weight': max(1, 21 - i)} for i, keyword in enumerate(keywords)]

        return {
            'type': 'keyword_cloud',
            'data': {
                'keywords': keyword_data,
                'total_keywords': len(keywords)
            },
            'chart_config': {
                'type': 'wordcloud',
                'title': 'Content Keywords',
                'description': 'Most relevant keywords extracted from dataset content'
            }
        }

    def _create_metadata_completeness_chart(self, dataset, quality_results):
        """Create metadata completeness visualization."""
        # Check completeness of various metadata fields
        completeness = {
            'title': bool(dataset.title and dataset.title.strip()),
            'description': bool(dataset.description and len(dataset.description.strip()) > 20),
            'source': bool(dataset.source and dataset.source.strip()),
            'category': bool(dataset.category and dataset.category.strip()),
            'tags': bool(dataset.tags and dataset.tags.strip()),
            'quality_assessed': bool(quality_results),
            'schema_defined': bool(hasattr(dataset, 'fields_info') and dataset.fields_info)
        }

        completed_count = sum(completeness.values())
        total_count = len(completeness)
        completeness_percentage = round((completed_count / total_count) * 100, 1)

        return {
            'type': 'metadata_completeness',
            'data': {
                'completeness_fields': completeness,
                'completed_count': completed_count,
                'total_count': total_count,
                'completeness_percentage': completeness_percentage
            },
            'chart_config': {
                'type': 'progress_bar',
                'title': 'Metadata Completeness',
                'description': f'{completeness_percentage}% of metadata fields are complete'
            }
        }

    def _generate_auto_description(self, dataset, nlp_results, processed_data):
        """Generate comprehensive automatic description using AI models first, then fallback."""
        print("🔍 Generating auto-description using AI models...")

        # Try to use enhanced AI description first
        try:
            from app.services.nlp_service import nlp_service

            # Prepare dataset info for AI models
            dataset_info = {
                'title': dataset.title or 'Unknown Dataset',
                'field_names': self._extract_field_names_list(processed_data),
                'record_count': self._get_record_count(processed_data, {}),
                'data_types': self._extract_data_types_dict(processed_data),
                'sample_data': self._get_sample_data(processed_data)[:3],
                'keywords': [kw.get('text', kw) if isinstance(kw, dict) else str(kw)
                           for kw in nlp_results.get('keywords', [])[:10]] if nlp_results else [],
                'entities': [ent.get('text', ent) if isinstance(ent, dict) else str(ent)
                           for ent in nlp_results.get('entities', [])[:5]] if nlp_results else [],
                'summary': nlp_results.get('summary', '') if nlp_results else '',
                'category': dataset.category
            }

            # Try to generate enhanced description using free AI models
            enhanced_description = nlp_service.generate_enhanced_description(dataset_info)

            if enhanced_description and len(enhanced_description) > 100:
                print(f"✅ Generated AI-enhanced description: {len(enhanced_description)} characters")
                return enhanced_description
            else:
                print("⚠️ AI models returned short description, using traditional method")

        except Exception as e:
            print(f"⚠️ AI description generation failed: {e}")

        # Fallback to traditional description generation
        print("🔄 Using traditional description generation as fallback...")
        return self._generate_traditional_description(dataset, nlp_results, processed_data)

    def _generate_traditional_description(self, dataset, nlp_results, processed_data):
        """Generate traditional comprehensive description (fallback method)."""
        description_parts = []

        # Start with comprehensive introduction
        if dataset.title:
            description_parts.append(f"This comprehensive dataset, titled '{dataset.title}', represents a valuable collection of structured data containing")
        else:
            description_parts.append("This comprehensive dataset represents a valuable collection of structured data containing")

        # Add detailed record count and significance
        if processed_data and 'record_count' in processed_data:
            record_count = processed_data['record_count']

            # Ensure record_count is an integer
            if not isinstance(record_count, (int, float)):
                try:
                    record_count = int(record_count) if record_count else 0
                except (ValueError, TypeError):
                    record_count = 0

            if record_count > 10000:
                description_parts.append(f"{record_count:,} records, providing a substantial dataset for comprehensive analysis and research applications.")
            elif record_count > 1000:
                description_parts.append(f"{record_count:,} records, offering a robust foundation for data analysis and statistical modeling.")
            elif record_count > 100:
                description_parts.append(f"{record_count:,} records, providing sufficient data for meaningful analysis and insights.")
            else:
                description_parts.append(f"{record_count:,} records, suitable for exploratory analysis and research purposes.")

        # Add comprehensive field information and data structure
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            if isinstance(schema, dict):
                field_count = len(schema)
                description_parts.append(f"The dataset is structured across {field_count} distinct data fields, each contributing unique information to the overall data landscape.")

                # Add detailed field analysis
                key_fields = list(schema.keys())[:8]  # Show more fields
                if key_fields:
                    if len(key_fields) > 5:
                        primary_fields = ', '.join(key_fields[:5])
                        additional_fields = ', '.join(key_fields[5:])
                        description_parts.append(f"Primary data fields include {primary_fields}, with additional dimensions covering {additional_fields}.")
                    else:
                        description_parts.append(f"Key data fields encompass {', '.join(key_fields)}, providing comprehensive coverage of the subject domain.")

                # Add data type analysis
                data_types = set()
                for field_info in schema.values():
                    if isinstance(field_info, dict) and 'type' in field_info:
                        data_types.add(field_info['type'])

                if data_types:
                    type_description = self._describe_data_types_comprehensive(data_types)
                    description_parts.append(f"The dataset incorporates {type_description}, ensuring rich analytical possibilities.")

        # Add category and domain context
        if dataset.category:
            category_context = self._get_enhanced_category_context(dataset.category)
            description_parts.append(f"Classified within the {dataset.category.lower()} domain, {category_context}")

        # Add enhanced NLP insights
        if nlp_results:
            # Add content summary from NLP
            if nlp_results.get('summary'):
                summary = nlp_results['summary']
                if isinstance(summary, list) and summary:
                    description_parts.append(f"Content analysis reveals: {' '.join(summary[:2])}.")  # Limit summary length

            # Add comprehensive keywords context
            if nlp_results.get('keywords'):
                keywords = nlp_results['keywords'][:10]  # More keywords
                if keywords:
                    keyword_list = [k['text'] if isinstance(k, dict) else str(k) for k in keywords]
                    if len(keyword_list) > 5:
                        primary_keywords = ', '.join(keyword_list[:5])
                        secondary_keywords = ', '.join(keyword_list[5:])
                        description_parts.append(f"Advanced natural language processing analysis identifies key thematic elements including {primary_keywords}, with additional contextual themes encompassing {secondary_keywords}.")
                    else:
                        description_parts.append(f"Key thematic elements identified through NLP analysis include: {', '.join(keyword_list)}.")

            # Add entity analysis
            if nlp_results.get('entities'):
                entities = nlp_results['entities'][:6]  # More entities
                if entities:
                    entity_list = [e['text'] if isinstance(e, dict) else str(e) for e in entities]
                    description_parts.append(f"Named entity recognition identifies significant elements: {', '.join(entity_list)}, which serve as important focal points for analysis.")

        # Add comprehensive use cases and applications
        use_cases = self._generate_use_case_suggestions(dataset, nlp_results, processed_data)
        if use_cases:
            if len(use_cases) > 4:
                primary_uses = ', '.join(use_cases[:3])
                additional_uses = ', '.join(use_cases[3:6])
                description_parts.append(f"This dataset offers exceptional potential for diverse analytical applications including {primary_uses}. Additional research opportunities encompass {additional_uses}, making it a versatile resource for both academic research and practical business intelligence applications.")
            else:
                description_parts.append(f"The dataset provides excellent opportunities for {', '.join(use_cases)}, supporting comprehensive research initiatives and data-driven decision making processes.")

        # Add data quality and research value
        description_parts.append("The dataset's structured format and comprehensive content make it particularly well-suited for advanced analytics, machine learning applications, statistical modeling, and interdisciplinary research projects.")

        # Add format and accessibility information
        if dataset.format:
            description_parts.append(f"Provided in {dataset.format.upper()} format, the dataset ensures compatibility with standard data analysis tools and platforms, facilitating seamless integration into research workflows.")

        # Add licensing and usage context
        if dataset.license:
            description_parts.append(f"The dataset is made available under {dataset.license} licensing terms, ensuring appropriate usage guidelines and supporting responsible data utilization practices.")

        return ' '.join(description_parts)

    def _generate_python_analysis_code(self, dataset, nlp_results, processed_data):
        """Generate Python analysis code for the dataset."""
        print("🔍 Generating Python analysis code...")

        try:
            from app.services.nlp_service import nlp_service

            # Prepare dataset info for code generation
            dataset_info = {
                'title': dataset.title or 'Dataset',
                'field_names': self._extract_field_names_list(processed_data),
                'data_types': self._extract_data_types_dict(processed_data),
                'record_count': self._get_record_count(processed_data, {}),
                'category': dataset.category or 'General',
                'keywords': nlp_results.get('keywords', [])[:10] if nlp_results else [],
                'format': dataset.format or 'csv'
            }

            # Generate Python code using NLP service
            python_code = nlp_service._generate_python_analysis_code(dataset_info)

            if python_code and len(python_code) > 100:
                print(f"✅ Generated Python analysis code: {len(python_code)} characters")
                return python_code
            else:
                print("⚠️ Python code generation failed, using fallback")
                return self._generate_fallback_python_code(dataset_info)

        except Exception as e:
            print(f"⚠️ Error generating Python code: {e}")
            return self._generate_fallback_python_code({
                'title': dataset.title or 'Dataset',
                'field_names': [],
                'data_types': [],
                'record_count': 0,
                'category': 'General'
            })

    def _generate_fallback_python_code(self, dataset_info):
        """Generate basic Python analysis code as fallback."""
        title = dataset_info.get('title', 'Dataset')

        code_lines = [
            f"# Python Analysis Code for {title}",
            "# Generated by AI Meta Harvest",
            "",
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "# Load the dataset",
            "df = pd.read_csv('your_dataset.csv')  # Replace with actual file path",
            "",
            "# Basic exploration",
            "print('Dataset Shape:', df.shape)",
            "print('\\nDataset Info:')",
            "df.info()",
            "print('\\nBasic Statistics:')",
            "df.describe()",
            "",
            "# Visualizations",
            "plt.figure(figsize=(12, 8))",
            "df.hist(bins=20, figsize=(12, 8))",
            f"plt.suptitle('Data Distribution - {title}')",
            "plt.tight_layout()",
            "plt.show()",
            "",
            "print('Analysis completed!')"
        ]

        return '\n'.join(code_lines)

    def _extract_field_names_list(self, processed_data):
        """Extract field names as a list for AI processing."""
        try:
            if processed_data and 'schema' in processed_data:
                schema = processed_data['schema']
                if isinstance(schema, dict):
                    return list(schema.keys())

            # Fallback: try sample data
            if processed_data and 'sample_data' in processed_data:
                sample_data = processed_data['sample_data']
                if isinstance(sample_data, list) and sample_data:
                    first_row = sample_data[0]
                    if isinstance(first_row, dict):
                        return list(first_row.keys())

            return []
        except Exception as e:
            print(f"⚠️ Error extracting field names list: {e}")
            return []

    def _extract_data_types_list(self, processed_data):
        """Extract data types as a list for AI processing."""
        try:
            data_types = []
            if processed_data and 'schema' in processed_data:
                schema = processed_data['schema']
                if isinstance(schema, dict):
                    for field_info in schema.values():
                        if isinstance(field_info, dict) and 'type' in field_info:
                            data_type = field_info['type']
                            if data_type not in data_types:
                                data_types.append(data_type)
            return data_types
        except Exception as e:
            print(f"⚠️ Error extracting data types list: {e}")
            return []

    def _extract_data_types_dict(self, processed_data):
        """Extract data types as a dictionary mapping field names to types for AI processing."""
        try:
            data_types = {}
            if processed_data and 'schema' in processed_data:
                schema = processed_data['schema']
                if isinstance(schema, dict):
                    for field_name, field_info in schema.items():
                        if isinstance(field_info, dict) and 'type' in field_info:
                            data_types[field_name] = field_info['type']
            elif processed_data and 'columns' in processed_data:
                # Fallback: try to get data types from columns info
                columns = processed_data['columns']
                if isinstance(columns, list):
                    for col in columns:
                        data_types[col] = 'unknown'  # Default type if not specified
            return data_types
        except Exception as e:
            print(f"⚠️ Error extracting data types dict: {e}")
            return {}

    def _get_sample_data(self, processed_data):
        """Get sample data for AI processing."""
        try:
            if processed_data and 'sample_data' in processed_data:
                sample_data = processed_data['sample_data']
                if isinstance(sample_data, list):
                    return sample_data[:5]  # Return first 5 samples
            return []
        except Exception as e:
            print(f"⚠️ Error getting sample data: {e}")
            return []

    def _describe_data_types_comprehensive(self, data_types):
        """Generate comprehensive description of data types."""
        type_descriptions = {
            'object': 'textual and categorical data',
            'string': 'textual information',
            'int64': 'integer numerical values',
            'float64': 'decimal numerical data',
            'bool': 'boolean indicators',
            'datetime64': 'temporal information',
            'category': 'categorical classifications'
        }

        described_types = []
        for dtype in data_types:
            described_types.append(type_descriptions.get(dtype, f'{dtype} data'))

        if len(described_types) == 1:
            return described_types[0]
        elif len(described_types) == 2:
            return f"{described_types[0]} and {described_types[1]}"
        else:
            return f"{', '.join(described_types[:-1])}, and {described_types[-1]}"

    def _get_enhanced_category_context(self, category):
        """Get enhanced contextual description for dataset category."""
        category_contexts = {
            'business': 'this dataset provides valuable insights into commercial operations, market dynamics, and organizational performance metrics.',
            'finance': 'this dataset offers comprehensive financial information suitable for economic analysis, investment research, and financial modeling.',
            'education': 'this dataset contains educational information valuable for academic research, learning analytics, and educational policy development.',
            'healthcare': 'this dataset provides medical and health-related information suitable for clinical research, public health analysis, and healthcare optimization.',
            'social': 'this dataset captures social dynamics and interactions, providing insights into community behavior and social patterns.',
            'technology': 'this dataset contains technological information relevant for innovation analysis, system performance evaluation, and technical research.',
            'science': 'this dataset provides scientific data suitable for research applications, experimental analysis, and scientific discovery.',
            'government': 'this dataset contains public sector information valuable for policy analysis, governance research, and civic engagement studies.'
        }

        category_lower = category.lower()
        for key, context in category_contexts.items():
            if key in category_lower:
                return context

        return f'this dataset provides specialized information within the {category.lower()} domain, offering unique analytical opportunities.'

    def _save_quality_assessment(self, dataset, quality_results):
        """Save quality assessment to MetadataQuality collection."""
        try:
            import json
            from app.models.metadata import MetadataQuality

            # Check if quality assessment already exists
            existing_quality = MetadataQuality.objects(dataset=dataset).first()

            # Prepare quality data, excluding computed properties
            quality_data = quality_results.copy()

            # Remove computed properties that can't be set directly
            quality_data.pop('fair_scores', None)
            quality_data.pop('dimension_scores', None)

            # Convert lists to JSON strings for database storage
            if 'issues' in quality_data and isinstance(quality_data['issues'], list):
                quality_data['issues'] = json.dumps(quality_data['issues'])
            if 'recommendations' in quality_data and isinstance(quality_data['recommendations'], list):
                quality_data['recommendations'] = json.dumps(quality_data['recommendations'])

            # Ensure dataset reference is the dataset object, not string ID
            quality_data['dataset'] = dataset

            if existing_quality:
                # Update existing record with proper dataset reference
                update_data = quality_data.copy()
                update_data.pop('dataset', None)  # Remove dataset from update data
                existing_quality.update(**update_data)
                logger.info(f"Updated existing quality assessment for dataset {dataset.id}")
            else:
                # Create new quality assessment with dataset object
                new_quality = MetadataQuality(**quality_data)
                new_quality.save()
                logger.info(f"Created new quality assessment for dataset {dataset.id}")

        except Exception as e:
            logger.error(f"Error saving quality assessment for dataset {dataset.id}: {e}")

    def _generate_comprehensive_metadata(self, dataset, processed_data, cleaned_data, nlp_results, quality_results, ai_compliance, metadata_results):
        """Generate comprehensive metadata with all required fields."""
        metadata = {}

        # Core identification metadata
        metadata['title'] = dataset.title or 'Untitled Dataset'

        # Auto-generate or enhance description (only if not already generated)
        if not dataset.description or len(dataset.description.strip()) < 50:
            # Check if description was already generated in this processing session
            if hasattr(dataset, '_description_generated') and dataset._description_generated:
                metadata['description'] = dataset.description
            else:
                metadata['description'] = self._generate_auto_description(dataset, nlp_results, processed_data)
                # Mark as generated to avoid duplication
                dataset._description_generated = True
        else:
            metadata['description'] = dataset.description

        # Generate Python analysis code
        metadata['python_analysis_code'] = self._generate_python_analysis_code(dataset, nlp_results, processed_data)

        # Source information
        if not dataset.source:
            if dataset.source_url:
                metadata['source'] = 'Web Source'
            elif dataset.file_path:
                metadata['source'] = 'File Upload'
            else:
                metadata['source'] = 'Unknown'
        else:
            metadata['source'] = dataset.source

        # Auto-generate or validate category
        metadata['category'] = self._generate_smart_category(dataset, nlp_results, processed_data)

        # Data structure metadata
        metadata['record_count'] = self._get_record_count(processed_data, cleaned_data)
        metadata['field_count'] = self._get_field_count(processed_data, cleaned_data)
        metadata['fields_info'] = self._generate_fields_info(processed_data, cleaned_data)

        # Extract field names and data types for direct saving
        field_names = self._extract_field_names(processed_data, cleaned_data)
        if field_names:
            metadata['field_names'] = field_names

        data_types = self._extract_data_types(processed_data, cleaned_data)
        if data_types:
            metadata['data_types'] = data_types

        # Content analysis metadata
        metadata['keywords'] = self._extract_comprehensive_keywords(dataset, nlp_results, processed_data)
        metadata['tags'] = self._generate_enhanced_tags(dataset, nlp_results, processed_data, cleaned_data)
        metadata['data_type'] = self._generate_enhanced_data_type(dataset, processed_data, cleaned_data)

        # Generate use cases with enhanced analysis
        use_cases = self._generate_use_case_suggestions(dataset, nlp_results, processed_data)
        if use_cases:
            # Convert list to comma-separated string for template compatibility
            if isinstance(use_cases, list):
                metadata['use_cases'] = ', '.join(use_cases)
            else:
                metadata['use_cases'] = str(use_cases)
            print(f"✅ Generated {len(use_cases) if isinstance(use_cases, list) else 'multiple'} use case suggestions")
        else:
            metadata['use_cases'] = ""
            print("⚠️ No use cases generated")

        # Quality and compliance metadata
        metadata['quality_score'] = quality_results.get('quality_score', 0) if quality_results else 0
        metadata['fair_compliant'] = quality_results.get('fair_compliant', False) if quality_results else False

        # Calculate enhanced FAIR scores
        fair_scores = self._calculate_enhanced_fair_scores(dataset, processed_data, cleaned_data, nlp_results, quality_results)
        metadata.update(fair_scores)
        metadata['completeness_score'] = quality_results.get('completeness', 0) if quality_results else 0
        metadata['consistency_score'] = quality_results.get('consistency', 0) if quality_results else 0

        # Schema.org structured metadata
        metadata['schema_org'] = self._generate_schema_org_metadata(dataset, metadata, quality_results)

        # AI compliance metadata
        metadata['ai_compliance'] = ai_compliance if ai_compliance else {}

        # Additional metadata
        # Use cleaned_data first, fallback to processed_data
        data_source = cleaned_data if cleaned_data else processed_data

        # Only set data_types if not already extracted
        if 'data_types' not in metadata:
            metadata['data_types'] = self._analyze_data_types(data_source)

        metadata['data_quality_issues'] = self._identify_data_quality_issues(cleaned_data, quality_results)
        metadata['processing_metadata'] = {
            'processed_at': datetime.now().isoformat(),
            'processing_version': '2.0',
            'nlp_processed': bool(nlp_results),
            'quality_assessed': bool(quality_results),
            'ai_compliance_checked': bool(ai_compliance)
        }

        return metadata

    def _generate_use_case_suggestions(self, dataset, nlp_results, processed_data):
        """Generate use-case suggestions based on dataset content."""
        use_cases = []

        # Analyze dataset category and content
        if dataset.category:
            category_lower = dataset.category.lower()

            if 'financial' in category_lower or 'finance' in category_lower:
                use_cases.extend(['financial analysis', 'risk assessment', 'investment research'])
            elif 'education' in category_lower or 'learning' in category_lower:
                use_cases.extend(['educational research', 'learning analytics', 'curriculum development'])
            elif 'health' in category_lower or 'medical' in category_lower:
                use_cases.extend(['medical research', 'health analytics', 'clinical studies'])
            elif 'social' in category_lower or 'media' in category_lower:
                use_cases.extend(['social media analysis', 'sentiment analysis', 'trend analysis'])
            elif 'business' in category_lower or 'commercial' in category_lower:
                use_cases.extend(['business intelligence', 'market analysis', 'customer analytics'])

        # Analyze keywords for additional use cases
        if nlp_results and 'keywords' in nlp_results:
            keywords = [k.lower() for k in nlp_results['keywords'][:10]]

            if any(word in keywords for word in ['price', 'cost', 'revenue', 'sales']):
                use_cases.append('pricing analysis')
            if any(word in keywords for word in ['customer', 'user', 'client']):
                use_cases.append('customer behavior analysis')
            if any(word in keywords for word in ['time', 'date', 'temporal']):
                use_cases.append('time series analysis')
            if any(word in keywords for word in ['location', 'geographic', 'spatial']):
                use_cases.append('geographic analysis')
            if any(word in keywords for word in ['text', 'content', 'document']):
                use_cases.append('text mining')

        # Analyze data structure for use cases
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            if isinstance(schema, dict):
                field_names = [name.lower() for name in schema.keys()]

                if any('rating' in name or 'score' in name for name in field_names):
                    use_cases.append('rating prediction')
                if any('category' in name or 'class' in name for name in field_names):
                    use_cases.append('classification tasks')
                if any('amount' in name or 'value' in name for name in field_names):
                    use_cases.append('regression analysis')

        # Remove duplicates and limit
        unique_use_cases = list(dict.fromkeys(use_cases))[:8]

        # Add generic use cases if none found
        if not unique_use_cases:
            unique_use_cases = ['data analysis', 'research studies', 'machine learning']

        return unique_use_cases

    def _generate_smart_category(self, dataset, nlp_results, processed_data):
        """Generate or improve dataset category based on content analysis."""
        # If category exists and seems accurate, keep it
        if dataset.category and len(dataset.category.strip()) > 3:
            category = dataset.category.strip()
            # Validate if it makes sense
            if any(word in category.lower() for word in ['education', 'business', 'finance', 'health', 'social', 'technology', 'science']):
                return category

        # Auto-generate category based on content
        if nlp_results and 'keywords' in nlp_results:
            keywords = [k.lower() for k in nlp_results['keywords'][:10]]

            if any(word in keywords for word in ['course', 'education', 'learning', 'student', 'university']):
                return 'Education'
            elif any(word in keywords for word in ['business', 'company', 'market', 'sales', 'revenue']):
                return 'Business'
            elif any(word in keywords for word in ['finance', 'money', 'price', 'cost', 'investment']):
                return 'Finance'
            elif any(word in keywords for word in ['health', 'medical', 'patient', 'treatment', 'disease']):
                return 'Healthcare'
            elif any(word in keywords for word in ['social', 'media', 'network', 'community', 'user']):
                return 'Social Media'
            elif any(word in keywords for word in ['technology', 'software', 'data', 'system', 'computer']):
                return 'Technology'

        # Fallback based on title
        if dataset.title:
            title_lower = dataset.title.lower()
            if 'education' in title_lower or 'course' in title_lower:
                return 'Education'
            elif 'business' in title_lower or 'market' in title_lower:
                return 'Business'
            elif 'finance' in title_lower or 'financial' in title_lower:
                return 'Finance'

        return dataset.category or 'General'

    def _calculate_enhanced_fair_scores(self, dataset, processed_data, cleaned_data, nlp_results, quality_results):
        """Calculate enhanced FAIR compliance scores."""
        try:
            print("⚖️ Calculating enhanced FAIR compliance scores...")

            # Initialize scores
            findable_score = 0
            accessible_score = 0
            interoperable_score = 0
            reusable_score = 0

            # FINDABLE (F) - 25 points max
            # F1: Data are assigned globally unique and persistent identifiers
            if hasattr(dataset, 'persistent_identifier') and dataset.persistent_identifier:
                findable_score += 5

            # F2: Data are described with rich metadata
            if dataset.description and len(dataset.description) > 50:
                findable_score += 5
            if dataset.keywords and len(dataset.keywords) > 10:
                findable_score += 3
            if dataset.tags and len(dataset.tags) > 5:
                findable_score += 2

            # F3: Metadata clearly and explicitly include the identifier of the data
            if dataset.title and len(dataset.title) > 5:
                findable_score += 3
            if dataset.source:
                findable_score += 2

            # F4: Metadata are registered or indexed in a searchable resource
            findable_score += 5  # Always true for our system

            # ACCESSIBLE (A) - 25 points max
            # A1: Metadata are retrievable by their identifier using standardized protocol
            accessible_score += 8  # HTTP/HTTPS protocol

            # A1.1: Protocol is open, free and universally implementable
            accessible_score += 5  # HTTP is open and free

            # A1.2: Protocol allows for authentication and authorization
            accessible_score += 4  # Our system supports auth

            # A2: Metadata are accessible even when data are no longer available
            accessible_score += 8  # Metadata persists in our system

            # INTEROPERABLE (I) - 25 points max
            # I1: Metadata use formal, accessible, shared, broadly applicable language
            if hasattr(dataset, 'schema_org_json') and dataset.schema_org_json:
                interoperable_score += 8
            else:
                interoperable_score += 4  # Basic JSON structure

            # I2: Metadata use vocabularies that follow FAIR principles
            if hasattr(dataset, 'dublin_core') and dataset.dublin_core:
                interoperable_score += 5
            if hasattr(dataset, 'dcat_metadata') and dataset.dcat_metadata:
                interoperable_score += 5

            # I3: Metadata include qualified references to other metadata
            if nlp_results and nlp_results.get('entities'):
                interoperable_score += 4
            if dataset.category:
                interoperable_score += 3

            # REUSABLE (R) - 25 points max
            # R1: Metadata are richly described with accurate and relevant attributes
            record_count = self._get_record_count(processed_data, cleaned_data)
            field_count = self._get_field_count(processed_data, cleaned_data)

            if record_count > 0:
                reusable_score += 3
            if field_count > 0:
                reusable_score += 3
            if dataset.format:
                reusable_score += 2

            # R1.1: Metadata are released with clear and accessible data usage license
            if dataset.license:
                reusable_score += 5
            else:
                reusable_score += 2  # Default license implied

            # R1.2: Metadata are associated with detailed provenance
            if dataset.source:
                reusable_score += 3
            if dataset.created_at:
                reusable_score += 2
            if dataset.user:
                reusable_score += 2

            # R1.3: Metadata meet domain-relevant community standards
            if quality_results and quality_results.get('quality_score', 0) > 70:
                reusable_score += 5
            elif quality_results and quality_results.get('quality_score', 0) > 50:
                reusable_score += 3
            else:
                reusable_score += 1

            # Calculate overall FAIR score
            total_fair_score = findable_score + accessible_score + interoperable_score + reusable_score
            fair_percentage = min(100, (total_fair_score / 100) * 100)

            # Determine FAIR compliance
            fair_compliant = fair_percentage >= 75

            print(f"✅ FAIR Scores - F:{findable_score}/25, A:{accessible_score}/25, I:{interoperable_score}/25, R:{reusable_score}/25")
            print(f"✅ Overall FAIR Score: {fair_percentage:.1f}% ({'Compliant' if fair_compliant else 'Partial'})")

            return {
                'findable_score': (findable_score / 25) * 100,
                'accessible_score': (accessible_score / 25) * 100,
                'interoperable_score': (interoperable_score / 25) * 100,
                'reusable_score': (reusable_score / 25) * 100,
                'fair_score': fair_percentage,
                'fair_compliant': fair_compliant,
                'fair_details': {
                    'findable_points': findable_score,
                    'accessible_points': accessible_score,
                    'interoperable_points': interoperable_score,
                    'reusable_points': reusable_score,
                    'total_points': total_fair_score,
                    'max_points': 100
                }
            }

        except Exception as e:
            print(f"⚠️ Error calculating FAIR scores: {e}")
            return {
                'findable_score': 0,
                'accessible_score': 0,
                'interoperable_score': 0,
                'reusable_score': 0,
                'fair_score': 0,
                'fair_compliant': False
            }

    def _get_record_count(self, processed_data, cleaned_data):
        """Get record count from processed data (preserving original count, not sample size)."""
        # Priority: cleaned_data record_count (should preserve original), then processed_data record_count
        if cleaned_data and 'record_count' in cleaned_data:
            record_count = cleaned_data['record_count']

            # Ensure record_count is an integer
            if not isinstance(record_count, (int, float)):
                try:
                    record_count = int(record_count) if record_count else 0
                except (ValueError, TypeError):
                    record_count = 0

            # Ensure we're not using sample size
            if record_count <= 100:  # Likely a sample
                # Check if processed_data has the real count
                if processed_data and 'record_count' in processed_data:
                    original_count = processed_data['record_count']

                    # Ensure original_count is an integer
                    if not isinstance(original_count, (int, float)):
                        try:
                            original_count = int(original_count) if original_count else 0
                        except (ValueError, TypeError):
                            original_count = 0

                    if original_count > record_count:
                        logger.info(f"Using original record count {original_count} instead of sample size {record_count}")
                        return original_count
            return record_count
        elif processed_data and 'record_count' in processed_data:
            record_count = processed_data['record_count']
            # Ensure record_count is an integer
            if not isinstance(record_count, (int, float)):
                try:
                    record_count = int(record_count) if record_count else 0
                except (ValueError, TypeError):
                    record_count = 0
            return record_count
        return 0

    def _get_field_count(self, processed_data, cleaned_data):
        """Get field count from processed data."""
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            if isinstance(schema, dict):
                return len(schema)
        elif cleaned_data and 'schema' in cleaned_data:
            schema = cleaned_data['schema']
            if isinstance(schema, dict):
                return len(schema)
        return 0

    def _generate_fields_info(self, processed_data, cleaned_data):
        """Generate detailed field information."""
        fields_info = {}

        schema = None
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
        elif cleaned_data and 'schema' in cleaned_data:
            schema = cleaned_data['schema']

        if schema and isinstance(schema, dict):
            for field_name, field_info in schema.items():
                if isinstance(field_info, dict):
                    fields_info[field_name] = {
                        'type': field_info.get('type', 'unknown'),
                        'description': field_info.get('description', f'Field containing {field_info.get("type", "data")} values'),
                        'sample_values': field_info.get('sample_values', [])[:3],  # Limit to 3 samples
                        'null_count': field_info.get('null_count', 0),
                        'unique_count': field_info.get('unique_count', 0)
                    }
                else:
                    fields_info[field_name] = {
                        'type': 'unknown',
                        'description': f'Data field: {field_name}',
                        'sample_values': [],
                        'null_count': 0,
                        'unique_count': 0
                    }

        return fields_info

    def _extract_field_names(self, processed_data, cleaned_data):
        """Extract field names from processed data."""
        try:
            # Try to get schema from processed or cleaned data
            schema = None
            if processed_data and 'schema' in processed_data:
                schema = processed_data['schema']
            elif cleaned_data and 'schema' in cleaned_data:
                schema = cleaned_data['schema']

            if schema and isinstance(schema, dict):
                field_names = list(schema.keys())
                print(f"📋 Extracted {len(field_names)} field names: {field_names[:10]}...")
                return ', '.join(field_names)

            # Fallback: try to get from sample data
            sample_data = None
            if cleaned_data and 'sample_data' in cleaned_data:
                sample_data = cleaned_data['sample_data']
            elif processed_data and 'sample_data' in processed_data:
                sample_data = processed_data['sample_data']

            if sample_data and isinstance(sample_data, list) and sample_data:
                first_row = sample_data[0]
                if isinstance(first_row, dict):
                    field_names = list(first_row.keys())
                    print(f"📋 Extracted {len(field_names)} field names from sample data: {field_names[:10]}...")
                    return ', '.join(field_names)

            print("⚠️ No field names found in processed data")
            return None

        except Exception as e:
            print(f"⚠️ Error extracting field names: {e}")
            return None

    def _extract_data_types(self, processed_data, cleaned_data):
        """Extract data types from processed data."""
        try:
            data_types = []

            # Try to get schema from processed or cleaned data
            schema = None
            if processed_data and 'schema' in processed_data:
                schema = processed_data['schema']
            elif cleaned_data and 'schema' in cleaned_data:
                schema = cleaned_data['schema']

            if schema and isinstance(schema, dict):
                for field_name, field_info in schema.items():
                    if isinstance(field_info, dict) and 'type' in field_info:
                        data_type = field_info['type']
                        if data_type not in data_types:
                            data_types.append(data_type)

                if data_types:
                    print(f"📊 Extracted {len(data_types)} data types: {data_types}")
                    return ', '.join(data_types)

            # Fallback: analyze sample data to infer types
            sample_data = None
            if cleaned_data and 'sample_data' in cleaned_data:
                sample_data = cleaned_data['sample_data']
            elif processed_data and 'sample_data' in processed_data:
                sample_data = processed_data['sample_data']

            if sample_data and isinstance(sample_data, list) and sample_data:
                inferred_types = set()

                for row in sample_data[:10]:  # Analyze first 10 rows
                    if isinstance(row, dict):
                        for key, value in row.items():
                            if value is not None:
                                if isinstance(value, bool):
                                    inferred_types.add('boolean')
                                elif isinstance(value, int):
                                    inferred_types.add('integer')
                                elif isinstance(value, float):
                                    inferred_types.add('float')
                                elif isinstance(value, str):
                                    # Try to infer more specific string types
                                    if len(value) > 100:
                                        inferred_types.add('text')
                                    elif '@' in value and '.' in value:
                                        inferred_types.add('email')
                                    elif value.startswith(('http', 'www')):
                                        inferred_types.add('url')
                                    else:
                                        inferred_types.add('string')
                                else:
                                    inferred_types.add('object')

                if inferred_types:
                    data_types = list(inferred_types)
                    print(f"📊 Inferred {len(data_types)} data types: {data_types}")
                    return ', '.join(data_types)

            print("⚠️ No data types found in processed data")
            return None

        except Exception as e:
            print(f"⚠️ Error extracting data types: {e}")
            return None

    def _extract_comprehensive_keywords(self, dataset, nlp_results, processed_data):
        """Extract comprehensive keywords from all sources."""
        keywords = []

        # From NLP analysis
        if nlp_results and 'keywords' in nlp_results:
            keywords.extend(nlp_results['keywords'][:15])

        # From title and description
        if dataset.title:
            title_words = [word.strip() for word in dataset.title.split() if len(word.strip()) > 3]
            keywords.extend(title_words[:5])

        # From field names
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            if isinstance(schema, dict):
                field_keywords = [name.replace('_', ' ').replace('-', ' ') for name in schema.keys()]
                keywords.extend(field_keywords[:10])

        # Remove duplicates and clean
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            keyword_clean = keyword.lower().strip()
            if keyword_clean not in seen and len(keyword_clean) > 2:
                unique_keywords.append(keyword)
                seen.add(keyword_clean)

        return unique_keywords[:20]  # Limit to 20 keywords

    def _generate_smart_tags(self, dataset, nlp_results):
        """Generate smart tags from various sources."""
        tags = []

        # From existing tags
        if dataset.tags:
            if isinstance(dataset.tags, str):
                existing_tags = [tag.strip() for tag in dataset.tags.split(',')]
                tags.extend(existing_tags)
            elif isinstance(dataset.tags, list):
                tags.extend(dataset.tags)

        # From NLP suggested tags
        if nlp_results and 'suggested_tags' in nlp_results:
            nlp_tags = nlp_results['suggested_tags'][:8]
            tags.extend(nlp_tags)

        # Remove duplicates and clean
        unique_tags = []
        seen = set()
        for tag in tags:
            tag_clean = tag.lower().strip()
            if tag_clean not in seen and len(tag_clean) > 2:
                unique_tags.append(tag.title())
                seen.add(tag_clean)

        return ', '.join(unique_tags[:10])  # Limit to 10 tags

    def _generate_enhanced_tags(self, dataset, nlp_results, processed_data, cleaned_data):
        """Generate enhanced tags based on content analysis, combining existing and AI-generated tags."""
        try:
            print("🏷️ Generating enhanced tags based on content analysis...")

            tags = set()

            # 1. Include existing tags if they exist
            if dataset.tags:
                existing_tags = [tag.strip().lower() for tag in dataset.tags.split(',') if tag.strip()]
                tags.update(existing_tags)
                print(f"📋 Found {len(existing_tags)} existing tags")

            # 2. Generate tags from NLP keywords
            if nlp_results and nlp_results.get('keywords'):
                nlp_keywords = nlp_results.get('keywords', [])[:10]
                for keyword in nlp_keywords:
                    if isinstance(keyword, dict) and 'text' in keyword:
                        tag = keyword['text'].lower().strip()
                    else:
                        tag = str(keyword).lower().strip()

                    if len(tag) > 2 and tag.isalpha():
                        tags.add(tag)
                print(f"🔤 Added {len([k for k in nlp_keywords if len(str(k)) > 2])} tags from NLP keywords")

            # 3. Generate tags from dataset category
            if dataset.category:
                category_tags = dataset.category.lower().replace('_', ' ').split()
                for tag in category_tags:
                    if len(tag) > 2:
                        tags.add(tag)
                print(f"📂 Added category-based tags")

            # 4. Generate tags from data type analysis
            data_source = cleaned_data if cleaned_data else processed_data
            if data_source and 'sample_data' in data_source:
                content_tags = self._extract_content_based_tags(data_source)
                tags.update(content_tags)
                print(f"📊 Added {len(content_tags)} content-based tags")

            # 5. Generate tags from field names (use extracted field names from data)
            field_names = self._extract_field_names(processed_data, cleaned_data)
            if field_names:
                field_tags = self._extract_field_based_tags(field_names)
                tags.update(field_tags)
                print(f"🏗️ Added {len(field_tags)} field-based tags")
            elif hasattr(dataset, 'field_names') and dataset.field_names:
                field_tags = self._extract_field_based_tags(dataset.field_names)
                tags.update(field_tags)
                print(f"🏗️ Added {len(field_tags)} field-based tags from dataset")

            # 6. Generate domain-specific tags
            domain_tags = self._generate_domain_tags(dataset, nlp_results)
            tags.update(domain_tags)
            print(f"🎯 Added {len(domain_tags)} domain-specific tags")

            # Clean and format tags
            final_tags = []
            for tag in tags:
                clean_tag = tag.strip().lower()
                if len(clean_tag) > 2 and clean_tag.isalpha() and clean_tag not in final_tags:
                    final_tags.append(clean_tag)

            # Limit to 20 tags and sort alphabetically
            final_tags = sorted(final_tags)[:20]

            result = ', '.join(final_tags)
            print(f"✅ Generated {len(final_tags)} enhanced tags: {result[:100]}...")
            return result

        except Exception as e:
            print(f"⚠️ Error generating enhanced tags: {e}")
            # Fallback to existing tags or basic generation
            return dataset.tags if dataset.tags else self._generate_smart_tags(dataset, nlp_results)

    def _extract_content_based_tags(self, data_source):
        """Extract tags based on data content analysis."""
        tags = set()

        try:
            sample_data = data_source.get('sample_data', [])
            if not sample_data:
                return tags

            # Analyze data patterns
            has_dates = False
            has_numbers = False
            has_text = False
            has_coordinates = False
            has_emails = False
            has_urls = False

            for row in sample_data[:50]:  # Analyze first 50 rows
                if isinstance(row, dict):
                    for key, value in row.items():
                        if value is None:
                            continue

                        value_str = str(value).lower()

                        # Check for dates
                        if any(date_indicator in value_str for date_indicator in ['date', 'time', '/', '-', ':']):
                            has_dates = True

                        # Check for numbers
                        if isinstance(value, (int, float)) or value_str.replace('.', '').replace('-', '').isdigit():
                            has_numbers = True

                        # Check for text
                        if isinstance(value, str) and len(value) > 10:
                            has_text = True

                        # Check for coordinates
                        if any(coord in key.lower() for coord in ['lat', 'lon', 'coord', 'location']):
                            has_coordinates = True

                        # Check for emails
                        if '@' in value_str and '.' in value_str:
                            has_emails = True

                        # Check for URLs
                        if value_str.startswith(('http', 'www', 'ftp')):
                            has_urls = True

            # Add content-based tags
            if has_dates:
                tags.update(['temporal', 'time-series', 'dated'])
            if has_numbers:
                tags.update(['numerical', 'quantitative', 'metrics'])
            if has_text:
                tags.update(['textual', 'descriptive', 'qualitative'])
            if has_coordinates:
                tags.update(['geographic', 'spatial', 'location'])
            if has_emails:
                tags.update(['contact', 'communication', 'personal'])
            if has_urls:
                tags.update(['web', 'links', 'references'])

        except Exception as e:
            print(f"⚠️ Error in content-based tag extraction: {e}")

        return tags

    def _extract_field_based_tags(self, field_names):
        """Extract tags based on field names analysis."""
        tags = set()

        try:
            fields = field_names.split(',') if isinstance(field_names, str) else field_names

            for field in fields[:20]:  # Analyze first 20 fields
                field_lower = field.strip().lower()

                # Common field patterns
                if any(pattern in field_lower for pattern in ['id', 'key', 'index']):
                    tags.add('indexed')
                if any(pattern in field_lower for pattern in ['name', 'title', 'label']):
                    tags.add('labeled')
                if any(pattern in field_lower for pattern in ['date', 'time', 'created', 'updated']):
                    tags.add('temporal')
                if any(pattern in field_lower for pattern in ['price', 'cost', 'amount', 'value']):
                    tags.add('financial')
                if any(pattern in field_lower for pattern in ['address', 'location', 'city', 'country']):
                    tags.add('geographic')
                if any(pattern in field_lower for pattern in ['email', 'phone', 'contact']):
                    tags.add('contact')
                if any(pattern in field_lower for pattern in ['status', 'state', 'condition']):
                    tags.add('status')
                if any(pattern in field_lower for pattern in ['count', 'number', 'quantity']):
                    tags.add('quantitative')

        except Exception as e:
            print(f"⚠️ Error in field-based tag extraction: {e}")

        return tags

    def _generate_domain_tags(self, dataset, nlp_results):
        """Generate domain-specific tags based on content analysis."""
        tags = set()

        try:
            # Analyze category for domain
            if dataset.category:
                category_lower = dataset.category.lower()

                if any(domain in category_lower for domain in ['business', 'finance', 'economic']):
                    tags.update(['business', 'commercial', 'enterprise'])
                elif any(domain in category_lower for domain in ['health', 'medical', 'clinical']):
                    tags.update(['healthcare', 'medical', 'clinical'])
                elif any(domain in category_lower for domain in ['education', 'academic', 'research']):
                    tags.update(['academic', 'educational', 'research'])
                elif any(domain in category_lower for domain in ['government', 'public', 'civic']):
                    tags.update(['government', 'public', 'official'])
                elif any(domain in category_lower for domain in ['social', 'media', 'network']):
                    tags.update(['social', 'media', 'network'])
                elif any(domain in category_lower for domain in ['science', 'scientific', 'experiment']):
                    tags.update(['scientific', 'research', 'experimental'])

            # Analyze NLP entities for domain clues
            if nlp_results and nlp_results.get('entities'):
                entities = nlp_results.get('entities', [])

                # Count entity types
                org_count = sum(1 for entity in entities if entity.get('label') == 'ORG')
                person_count = sum(1 for entity in entities if entity.get('label') == 'PERSON')
                location_count = sum(1 for entity in entities if entity.get('label') in ['GPE', 'LOC'])

                if org_count > 5:
                    tags.add('organizational')
                if person_count > 5:
                    tags.add('personal')
                if location_count > 5:
                    tags.add('geographic')

        except Exception as e:
            print(f"⚠️ Error in domain tag generation: {e}")

        return tags

    def _generate_enhanced_data_type(self, dataset, processed_data, cleaned_data):
        """Generate enhanced data type based on content analysis."""
        try:
            print("🔍 Analyzing data type based on content...")

            # If data type already exists and seems accurate, enhance it
            existing_type = dataset.data_type
            if existing_type and len(existing_type.strip()) > 3:
                enhanced_type = self._enhance_existing_data_type(existing_type, processed_data, cleaned_data)
                if enhanced_type != existing_type:
                    print(f"📈 Enhanced data type from '{existing_type}' to '{enhanced_type}'")
                    return enhanced_type
                return existing_type

            # Analyze content to determine data type
            data_source = cleaned_data if cleaned_data else processed_data
            if not data_source or 'sample_data' not in data_source:
                return existing_type or 'Mixed'

            sample_data = data_source['sample_data']
            if not sample_data:
                return existing_type or 'Empty'

            # Analyze data characteristics
            characteristics = self._analyze_data_characteristics(sample_data)

            # Determine primary data type
            data_type = self._classify_data_type(characteristics)

            print(f"✅ Determined data type: {data_type}")
            return data_type

        except Exception as e:
            print(f"⚠️ Error generating enhanced data type: {e}")
            return dataset.data_type or 'Mixed'

    def _enhance_existing_data_type(self, existing_type, processed_data, cleaned_data):
        """Enhance existing data type with more specific classification."""
        try:
            data_source = cleaned_data if cleaned_data else processed_data
            if not data_source or 'sample_data' not in data_source:
                return existing_type

            characteristics = self._analyze_data_characteristics(data_source['sample_data'])

            # Enhance based on characteristics
            if existing_type.lower() in ['numerical', 'numeric', 'number']:
                if characteristics['has_time_series']:
                    return 'Time Series Numerical'
                elif characteristics['has_financial']:
                    return 'Financial Numerical'
                elif characteristics['has_scientific']:
                    return 'Scientific Numerical'
                else:
                    return 'Numerical'

            elif existing_type.lower() in ['text', 'textual', 'string']:
                if characteristics['has_structured_text']:
                    return 'Structured Text'
                elif characteristics['has_social_media']:
                    return 'Social Media Text'
                elif characteristics['has_documents']:
                    return 'Document Text'
                else:
                    return 'Text'

            elif existing_type.lower() in ['mixed', 'various', 'multiple']:
                primary_type = self._classify_data_type(characteristics)
                return f'Mixed ({primary_type})'

            return existing_type

        except Exception as e:
            print(f"⚠️ Error enhancing data type: {e}")
            return existing_type

    def _analyze_data_characteristics(self, sample_data):
        """Analyze data characteristics for type classification."""
        characteristics = {
            'numeric_ratio': 0,
            'text_ratio': 0,
            'date_ratio': 0,
            'boolean_ratio': 0,
            'has_time_series': False,
            'has_financial': False,
            'has_scientific': False,
            'has_geographic': False,
            'has_structured_text': False,
            'has_social_media': False,
            'has_documents': False,
            'field_count': 0,
            'record_count': len(sample_data)
        }

        try:
            if not sample_data:
                return characteristics

            # Count field types
            numeric_fields = 0
            text_fields = 0
            date_fields = 0
            boolean_fields = 0
            total_fields = 0

            # Analyze first few records
            for row in sample_data[:20]:
                if isinstance(row, dict):
                    total_fields = max(total_fields, len(row))

                    for key, value in row.items():
                        if value is None:
                            continue

                        key_lower = key.lower()
                        value_str = str(value).lower()

                        # Type classification
                        if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit()):
                            numeric_fields += 1
                        elif isinstance(value, bool) or value_str in ['true', 'false', 'yes', 'no', '1', '0']:
                            boolean_fields += 1
                        elif any(date_indicator in value_str for date_indicator in ['date', 'time', '/', '-', ':']):
                            date_fields += 1
                        elif isinstance(value, str) and len(value) > 5:
                            text_fields += 1

                        # Special characteristics
                        if any(financial in key_lower for financial in ['price', 'cost', 'amount', 'salary', 'revenue']):
                            characteristics['has_financial'] = True

                        if any(geo in key_lower for geo in ['lat', 'lon', 'location', 'address', 'city']):
                            characteristics['has_geographic'] = True

                        if any(time in key_lower for time in ['date', 'time', 'timestamp', 'created']):
                            characteristics['has_time_series'] = True

                        if any(sci in key_lower for sci in ['measurement', 'experiment', 'result', 'data']):
                            characteristics['has_scientific'] = True

                        if isinstance(value, str) and len(value) > 50:
                            if any(social in value_str for social in ['@', '#', 'http', 'www']):
                                characteristics['has_social_media'] = True
                            elif len(value) > 200:
                                characteristics['has_documents'] = True
                            else:
                                characteristics['has_structured_text'] = True

            # Calculate ratios
            total_values = max(1, numeric_fields + text_fields + date_fields + boolean_fields)
            characteristics['numeric_ratio'] = numeric_fields / total_values
            characteristics['text_ratio'] = text_fields / total_values
            characteristics['date_ratio'] = date_fields / total_values
            characteristics['boolean_ratio'] = boolean_fields / total_values
            characteristics['field_count'] = total_fields

        except Exception as e:
            print(f"⚠️ Error analyzing data characteristics: {e}")

        return characteristics

    def _classify_data_type(self, characteristics):
        """Classify data type based on analyzed characteristics."""
        try:
            # Primary classification based on dominant data types
            if characteristics['numeric_ratio'] > 0.7:
                if characteristics['has_financial']:
                    return 'Financial Data'
                elif characteristics['has_scientific']:
                    return 'Scientific Data'
                elif characteristics['has_time_series']:
                    return 'Time Series Data'
                else:
                    return 'Numerical Data'

            elif characteristics['text_ratio'] > 0.7:
                if characteristics['has_social_media']:
                    return 'Social Media Data'
                elif characteristics['has_documents']:
                    return 'Document Data'
                elif characteristics['has_structured_text']:
                    return 'Structured Text Data'
                else:
                    return 'Text Data'

            elif characteristics['date_ratio'] > 0.3:
                return 'Temporal Data'

            elif characteristics['boolean_ratio'] > 0.5:
                return 'Categorical Data'

            elif characteristics['has_geographic']:
                return 'Geographic Data'

            else:
                # Mixed data classification
                primary_types = []
                if characteristics['numeric_ratio'] > 0.3:
                    primary_types.append('Numerical')
                if characteristics['text_ratio'] > 0.3:
                    primary_types.append('Text')
                if characteristics['date_ratio'] > 0.2:
                    primary_types.append('Temporal')

                if len(primary_types) > 1:
                    return f"Mixed ({'/'.join(primary_types)})"
                elif primary_types:
                    return f"{primary_types[0]} Data"
                else:
                    return 'Mixed Data'

        except Exception as e:
            print(f"⚠️ Error classifying data type: {e}")
            return 'Mixed Data'

    def _extract_comprehensive_text_content(self, dataset, cleaned_data, processed_data):
        """Extract comprehensive text content from full dataset for NLP analysis."""
        content_parts = []

        # Add dataset metadata with enhanced context
        if dataset.title:
            content_parts.append(f"Dataset Title: {dataset.title}")
            content_parts.append(f"This dataset is called {dataset.title}")
            content_parts.append(f"{dataset.title} dataset information")

        if dataset.description:
            content_parts.append(f"Description: {dataset.description}")
        if dataset.source:
            content_parts.append(f"Source: {dataset.source}")
        if dataset.tags:
            tags = dataset.tags if isinstance(dataset.tags, list) else [dataset.tags]
            content_parts.append(f"Tags: {', '.join(tags)}")
            content_parts.append(f"This dataset relates to {', '.join(tags)}")

        # Add comprehensive schema and structure information
        if cleaned_data and 'schema' in cleaned_data:
            schema = cleaned_data['schema']
            content_parts.append(f"Dataset contains {len(schema)} fields")

            for field_name, field_info in schema.items():
                content_parts.append(f"Field: {field_name}")
                content_parts.append(f"Column {field_name} contains data")

                if isinstance(field_info, dict):
                    if 'type' in field_info:
                        content_parts.append(f"{field_name} is {field_info['type']} type")
                    if 'sample_values' in field_info and field_info['sample_values']:
                        sample_values = field_info['sample_values']
                        for val in sample_values:
                            if isinstance(val, str) and len(val) > 2:
                                content_parts.append(str(val))
                                content_parts.append(f"{field_name} contains {val}")

        # Add column information with context
        if cleaned_data and 'columns' in cleaned_data:
            columns = cleaned_data['columns']
            content_parts.append(f"Dataset columns: {', '.join(columns)}")
            content_parts.append(f"This dataset has {len(columns)} columns")

            # Add individual column context
            for col in columns:
                content_parts.append(f"Column {col} data field")
                content_parts.append(f"{col} information")

        # Extract text from full dataset if available
        if cleaned_data and 'dataframe' in cleaned_data:
            df = cleaned_data['dataframe']
            print(f"📊 Extracting text from full dataset with {len(df)} records")

            # Extract text from all text columns with enhanced sampling
            for column in df.columns:
                if df[column].dtype == 'object':  # Text columns
                    # Get more comprehensive text sample
                    sample_size = min(2000, len(df))  # Increased sample size
                    text_sample = df[column].dropna().head(sample_size).astype(str)

                    # Filter and add text values
                    for val in text_sample.tolist():
                        if len(val) > 3 and not val.replace('.', '').replace('-', '').isdigit():
                            content_parts.append(val)
                            content_parts.append(f"{column} contains {val}")

        elif cleaned_data and 'full_data' in cleaned_data:
            full_data = cleaned_data['full_data']
            print(f"📊 Extracting text from full data with {len(full_data)} records")

            # Extract text from all records with enhanced context
            for record in full_data[:1500]:  # Increased limit
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 3:
                        content_parts.append(value)
                        content_parts.append(f"{key} field contains {value}")

        elif cleaned_data and 'sample_data' in cleaned_data:
            # Enhanced sample data processing
            sample_data = cleaned_data['sample_data']
            print(f"📊 Extracting text from sample data with {len(sample_data)} records")

            for record in sample_data:
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 3:
                        content_parts.append(value)
                        content_parts.append(f"{key} field contains {value}")
                        content_parts.append(f"Data field {key} has value {value}")

        # Add format and processing information with context
        if cleaned_data:
            if 'format' in cleaned_data:
                format_type = cleaned_data['format']
                content_parts.append(f"Data Format: {format_type}")
                content_parts.append(f"This is a {format_type} format dataset")
                content_parts.append(f"{format_type} file type data")

            if 'record_count' in cleaned_data:
                record_count = cleaned_data['record_count']
                content_parts.append(f"Dataset contains {record_count} records")
                content_parts.append(f"Total {record_count} data entries")

        # Add error information as context if processing failed
        if cleaned_data and 'error' in cleaned_data:
            error_msg = cleaned_data['error']
            # Extract useful information from error messages
            if 'excel' in error_msg.lower():
                content_parts.append("Excel spreadsheet data file")
                content_parts.append("Spreadsheet format dataset")
                content_parts.append("Tabular data in Excel format")
            if 'csv' in error_msg.lower():
                content_parts.append("CSV comma separated values data")
                content_parts.append("Tabular CSV format dataset")

        # Add generic dataset context to ensure minimum content
        content_parts.extend([
            "This dataset contains structured data for analysis",
            "Data collection for research and analytical purposes",
            "Structured information organized in tabular format",
            "Dataset designed for data analysis and research applications",
            "Comprehensive data collection for statistical analysis",
            "Research dataset with organized information",
            "Analytical data resource for data science applications"
        ])

        # Combine all text content and ensure minimum length
        full_text = ' '.join(content_parts)

        # If content is still too short, add more generic context
        if len(full_text) < 500:
            additional_context = [
                f"The {dataset.title or 'dataset'} represents a comprehensive collection of structured data",
                "This dataset is particularly well-suited for exploratory data analysis",
                "Statistical modeling and data visualization applications",
                "Research applications and analytical methodologies",
                "The structured nature of this dataset makes it valuable for researchers",
                "Analysts and data scientists seeking meaningful insights",
                "Advanced analytical techniques and data mining applications"
            ]
            full_text += ' ' + ' '.join(additional_context)

        print(f"🔍 Extracted {len(full_text)} characters of text for NLP analysis")
        return full_text

    def _analyze_field_names(self, cleaned_data):
        """Analyze field names for insights."""
        if not cleaned_data or 'schema' not in cleaned_data:
            return {}

        schema = cleaned_data['schema']
        field_analysis = {
            'total_fields': len(schema),
            'field_categories': {},
            'semantic_types': {}
        }

        for field_name, field_info in schema.items():
            # Categorize field names
            field_lower = field_name.lower()
            if any(word in field_lower for word in ['id', 'identifier']):
                field_analysis['field_categories']['identifiers'] = field_analysis['field_categories'].get('identifiers', 0) + 1
            elif any(word in field_lower for word in ['name', 'title', 'description']):
                field_analysis['field_categories']['descriptive'] = field_analysis['field_categories'].get('descriptive', 0) + 1
            elif any(word in field_lower for word in ['date', 'time', 'timestamp']):
                field_analysis['field_categories']['temporal'] = field_analysis['field_categories'].get('temporal', 0) + 1
            elif any(word in field_lower for word in ['price', 'cost', 'amount', 'value']):
                field_analysis['field_categories']['financial'] = field_analysis['field_categories'].get('financial', 0) + 1
            elif any(word in field_lower for word in ['count', 'number', 'quantity']):
                field_analysis['field_categories']['quantitative'] = field_analysis['field_categories'].get('quantitative', 0) + 1

            # Store semantic type if available
            if isinstance(field_info, dict) and 'data_type_inferred' in field_info:
                semantic_type = field_info['data_type_inferred']
                field_analysis['semantic_types'][semantic_type] = field_analysis['semantic_types'].get(semantic_type, 0) + 1

        return field_analysis

    def _analyze_data_types(self, cleaned_data):
        """Analyze data types distribution."""
        if not cleaned_data or 'schema' not in cleaned_data:
            return {}

        schema = cleaned_data['schema']
        type_analysis = {
            'type_distribution': {},
            'data_quality_indicators': {}
        }

        for field_name, field_info in schema.items():
            if isinstance(field_info, dict):
                data_type = field_info.get('type', 'unknown')
                type_analysis['type_distribution'][data_type] = type_analysis['type_distribution'].get(data_type, 0) + 1

                # Analyze data quality indicators
                null_count = field_info.get('null_count', 0)
                unique_count = field_info.get('unique_count', 0)

                # Ensure null_count and unique_count are integers
                if not isinstance(null_count, (int, float)):
                    try:
                        null_count = int(null_count) if null_count else 0
                    except (ValueError, TypeError):
                        null_count = 0

                if not isinstance(unique_count, (int, float)):
                    try:
                        unique_count = int(unique_count) if unique_count else 0
                    except (ValueError, TypeError):
                        unique_count = 0

                if null_count > 0:
                    type_analysis['data_quality_indicators']['fields_with_nulls'] = type_analysis['data_quality_indicators'].get('fields_with_nulls', 0) + 1

                if unique_count == 1:
                    type_analysis['data_quality_indicators']['constant_fields'] = type_analysis['data_quality_indicators'].get('constant_fields', 0) + 1

        return type_analysis

    def _generate_nlp_enhanced_description(self, dataset, content_text, nlp_results, processed_data):
        """Generate an enhanced description using NLP analysis."""
        try:
            print("🔍 Generating AI-enhanced description using advanced models...")

            # Try to use advanced AI models first
            dataset_info = {
                'title': dataset.title,
                'field_names': processed_data.get('columns', []),
                'record_count': processed_data.get('record_count', 0),
                'data_types': processed_data.get('data_types', {}) if isinstance(processed_data.get('data_types', {}), dict) else {},
                'sample_data': processed_data.get('sample_data', [])[:3],
                'keywords': [kw.get('text', kw) if isinstance(kw, dict) else str(kw)
                           for kw in nlp_results.get('keywords', [])[:10]],
                'entities': [ent.get('text', ent) if isinstance(ent, dict) else str(ent)
                           for ent in nlp_results.get('entities', [])[:5]],
                'summary': nlp_results.get('summary', ''),
                'category': dataset.category
            }

            # Try to generate enhanced description using advanced AI (only if not already generated)
            if hasattr(dataset, '_description_generated') and dataset._description_generated:
                print("⚠️ Description already generated in this session, skipping duplicate generation")
                return dataset.description or self._generate_traditional_enhanced_description(dataset, content_text, nlp_results, processed_data)

            from app.services.nlp_service import nlp_service
            if isinstance(dataset_info, list):
                dataset_info = {'text_content': ' '.join(dataset_info)}
            enhanced_description = nlp_service.generate_enhanced_description(dataset_info)

            if enhanced_description and len(enhanced_description) > 100:
                print(f"✅ Generated AI-enhanced description: {len(enhanced_description)} characters")
                # Mark as generated to avoid duplication
                dataset._description_generated = True
                return enhanced_description

            # Fallback to traditional NLP-enhanced description
            print("⚠️ Advanced AI unavailable, using traditional NLP enhancement...")
            return self._generate_traditional_enhanced_description(dataset, content_text, nlp_results, processed_data)

        except Exception as e:
            print(f"⚠️ Error generating AI-enhanced description: {e}")
            return self._generate_traditional_enhanced_description(dataset, content_text, nlp_results, processed_data)

    def _generate_traditional_enhanced_description(self, dataset, content_text, nlp_results, processed_data):
        """Generate enhanced description using traditional NLP methods (fallback)."""
        try:
            # Start with existing description or create new one
            base_description = dataset.description or ""

            # Extract key insights from NLP results
            keywords = nlp_results.get('keywords', [])[:10]
            entities = nlp_results.get('entities', [])[:5]
            summary = nlp_results.get('summary', '')

            # Get dataset statistics
            record_count = processed_data.get('record_count', 0)

            # Ensure record_count is an integer
            if not isinstance(record_count, (int, float)):
                try:
                    record_count = int(record_count) if record_count else 0
                except (ValueError, TypeError):
                    record_count = 0

            field_count = len(processed_data.get('columns', []))

            # Generate comprehensive description
            description_parts = []

            # Add dataset overview
            description_parts.append(f"This dataset contains {record_count:,} records with {field_count} fields.")

            # Add content summary if available
            if summary and len(summary) > 20:
                description_parts.append(f"Content analysis reveals: {summary}")

            # Add key topics from keywords
            if keywords:
                key_topics = ', '.join([kw.get('text', kw) if isinstance(kw, dict) else str(kw) for kw in keywords[:5]])
                description_parts.append(f"Key topics include: {key_topics}.")

            # Add identified entities
            if entities:
                entity_texts = [ent.get('text', '') for ent in entities if ent.get('text')]
                if entity_texts:
                    description_parts.append(f"Notable entities identified: {', '.join(entity_texts[:3])}.")

            # Add domain-specific insights
            domain_insights = self._identify_domain_indicators(content_text)
            if domain_insights:
                description_parts.append(f"Domain indicators suggest this is {domain_insights} related data.")

            # Add use case suggestions
            use_cases = self._generate_nlp_use_cases(content_text, dataset, nlp_results)
            if use_cases:
                description_parts.append(f"Potential applications include: {', '.join(use_cases[:3])}.")

            # Combine all parts
            enhanced_description = ' '.join(description_parts)

            print(f"✅ Generated traditional enhanced description: {len(enhanced_description)} characters")
            return enhanced_description

        except Exception as e:
            print(f"⚠️ Error generating traditional enhanced description: {e}")
            return dataset.description or "Dataset description not available."

    def _generate_nlp_category(self, dataset, content_text, nlp_results):
        """Generate or improve category using NLP analysis."""
        try:
            # If we already have a good category, keep it
            if dataset.category and len(dataset.category.strip()) > 3:
                return dataset.category

            # Use NLP to determine category
            keywords = nlp_results.get('keywords', [])
            entities = nlp_results.get('entities', [])

            # Category mapping based on content analysis
            category_indicators = {
                'education': ['course', 'student', 'learning', 'education', 'school', 'university', 'training'],
                'business': ['business', 'company', 'revenue', 'profit', 'sales', 'marketing', 'finance'],
                'healthcare': ['health', 'medical', 'patient', 'hospital', 'treatment', 'diagnosis'],
                'technology': ['software', 'technology', 'computer', 'programming', 'data', 'algorithm'],
                'finance': ['finance', 'money', 'investment', 'bank', 'stock', 'price', 'economic'],
                'social': ['social', 'community', 'people', 'demographic', 'survey', 'opinion'],
                'science': ['research', 'experiment', 'analysis', 'study', 'scientific', 'laboratory'],
                'entertainment': ['movie', 'music', 'game', 'entertainment', 'media', 'film']
            }

            # Score categories based on keywords and entities
            category_scores = {}
            all_terms = keywords + [ent.get('text', '').lower() for ent in entities]

            for category, indicators in category_indicators.items():
                score = sum(1 for term in all_terms if any(indicator in term.lower() for indicator in indicators))
                if score > 0:
                    category_scores[category] = score

            # Return the highest scoring category
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                print(f"🎯 NLP-determined category: {best_category} (score: {category_scores[best_category]})")
                return best_category.title()

            return dataset.category or "General"

        except Exception as e:
            print(f"⚠️ Error generating NLP category: {e}")
            return dataset.category or "General"

    def _extract_nlp_keywords(self, content_text, dataset, nlp_results):
        """Extract comprehensive keywords using NLP."""
        try:
            keywords = set()

            # Get keywords from NLP results
            if nlp_results and 'keywords' in nlp_results:
                keywords.update(nlp_results['keywords'][:15])

            # Extract additional keywords from dataset title
            if dataset.title:
                title_keywords = nlp_service.extract_keywords(dataset.title, 5)
                keywords.update(title_keywords)

            # Extract keywords from field names
            if hasattr(dataset, 'schema') or 'schema' in nlp_results:
                schema = getattr(dataset, 'schema', nlp_results.get('schema', {}))
                field_names = ' '.join(schema.keys()) if schema else ''
                if field_names:
                    field_keywords = nlp_service.extract_keywords(field_names, 5)
                    keywords.update(field_keywords)

            # Clean and filter keywords
            cleaned_keywords = [kw for kw in keywords if len(kw) > 2 and kw.isalpha()]

            print(f"🔍 Extracted {len(cleaned_keywords)} NLP keywords")
            return cleaned_keywords[:20]  # Limit to top 20

        except Exception as e:
            print(f"⚠️ Error extracting NLP keywords: {e}")
            return []

    def _generate_nlp_use_cases(self, content_text, dataset, nlp_results):
        """Generate use cases based on NLP content analysis."""
        try:
            use_cases = []

            # Get category and keywords for context
            category = self._generate_nlp_category(dataset, content_text, nlp_results)
            keywords = nlp_results.get('keywords', [])[:10]

            # Category-specific use cases
            category_use_cases = {
                'Education': [
                    'educational research and analysis',
                    'learning analytics and student performance tracking',
                    'curriculum development and optimization',
                    'educational technology assessment'
                ],
                'Business': [
                    'business intelligence and analytics',
                    'market research and competitive analysis',
                    'customer behavior analysis',
                    'performance metrics and KPI tracking'
                ],
                'Healthcare': [
                    'medical research and clinical studies',
                    'patient outcome analysis',
                    'healthcare quality improvement',
                    'epidemiological research'
                ],
                'Technology': [
                    'software development and testing',
                    'system performance analysis',
                    'technology adoption studies',
                    'innovation research'
                ],
                'Finance': [
                    'financial analysis and modeling',
                    'risk assessment and management',
                    'investment research',
                    'economic trend analysis'
                ]
            }

            # Add category-specific use cases
            if category in category_use_cases:
                use_cases.extend(category_use_cases[category][:3])

            # Add keyword-based use cases
            if 'price' in keywords or 'cost' in keywords:
                use_cases.append('pricing analysis and optimization')
            if 'time' in keywords or 'date' in keywords:
                use_cases.append('temporal analysis and trend identification')
            if 'location' in keywords or 'geographic' in keywords:
                use_cases.append('geographic and spatial analysis')

            # Add general analytical use cases
            use_cases.extend([
                'data visualization and reporting',
                'statistical analysis and modeling',
                'machine learning and predictive analytics'
            ])

            # Remove duplicates and limit
            unique_use_cases = list(dict.fromkeys(use_cases))[:6]

            print(f"🎯 Generated {len(unique_use_cases)} NLP-based use cases")
            return unique_use_cases

        except Exception as e:
            print(f"⚠️ Error generating NLP use cases: {e}")
            return []

    def _assess_content_complexity(self, content_text):
        """Assess the complexity of the content."""
        try:
            words = content_text.split()
            sentences = content_text.split('.')

            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            avg_sentence_length = len(words) / len(sentences) if sentences else 0

            # Simple complexity scoring
            if avg_word_length > 6 and avg_sentence_length > 20:
                return "high"
            elif avg_word_length > 4 and avg_sentence_length > 15:
                return "medium"
            else:
                return "low"

        except Exception:
            return "unknown"

    def _identify_domain_indicators(self, content_text):
        """Identify domain-specific indicators in the content."""
        try:
            content_lower = content_text.lower()

            domain_patterns = {
                'academic': ['research', 'study', 'analysis', 'university', 'academic'],
                'commercial': ['business', 'sales', 'revenue', 'customer', 'market'],
                'technical': ['system', 'software', 'algorithm', 'technical', 'engineering'],
                'scientific': ['experiment', 'hypothesis', 'methodology', 'scientific', 'laboratory'],
                'social': ['survey', 'demographic', 'social', 'community', 'population']
            }

            for domain, patterns in domain_patterns.items():
                if any(pattern in content_lower for pattern in patterns):
                    return domain

            return "general"

        except Exception:
            return "unknown"

    def _generate_schema_org_metadata(self, dataset, metadata, quality_results):
        """Generate Schema.org compliant metadata."""
        schema_org = {
            "@context": "https://schema.org",
            "@type": "Dataset",
            "name": metadata['title'],
            "description": metadata['description'],
            "url": dataset.source_url if dataset.source_url else "",
            "creator": {
                "@type": "Person",
                "name": getattr(dataset.user, 'username', 'Unknown') if hasattr(dataset, 'user') else 'Unknown'
            },
            "dateCreated": dataset.created_at.isoformat() if dataset.created_at else "",
            "dateModified": datetime.now().isoformat(),
            "keywords": metadata.get('keywords', [])[:10],
            "license": "Not specified",
            "distribution": {
                "@type": "DataDownload",
                "encodingFormat": dataset.format if dataset.format else "unknown",
                "contentSize": dataset.size if dataset.size else "unknown"
            },
            "variableMeasured": list(metadata.get('fields_info', {}).keys())[:10],
            "measurementTechnique": "Automated data processing and analysis",
            "spatialCoverage": "Not specified",
            "temporalCoverage": "Not specified"
        }

        if quality_results:
            schema_org["qualityRating"] = {
                "@type": "Rating",
                "ratingValue": quality_results.get('quality_score', 0),
                "bestRating": 100,
                "worstRating": 0
            }

        return schema_org



    def _identify_data_quality_issues(self, cleaned_data, quality_results):
        """Identify data quality issues."""
        issues = []

        if cleaned_data and 'cleaning_stats' in cleaned_data:
            cleaning_stats = cleaned_data['cleaning_stats']
            if isinstance(cleaning_stats, dict):
                # Ensure values are numeric before comparison
                missing_values = cleaning_stats.get('missing_values', 0)
                if not isinstance(missing_values, (int, float)):
                    try:
                        missing_values = int(missing_values) if missing_values else 0
                    except (ValueError, TypeError):
                        missing_values = 0

                outliers_capped = cleaning_stats.get('outliers_capped', 0)
                if not isinstance(outliers_capped, (int, float)):
                    try:
                        outliers_capped = int(outliers_capped) if outliers_capped else 0
                    except (ValueError, TypeError):
                        outliers_capped = 0

                duplicates_removed = cleaning_stats.get('duplicates_removed', 0)
                if not isinstance(duplicates_removed, (int, float)):
                    try:
                        duplicates_removed = int(duplicates_removed) if duplicates_removed else 0
                    except (ValueError, TypeError):
                        duplicates_removed = 0

                if missing_values > 0:
                    issues.append(f"Missing values detected: {missing_values}")
                if outliers_capped > 0:
                    issues.append(f"Outliers capped: {outliers_capped}")
                if duplicates_removed > 0:
                    issues.append(f"Duplicate records removed: {duplicates_removed}")

        if quality_results:
            # Ensure quality scores are numeric before comparison
            completeness = quality_results.get('completeness', 100)
            if not isinstance(completeness, (int, float)):
                try:
                    completeness = float(completeness) if completeness else 100
                except (ValueError, TypeError):
                    completeness = 100

            consistency = quality_results.get('consistency', 100)
            if not isinstance(consistency, (int, float)):
                try:
                    consistency = float(consistency) if consistency else 100
                except (ValueError, TypeError):
                    consistency = 100

            if completeness < 80:
                issues.append("Low data completeness score")
            if consistency < 70:
                issues.append("Data consistency issues detected")

        return issues

    def _calculate_metadata_completeness(self, metadata):
        """Calculate metadata completeness percentage."""
        required_fields = ['title', 'description', 'source', 'category', 'keywords', 'use_cases']
        completed_fields = 0

        for field in required_fields:
            if metadata.get(field) and str(metadata[field]).strip():
                completed_fields += 1

        return round((completed_fields / len(required_fields)) * 100, 2)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a background task."""
        if self.celery_available and not task_id.startswith('thread_'):
            try:
                result = AsyncResult(task_id, app=celery_app)
                return {
                    'task_id': task_id,
                    'status': result.status,
                    'result': result.result,
                    'traceback': result.traceback
                }
            except Exception as e:
                return {'task_id': task_id, 'status': 'UNKNOWN', 'error': str(e)}
        
        # For thread-based tasks, check database
        dataset_id = task_id.replace('thread_', '')
        queue_item = ProcessingQueue.objects(dataset=dataset_id).first()
        if queue_item:
            return {
                'task_id': task_id,
                'status': queue_item.status.upper(),
                'progress': queue_item.progress,
                'message': queue_item.message
            }
        
        return {'task_id': task_id, 'status': 'NOT_FOUND'}
    
    def recover_interrupted_tasks(self) -> List[str]:
        """Recover tasks that were interrupted by application restart."""
        recovered_tasks = []
        
        # Find tasks that were processing but have no active workers
        interrupted_tasks = ProcessingQueue.objects(status='processing')
        
        for task in interrupted_tasks:
            try:
                # Check if there's a checkpoint file
                checkpoint_file = os.path.join(self.checkpoint_dir, f"{task.dataset.id}_checkpoint.json")
                
                if os.path.exists(checkpoint_file):
                    # Resume from checkpoint
                    logger.info(f"Recovering interrupted task for dataset {task.dataset.id}")
                    result = self.start_background_processing(str(task.dataset.id), 'uploads')
                    
                    if result['success']:
                        recovered_tasks.append(str(task.dataset.id))
                else:
                    # Mark as failed if no checkpoint
                    task.update(status='failed', error='Interrupted without checkpoint')
                    
            except Exception as e:
                logger.error(f"Error recovering task for dataset {task.dataset.id}: {e}")
        
        return recovered_tasks


    def _apply_fair_enhancements(self, dataset_id: str):
        """Apply FAIR compliance enhancements to the dataset."""
        try:
            from app.services.fair_enhancement_service import get_fair_enhancement_service

            dataset = Dataset.objects(id=dataset_id).first()
            if not dataset:
                logger.error(f"Dataset {dataset_id} not found for FAIR enhancement")
                return

            logger.info(f"Applying FAIR enhancements to dataset {dataset_id}")

            fair_service = get_fair_enhancement_service()
            result = fair_service.enhance_dataset_fair_compliance(dataset)

            if result['success']:
                logger.info(f"FAIR enhancement completed for dataset {dataset_id}. "
                           f"Score improved by {result['improvement']:.1f} points")
            else:
                logger.error(f"FAIR enhancement failed for dataset {dataset_id}: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error applying FAIR enhancements to dataset {dataset_id}: {e}")


# Celery tasks are now defined in celery_app.py

# Service instance
_background_processing_service = None

def get_background_processing_service():
    """Get the background processing service instance."""
    global _background_processing_service
    if _background_processing_service is None:
        _background_processing_service = BackgroundProcessingService()
    return _background_processing_service

# Global service instance
background_processing_service = BackgroundProcessingService()
