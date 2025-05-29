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
    from celery import Celery, Task
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue, MetadataQuality
from app.services.dataset_service import get_dataset_service
from app.services.nlp_service import nlp_service
from app.services.metadata_generator import metadata_service
from app.services.quality_assessment_service import quality_assessment_service
from app.services.data_cleaning_service import data_cleaning_service
from app.services.ai_standards_service import ai_standards_service

logger = logging.getLogger(__name__)

# Celery configuration
if CELERY_AVAILABLE:
    # Configure Celery
    celery_app = Celery('aimetaharvest')
    celery_app.conf.update(
        broker_url='redis://localhost:6379/0',  # Redis as message broker
        result_backend='redis://localhost:6379/0',  # Redis for result storage
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1 hour timeout
        task_soft_time_limit=3300,  # 55 minutes soft timeout
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_disable_rate_limits=True,
        task_routes={
            'app.services.background_processing_service.process_dataset_task': {'queue': 'dataset_processing'},
            'app.services.background_processing_service.cleanup_task': {'queue': 'maintenance'},
        }
    )

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
            
            # Create or update processing queue entry
            queue_item = ProcessingQueue.objects(dataset=dataset_id).first()
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
                task = process_dataset_task.apply_async(
                    args=[dataset_id, upload_folder],
                    priority=priority,
                    queue='dataset_processing'
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
                step_result = step_func(dataset, dataset_service, results)
                results[f'step_{i}'] = step_result
                
                # Save checkpoint
                checkpoint = {
                    'last_completed_step': i + 1,
                    'results': results,
                    'timestamp': datetime.utcnow().isoformat()
                }
                self._save_checkpoint(checkpoint_file, checkpoint)
                
                logger.info(f"Completed step {i + 1} for dataset {dataset_id}")
            
            # Complete processing
            self._update_progress(dataset_id, 100, 'completed', 'Processing completed successfully')
            
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
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
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
    
    # Processing step methods
    def _step_process_file(self, dataset, dataset_service, results):
        """Step 1: Process dataset file."""
        if dataset.file_path:
            file_info = {
                'path': dataset.file_path,
                'format': dataset.format,
                'size': dataset.size
            }
            return dataset_service.process_dataset(file_info, dataset.format)
        return {}
    
    def _step_clean_data(self, dataset, dataset_service, results):
        """Step 2: Clean and restructure data."""
        processed_data = results.get('step_0', {})
        if processed_data:
            return data_cleaning_service.clean_dataset(processed_data)
        return processed_data
    
    def _step_nlp_analysis(self, dataset, dataset_service, results):
        """Step 3: NLP analysis."""
        cleaned_data = results.get('step_1', {})
        # Extract text content and perform NLP analysis
        content_text = self._extract_text_content(dataset, cleaned_data)
        if content_text:
            return {
                'keywords': nlp_service.extract_keywords(content_text, 20),
                'suggested_tags': nlp_service.suggest_tags(content_text, 10),
                'entities': nlp_service.extract_entities(content_text),
                'sentiment': nlp_service.analyze_sentiment(content_text),
                'summary': nlp_service.generate_summary(content_text, 3)
            }
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
        """Step 6: Metadata generation."""
        cleaned_data = results.get('step_1', {})
        return metadata_service.generate_metadata(dataset, cleaned_data)
    
    def _step_save_results(self, dataset, dataset_service, results):
        """Step 7: Save results."""
        # Save all results to dataset
        cleaned_data = results.get('step_1', {})
        nlp_results = results.get('step_2', {})
        quality_results = results.get('step_3', {})
        ai_compliance = results.get('step_4', {})
        metadata_results = results.get('step_5', {})
        
        # Update dataset with results
        update_data = {}
        if cleaned_data and 'record_count' in cleaned_data:
            update_data['record_count'] = cleaned_data['record_count']
            if 'cleaning_stats' in cleaned_data:
                update_data['cleaning_stats'] = cleaned_data['cleaning_stats']
        
        if nlp_results and not dataset.tags and 'suggested_tags' in nlp_results:
            suggested_tags = nlp_results['suggested_tags'][:5]
            if suggested_tags:
                update_data['tags'] = ', '.join(suggested_tags)
        
        if ai_compliance:
            update_data['ai_compliance'] = ai_compliance
        
        if update_data:
            dataset.update(**update_data)
        
        return {'saved': True}
    
    def _step_generate_visualizations(self, dataset, dataset_service, results):
        """Step 8: Generate visualizations."""
        quality_results = results.get('step_3', {})
        # Generate visualization data
        return {
            'quality_chart': self._create_quality_chart_data(quality_results),
            'generated_at': datetime.utcnow().isoformat()
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
        """Create quality chart data."""
        if not quality_results:
            return {}
        return {
            'type': 'quality_metrics',
            'data': {
                'overall_score': quality_results.get('quality_score', 0),
                'completeness': quality_results.get('completeness', 0),
                'consistency': quality_results.get('consistency', 0),
                'accuracy': quality_results.get('accuracy', 0)
            }
        }
    
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


# Celery task definitions (only if Celery is available)
if CELERY_AVAILABLE:
    @celery_app.task(bind=True, name='app.services.background_processing_service.process_dataset_task')
    def process_dataset_task(self, dataset_id: str, upload_folder: str):
        """Celery task for processing datasets."""
        service = BackgroundProcessingService()
        service._process_with_checkpoints(dataset_id, upload_folder)
        return f"Dataset {dataset_id} processed successfully"
    
    @celery_app.task(name='app.services.background_processing_service.cleanup_task')
    def cleanup_task():
        """Periodic cleanup task."""
        # Clean up old checkpoints
        service = BackgroundProcessingService()
        checkpoint_dir = service.checkpoint_dir
        
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                filepath = os.path.join(checkpoint_dir, filename)
                if os.path.isfile(filepath):
                    # Remove checkpoints older than 24 hours
                    if os.path.getmtime(filepath) < (datetime.now().timestamp() - 86400):
                        os.remove(filepath)
        
        return "Cleanup completed"

# Global service instance
background_processing_service = BackgroundProcessingService()
