"""
Persistent Processing Service for AIMetaHarvest.

This service provides enhanced processing that survives application restarts
without requiring external dependencies like Redis or Celery.

Features:
- File-based task persistence
- Checkpoint-based recovery
- Non-daemon threads for longer survival
- Automatic resumption on restart
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue, MetadataQuality
from app.services.dataset_service import get_dataset_service
from app.services.nlp_service import nlp_service
from app.services.metadata_generator import metadata_service
from app.services.quality_assessment_service import quality_assessment_service
from app.services.data_cleaning_service import data_cleaning_service
from app.services.ai_standards_service import ai_standards_service
from app.services.description_generator import description_generator
from app.services.data_visualization_service import data_visualization_service
from app.services.metadata_completion_service import metadata_completion_service

logger = logging.getLogger(__name__)

class PersistentProcessingService:
    """
    Enhanced processing service with persistence and recovery capabilities.
    
    This service provides:
    - File-based task persistence
    - Checkpoint-based recovery from interruptions
    - Non-daemon threads that survive longer
    - Automatic task resumption on application restart
    """
    
    def __init__(self):
        """Initialize the persistent processing service."""
        self.active_processes = {}
        self.progress_callbacks = {}
        self.checkpoint_dir = "app/cache/checkpoints"
        self.task_dir = "app/cache/tasks"
        self.ensure_directories()
        
        # Recovery on initialization
        self.recover_interrupted_tasks()
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.task_dir, exist_ok=True)
    
    def start_persistent_processing(self, dataset_id: str, upload_folder: str, progress_callback: Optional = None) -> bool:
        """
        Start persistent processing for a dataset.
        
        Args:
            dataset_id: ID of the dataset to process
            upload_folder: Path to upload folder
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if processing started successfully
        """
        try:
            # Check if already processing
            if dataset_id in self.active_processes:
                return False
            
            # Get dataset
            dataset = Dataset.find_by_id(dataset_id)
            if not dataset:
                return False
            
            # Create or update processing queue entry
            queue_item = ProcessingQueue.objects(dataset=dataset_id).first()
            if not queue_item:
                queue_item = ProcessingQueue.create(
                    dataset=dataset_id,
                    status='pending',
                    priority=1
                )
            else:
                queue_item.update(
                    status='pending',
                    progress=0,
                    started_at=None,
                    completed_at=None,
                    error=None
                )
            
            # Create task file for persistence
            task_file = os.path.join(self.task_dir, f"{dataset_id}_task.json")
            task_data = {
                'dataset_id': dataset_id,
                'upload_folder': upload_folder,
                'status': 'pending',
                'created_at': datetime.utcnow().isoformat(),
                'pid': os.getpid()
            }
            
            with open(task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
            
            # Store progress callback
            if progress_callback:
                self.progress_callbacks[dataset_id] = progress_callback
            
            # Start processing in non-daemon thread (survives longer)
            thread = threading.Thread(
                target=self._process_with_persistence,
                args=(dataset_id, upload_folder),
                daemon=False  # Non-daemon thread
            )
            thread.start()
            
            # Track the process
            self.active_processes[dataset_id] = {
                'thread': thread,
                'started_at': datetime.utcnow(),
                'status': 'starting',
                'task_file': task_file
            }
            
            logger.info(f"Started persistent processing for dataset {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting persistent processing for dataset {dataset_id}: {e}")
            return False
    
    def _process_with_persistence(self, dataset_id: str, upload_folder: str):
        """Process dataset with full persistence and recovery."""
        task_file = os.path.join(self.task_dir, f"{dataset_id}_task.json")
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{dataset_id}_checkpoint.json")
        
        try:
            # Update task status
            self._update_task_file(task_file, {'status': 'processing', 'started_at': datetime.utcnow().isoformat()})
            
            # Load existing checkpoint if available
            checkpoint = self._load_checkpoint(checkpoint_file)
            start_step = checkpoint.get('last_completed_step', 0)
            
            logger.info(f"Processing dataset {dataset_id}, starting from step {start_step}")
            
            # Update status
            self._update_progress(dataset_id, start_step * 12.5, 'processing', 
                               f'{"Resuming" if start_step > 0 else "Starting"} processing...')
            
            # Get dataset and services
            dataset = Dataset.find_by_id(dataset_id)
            dataset_service = get_dataset_service(upload_folder)
            
            if not dataset:
                raise Exception('Dataset not found')
            
            # Processing steps with detailed checkpointing
            steps = [
                (self._step_process_file, "Processing dataset file", 10.0),
                (self._step_clean_data, "Cleaning and restructuring data", 20.0),
                (self._step_nlp_analysis, "Performing NLP analysis", 30.0),
                (self._step_quality_assessment, "Assessing dataset quality", 40.0),
                (self._step_ai_standards, "Assessing AI standards compliance", 50.0),
                (self._step_generate_description, "Generating intelligent description", 60.0),
                (self._step_complete_metadata, "Completing and enhancing metadata", 70.0),
                (self._step_metadata_generation, "Generating comprehensive metadata", 80.0),
                (self._step_save_results, "Saving processing results", 90.0),
                (self._step_generate_visualizations, "Generating visualizations", 100.0)
            ]
            
            # Execute steps starting from checkpoint
            results = checkpoint.get('results', {})
            
            for i, (step_func, step_name, progress) in enumerate(steps):
                if i < start_step:
                    continue  # Skip already completed steps
                
                self._update_progress(dataset_id, progress - 5, 'processing', f'{step_name}...')
                
                # Execute step with error handling
                try:
                    step_result = step_func(dataset, dataset_service, results)
                    results[f'step_{i}'] = step_result
                    
                    # Save checkpoint after each step
                    checkpoint = {
                        'last_completed_step': i + 1,
                        'results': results,
                        'timestamp': datetime.utcnow().isoformat(),
                        'dataset_id': dataset_id
                    }
                    self._save_checkpoint(checkpoint_file, checkpoint)
                    
                    # Update progress
                    self._update_progress(dataset_id, progress, 'processing', f'{step_name} completed')
                    
                    logger.info(f"Completed step {i + 1}/{len(steps)} for dataset {dataset_id}: {step_name}")
                    
                    # Small delay to allow for interruption detection
                    time.sleep(0.1)
                    
                except Exception as step_error:
                    logger.error(f"Error in step {i + 1} for dataset {dataset_id}: {step_error}")
                    # Save error checkpoint
                    checkpoint['error'] = {
                        'step': i,
                        'message': str(step_error),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    self._save_checkpoint(checkpoint_file, checkpoint)
                    raise step_error
            
            # Complete processing
            self._update_progress(dataset_id, 100, 'completed', 'Processing completed successfully')
            
            # Update task file
            self._update_task_file(task_file, {
                'status': 'completed',
                'completed_at': datetime.utcnow().isoformat()
            })
            
            # Clean up files
            self._cleanup_task_files(dataset_id)
            
            logger.info(f"Successfully completed processing for dataset {dataset_id}")
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Error processing dataset {dataset_id}: {e}")
            
            # Update task file with error
            self._update_task_file(task_file, {
                'status': 'failed',
                'error': error_msg,
                'failed_at': datetime.utcnow().isoformat()
            })
            
            self._update_progress(dataset_id, 0, 'failed', error_msg)
            
        finally:
            # Clean up active process tracking
            if dataset_id in self.active_processes:
                del self.active_processes[dataset_id]
            if dataset_id in self.progress_callbacks:
                del self.progress_callbacks[dataset_id]
    
    def _load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load processing checkpoint."""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    logger.info(f"Loaded checkpoint: step {checkpoint.get('last_completed_step', 0)}")
                    return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {}
    
    def _save_checkpoint(self, checkpoint_file: str, checkpoint: Dict[str, Any]):
        """Save processing checkpoint."""
        try:
            # Convert datetime objects to ISO format strings
            serializable_checkpoint = self._make_json_serializable(checkpoint)
            with open(checkpoint_file, 'w') as f:
                json.dump(serializable_checkpoint, f, indent=2)
            logger.debug(f"Saved checkpoint: step {checkpoint.get('last_completed_step', 0)}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Other datetime-like objects
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, '__dict__'):  # Custom objects
            return str(obj)
        else:
            return obj
    
    def _update_task_file(self, task_file: str, updates: Dict[str, Any]):
        """Update task file with new information."""
        try:
            if os.path.exists(task_file):
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
            else:
                task_data = {}
            
            task_data.update(updates)
            task_data['updated_at'] = datetime.utcnow().isoformat()
            
            with open(task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update task file: {e}")
    
    def _cleanup_task_files(self, dataset_id: str):
        """Clean up task and checkpoint files after successful completion."""
        try:
            task_file = os.path.join(self.task_dir, f"{dataset_id}_task.json")
            checkpoint_file = os.path.join(self.checkpoint_dir, f"{dataset_id}_checkpoint.json")
            
            if os.path.exists(task_file):
                os.remove(task_file)
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                
            logger.info(f"Cleaned up task files for dataset {dataset_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup task files: {e}")
    
    def recover_interrupted_tasks(self) -> List[str]:
        """Recover tasks that were interrupted by application restart."""
        recovered_tasks = []
        
        try:
            if not os.path.exists(self.task_dir):
                return recovered_tasks
            
            current_pid = os.getpid()
            
            for filename in os.listdir(self.task_dir):
                if filename.endswith('_task.json'):
                    task_file = os.path.join(self.task_dir, filename)
                    
                    try:
                        with open(task_file, 'r') as f:
                            task_data = json.load(f)
                        
                        dataset_id = task_data.get('dataset_id')
                        old_pid = task_data.get('pid')
                        status = task_data.get('status')
                        
                        # Check if task was processing and from different process
                        if status == 'processing' and old_pid != current_pid and dataset_id:
                            logger.info(f"Recovering interrupted task for dataset {dataset_id}")
                            
                            # Check if dataset still exists
                            dataset = Dataset.find_by_id(dataset_id)
                            if dataset:
                                # Restart processing
                                upload_folder = task_data.get('upload_folder', 'uploads')
                                success = self.start_persistent_processing(dataset_id, upload_folder)
                                
                                if success:
                                    recovered_tasks.append(dataset_id)
                                    logger.info(f"Successfully recovered task for dataset {dataset_id}")
                                else:
                                    logger.warning(f"Failed to recover task for dataset {dataset_id}")
                            else:
                                # Dataset no longer exists, clean up
                                os.remove(task_file)
                                logger.info(f"Cleaned up task file for deleted dataset {dataset_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing task file {filename}: {e}")
            
            if recovered_tasks:
                logger.info(f"Recovered {len(recovered_tasks)} interrupted tasks: {recovered_tasks}")
            
        except Exception as e:
            logger.error(f"Error during task recovery: {e}")
        
        return recovered_tasks
    
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
                
            # Call progress callback if available
            if dataset_id in self.progress_callbacks:
                callback = self.progress_callbacks[dataset_id]
                callback(dataset_id, progress, status, message)
                
        except Exception as e:
            logger.error(f"Error updating progress for dataset {dataset_id}: {e}")
    
    # Processing step methods (same as background_processing_service.py)
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

    def _step_generate_description(self, dataset, dataset_service, results):
        """Step 6: Generate intelligent description."""
        cleaned_data = results.get('step_1', {})
        nlp_results = results.get('step_2', {})
        quality_results = results.get('step_3', {})

        # Generate comprehensive description
        generated_description = description_generator.generate_description(
            dataset, cleaned_data, nlp_results, quality_results
        )

        # Update dataset with generated description if it's better than existing
        if generated_description and len(generated_description.strip()) > 50:
            # Check if we should update the description
            existing_desc = dataset.description or ""

            # Update if no existing description or existing is very short
            if not existing_desc or len(existing_desc.strip()) < 100:
                dataset.update(description=generated_description)
                logger.info(f"Updated dataset description for {dataset.id}")
            elif "[Auto-generated description" not in existing_desc and "[Enhanced with auto-generated" not in existing_desc:
                # Enhance existing description if it hasn't been auto-enhanced before
                enhanced_description = description_generator._enhance_existing_description(
                    existing_desc, dataset, cleaned_data, nlp_results, quality_results
                )
                if enhanced_description != existing_desc:
                    dataset.update(description=enhanced_description)
                    logger.info(f"Enhanced dataset description for {dataset.id}")

        return {
            'generated_description': generated_description,
            'description_length': len(generated_description) if generated_description else 0,
            'updated_dataset': True
        }

    def _step_complete_metadata(self, dataset, dataset_service, results):
        """Step 7: Complete and enhance metadata."""
        cleaned_data = results.get('step_1', {})
        quality_results = results.get('step_3', {})

        # Get user information for metadata completion
        user_info = {}
        if hasattr(dataset, 'user') and dataset.user:
            user_info = {
                'username': getattr(dataset.user, 'username', 'Unknown User'),
                'organization': getattr(dataset.user, 'organization', None)
            }

        # Complete metadata using the completion service
        metadata_enhancements = metadata_completion_service.complete_dataset_metadata(
            dataset, cleaned_data, quality_results, user_info
        )

        # Apply enhancements to dataset
        if metadata_enhancements:
            # Filter and convert complex objects for database storage
            filtered_enhancements = {}
            for key, value in metadata_enhancements.items():
                if value is not None and not callable(value):
                    # Convert datetime objects to ISO strings for storage
                    if hasattr(value, 'isoformat'):
                        filtered_enhancements[key] = value.isoformat()
                    # Convert lists and dicts to JSON strings for string fields
                    elif isinstance(value, (list, dict)):
                        import json
                        filtered_enhancements[key] = json.dumps(value)
                    # Convert other complex objects to strings
                    elif not isinstance(value, (str, int, float, bool)):
                        filtered_enhancements[key] = str(value)
                    else:
                        filtered_enhancements[key] = value

            if filtered_enhancements:
                try:
                    dataset.update(**filtered_enhancements)
                    logger.info(f"Applied {len(filtered_enhancements)} metadata enhancements to dataset {dataset.id}")
                except Exception as update_error:
                    logger.warning(f"Error updating dataset with enhancements: {update_error}")
                    # Try updating fields individually to identify problematic ones
                    for key, value in filtered_enhancements.items():
                        try:
                            dataset.update(**{key: value})
                        except Exception as field_error:
                            logger.warning(f"Could not update field {key}: {field_error}")

        return {
            'enhancements_applied': len(metadata_enhancements) if metadata_enhancements else 0,
            'enhanced_fields': list(metadata_enhancements.keys()) if metadata_enhancements else [],
            'metadata_completed': True
        }

    def _step_metadata_generation(self, dataset, dataset_service, results):
        """Step 6: Metadata generation."""
        cleaned_data = results.get('step_1', {})
        return metadata_service.generate_metadata(dataset, cleaned_data)
    
    def _step_save_results(self, dataset, dataset_service, results):
        """Step 7: Save results."""
        cleaned_data = results.get('step_1', {})
        nlp_results = results.get('step_2', {})
        quality_results = results.get('step_3', {})
        ai_compliance = results.get('step_4', {})
        
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
        """Step 9: Generate comprehensive visualizations."""
        cleaned_data = results.get('step_1', {})
        quality_results = results.get('step_3', {})
        ai_compliance = results.get('step_4', {})

        # Generate comprehensive visualizations using the new service
        visualizations = data_visualization_service.generate_comprehensive_visualizations(
            dataset, cleaned_data, quality_results, ai_compliance
        )

        # Save visualizations to dataset
        if visualizations and 'charts' in visualizations:
            dataset.update(visualizations=visualizations)
            logger.info(f"Generated {len(visualizations['charts'])} visualizations for dataset {dataset.id}")

        return visualizations
    
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
    
    def get_processing_status(self, dataset_id: str) -> Dict[str, Any]:
        """Get current processing status for a dataset."""
        # Check active processes first
        if dataset_id in self.active_processes:
            process_info = self.active_processes[dataset_id]
            return {
                'active': True,
                'status': process_info['status'],
                'progress': process_info.get('progress', 0),
                'message': process_info.get('message', ''),
                'started_at': process_info['started_at'].isoformat(),
                'method': 'persistent_processing'
            }
        
        # Check database for completed/failed processes
        queue_item = ProcessingQueue.objects(dataset=dataset_id).first()
        if queue_item:
            return {
                'active': False,
                'status': queue_item.status,
                'progress': queue_item.progress,
                'message': queue_item.message or '',
                'started_at': queue_item.started_at.isoformat() if queue_item.started_at else None,
                'completed_at': queue_item.completed_at.isoformat() if queue_item.completed_at else None,
                'error': queue_item.error,
                'method': 'persistent_processing'
            }
        
        return {
            'active': False,
            'status': 'not_started',
            'progress': 0,
            'message': 'Processing not started',
            'method': 'persistent_processing'
        }


# Global service instance
persistent_processing_service = PersistentProcessingService()
