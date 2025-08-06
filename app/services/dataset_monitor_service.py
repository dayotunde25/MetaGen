"""
Dataset Monitor Service for automatic detection and processing of unprocessed datasets.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue
from app.services.background_processing_service import get_background_processing_service

logger = logging.getLogger(__name__)


class DatasetMonitorService:
    """
    Service for monitoring and automatically processing unprocessed datasets.
    
    Features:
    - Detects datasets that haven't been processed
    - Automatically queues them for background processing
    - Monitors failed processing attempts
    - Provides retry mechanisms for failed datasets
    """
    
    def __init__(self):
        """Initialize the dataset monitor service."""
        self.bg_service = get_background_processing_service()
        
    def scan_for_unprocessed_datasets(self) -> Dict[str, Any]:
        """
        Scan the database for datasets that need processing.
        
        Returns:
            Dictionary with scan results and actions taken
        """
        try:
            results = {
                'scanned': 0,
                'unprocessed_found': 0,
                'queued_for_processing': 0,
                'already_in_queue': 0,
                'failed_to_queue': 0,
                'errors': []
            }
            
            # Get all datasets
            datasets = Dataset.objects.all()
            results['scanned'] = len(datasets)
            
            logger.info(f"Scanning {results['scanned']} datasets for processing status...")
            
            for dataset in datasets:
                try:
                    # Check if dataset needs processing
                    if self._needs_processing(dataset):
                        results['unprocessed_found'] += 1
                        
                        # Check if already in queue
                        queue_item = ProcessingQueue.get_by_dataset(str(dataset.id))
                        if queue_item and queue_item.status in ['pending', 'processing']:
                            results['already_in_queue'] += 1
                            logger.debug(f"Dataset {dataset.id} already in queue with status: {queue_item.status}")
                            continue
                        
                        # Queue for processing
                        if self._queue_dataset_for_processing(dataset):
                            results['queued_for_processing'] += 1
                            logger.info(f"Queued dataset {dataset.id} ({dataset.title}) for processing")
                        else:
                            results['failed_to_queue'] += 1
                            logger.warning(f"Failed to queue dataset {dataset.id} for processing")
                            
                except Exception as e:
                    error_msg = f"Error processing dataset {dataset.id}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Scan complete: {results['queued_for_processing']} datasets queued for processing")
            return results
            
        except Exception as e:
            logger.error(f"Error during dataset scan: {e}")
            return {
                'scanned': 0,
                'error': str(e)
            }
    
    def _needs_processing(self, dataset: Dataset) -> bool:
        """
        Check if a dataset needs processing.
        
        Args:
            dataset: Dataset to check
            
        Returns:
            True if dataset needs processing
        """
        # Check if dataset has file or URL to process
        if not (dataset.file_path or dataset.source_url):
            return False
        
        # Check if dataset has been processed (has metadata)
        if hasattr(dataset, 'metadata') and dataset.metadata:
            # Check if metadata is complete
            metadata = dataset.metadata
            if (hasattr(metadata, 'quality_score') and metadata.quality_score and
                hasattr(metadata, 'fair_compliant') and metadata.fair_compliant is not None):
                return False  # Already processed
        
        # Check if dataset has quality assessment
        if hasattr(dataset, 'quality') and dataset.quality:
            quality = dataset.quality
            if (hasattr(quality, 'quality_score') and quality.quality_score and
                hasattr(quality, 'fair_compliant') and quality.fair_compliant is not None):
                return False  # Already processed
        
        # Check if dataset has processing record count
        if hasattr(dataset, 'record_count') and dataset.record_count and dataset.record_count > 0:
            # Might be processed, but check if it has complete metadata
            if not (hasattr(dataset, 'metadata') and dataset.metadata):
                return True  # Has data but no metadata
        
        return True  # Needs processing
    
    def _queue_dataset_for_processing(self, dataset: Dataset) -> bool:
        """
        Queue a dataset for background processing.
        
        Args:
            dataset: Dataset to queue
            
        Returns:
            True if successfully queued
        """
        try:
            # Start background processing
            result = self.bg_service.start_background_processing(
                str(dataset.id),
                "uploads",  # Default upload folder
                priority=2  # Lower priority for auto-detected datasets
            )
            
            if result['success']:
                # Create or update queue entry
                queue_item = ProcessingQueue.get_by_dataset(str(dataset.id))
                if queue_item:
                    queue_item.update(
                        status='processing',
                        started_at=datetime.now(),
                        message=f'Auto-queued for processing with {result.get("method", "unknown")}'
                    )
                else:
                    ProcessingQueue.create(
                        dataset=str(dataset.id),
                        status='processing',
                        priority=2,
                        started_at=datetime.now(),
                        message=f'Auto-queued for processing with {result.get("method", "unknown")}'
                    )
                
                return True
            else:
                # Create queue entry for manual processing
                ProcessingQueue.create(
                    dataset=str(dataset.id),
                    status='pending',
                    priority=2,
                    error=result.get('error', 'Failed to start processing')
                )
                return False
                
        except Exception as e:
            logger.error(f"Error queuing dataset {dataset.id}: {e}")
            return False
    
    def retry_failed_processing(self, max_retries: int = 3) -> Dict[str, Any]:
        """
        Retry processing for failed datasets.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with retry results
        """
        try:
            results = {
                'failed_found': 0,
                'retried': 0,
                'max_retries_reached': 0,
                'errors': []
            }
            
            # Get failed queue items
            failed_items = ProcessingQueue.objects(status='failed')
            results['failed_found'] = len(failed_items)
            
            logger.info(f"Found {results['failed_found']} failed processing items")
            
            for item in failed_items:
                try:
                    # Check retry count (if we track it)
                    retry_count = getattr(item, 'retry_count', 0)
                    
                    if retry_count >= max_retries:
                        results['max_retries_reached'] += 1
                        continue
                    
                    # Retry processing
                    dataset = Dataset.objects(id=item.dataset.id).first()
                    if dataset and self._queue_dataset_for_processing(dataset):
                        # Update retry count
                        item.update(
                            retry_count=retry_count + 1,
                            status='processing',
                            started_at=datetime.now(),
                            error=None,
                            message=f'Retry attempt {retry_count + 1}'
                        )
                        results['retried'] += 1
                        logger.info(f"Retrying processing for dataset {dataset.id} (attempt {retry_count + 1})")
                    
                except Exception as e:
                    error_msg = f"Error retrying dataset {item.dataset.id}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Retry complete: {results['retried']} datasets retried")
            return results
            
        except Exception as e:
            logger.error(f"Error during retry operation: {e}")
            return {
                'failed_found': 0,
                'error': str(e)
            }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get overall processing status.
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            total_datasets = Dataset.objects.count()
            
            # Count queue statuses
            pending = ProcessingQueue.objects(status='pending').count()
            processing = ProcessingQueue.objects(status='processing').count()
            completed = ProcessingQueue.objects(status='completed').count()
            failed = ProcessingQueue.objects(status='failed').count()
            
            # Estimate unprocessed datasets
            unprocessed = 0
            for dataset in Dataset.objects.all():
                if self._needs_processing(dataset):
                    unprocessed += 1
            
            return {
                'total_datasets': total_datasets,
                'unprocessed': unprocessed,
                'queue_status': {
                    'pending': pending,
                    'processing': processing,
                    'completed': completed,
                    'failed': failed
                },
                'celery_available': self.bg_service.celery_available,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global service instance
dataset_monitor_service = DatasetMonitorService()


def get_dataset_monitor_service() -> DatasetMonitorService:
    """Get the dataset monitor service instance."""
    return dataset_monitor_service
