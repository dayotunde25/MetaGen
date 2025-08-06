#!/usr/bin/env python3
"""
Standalone Celery Application for MetadataHarvest Background Processing.

This file creates a Celery app that can be run independently of the Flask application
to avoid import issues when starting Celery workers.

Usage:
    celery -A celery_app.celery_app worker --loglevel=info --queue=dataset_processing
"""

import os
import sys
from celery import Celery

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure Celery
celery_app = Celery('aimetaharvest')

# Enhanced Celery configuration for Windows compatibility
celery_app.conf.update(
    # Broker and backend
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),

    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='UTC',
    enable_utc=True,

    # Windows-specific optimizations
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_pool='solo',  # Force solo pool for Windows

    # Connection settings for Windows
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,

    # Task execution settings
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    worker_disable_rate_limits=True,

    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
)

# Import tasks after Celery configuration
@celery_app.task(bind=True, name='celery_app.process_dataset_task')
def process_dataset_task(self, dataset_id: str, upload_folder: str = 'uploads'):
    """
    Celery task for processing datasets.
    
    Args:
        dataset_id: ID of the dataset to process
        upload_folder: Path to upload folder
        
    Returns:
        Processing result message
    """
    try:
        # Initialize Flask app context for database access
        from app import create_app
        app = create_app()

        with app.app_context():
            # Import here to avoid circular imports
            from app.services.background_processing_service import BackgroundProcessingService

            # Update task status
            self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting processing...'})

            # Create service instance and process
            service = BackgroundProcessingService()
            service._process_with_checkpoints(dataset_id, upload_folder)

            return {
                'status': 'completed',
                'message': f'Dataset {dataset_id} processed successfully',
                'dataset_id': dataset_id
            }
        
    except Exception as e:
        # Update task status with error
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'dataset_id': dataset_id}
        )
        raise e

@celery_app.task(name='celery_app.cleanup_task')
def cleanup_task():
    """
    Periodic cleanup task for maintenance.
    
    Returns:
        Cleanup result message
    """
    try:
        import os
        from datetime import datetime
        
        # Clean up old checkpoints
        checkpoint_dir = "app/cache/checkpoints"
        cleaned_files = 0
        
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                filepath = os.path.join(checkpoint_dir, filename)
                if os.path.isfile(filepath):
                    # Remove checkpoints older than 24 hours
                    if os.path.getmtime(filepath) < (datetime.now().timestamp() - 86400):
                        os.remove(filepath)
                        cleaned_files += 1
        
        return {
            'status': 'completed',
            'message': f'Cleanup completed. Removed {cleaned_files} old checkpoint files.',
            'cleaned_files': cleaned_files
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'message': f'Cleanup failed: {str(e)}',
            'error': str(e)
        }

@celery_app.task(name='celery_app.test_task')
def test_task(message: str = "Hello from Celery!"):
    """
    Simple test task to verify Celery is working.
    
    Args:
        message: Test message to return
        
    Returns:
        Test result
    """
    import time
    from datetime import datetime

    # Simulate some work
    time.sleep(2)

    return {
        'status': 'completed',
        'message': message,
        'timestamp': datetime.now().isoformat()
    }

@celery_app.task(bind=True, name='celery_app.long_running_task')
def long_running_task(self, duration: int = 10):
    """
    Long-running task for testing progress updates.
    
    Args:
        duration: Duration in seconds
        
    Returns:
        Task completion result
    """
    import time
    
    for i in range(duration):
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': i + 1, 'total': duration, 'status': f'Processing step {i + 1}/{duration}'}
        )
        time.sleep(1)
    
    return {
        'status': 'completed',
        'message': f'Long running task completed after {duration} seconds',
        'duration': duration
    }

# Health check task
@celery_app.task(name='celery_app.health_check')
def health_check():
    """
    Health check task to verify worker is responsive.

    Returns:
        Health status
    """
    from datetime import datetime

    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'worker_id': os.getpid(),
        'message': 'Celery worker is healthy and responsive'
    }

# Queue cleanup task
@celery_app.task(name='celery_app.cleanup_completed_queue_item')
def cleanup_completed_queue_item(dataset_id):
    """
    Remove a specific completed queue item.

    Args:
        dataset_id: ID of the dataset whose queue item should be cleaned up

    Returns:
        Cleanup result
    """
    try:
        from app.models.metadata import ProcessingQueue

        queue_item = ProcessingQueue.objects(dataset=dataset_id, status='completed').first()
        if queue_item:
            queue_item.delete()
            return {
                'status': 'completed',
                'message': f'Cleaned up completed queue item for dataset {dataset_id}',
                'dataset_id': dataset_id
            }
        else:
            return {
                'status': 'not_found',
                'message': f'No completed queue item found for dataset {dataset_id}',
                'dataset_id': dataset_id
            }
    except Exception as e:
        return {
            'status': 'failed',
            'message': f'Cleanup failed for dataset {dataset_id}: {str(e)}',
            'dataset_id': dataset_id,
            'error': str(e)
        }

if __name__ == '__main__':
    # Allow running this file directly for testing
    print("Celery app configured successfully!")
    print("Available tasks:")
    for task_name in celery_app.tasks:
        if not task_name.startswith('celery.'):
            print(f"  - {task_name}")
    
    print("\nTo start a worker, run:")
    print("celery -A celery_app.celery_app worker --loglevel=info --queue=dataset_processing")
    
    print("\nTo test the worker, run:")
    print("python -c \"from celery_app import test_task; result = test_task.delay('Test message'); print(result.get())\"")
