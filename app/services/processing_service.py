"""
Processing Service for automatic dataset processing with progress tracking.
"""

import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue, MetadataQuality
from app.services.dataset_service import get_dataset_service
from app.services.nlp_service import nlp_service
from app.services.metadata_generator import metadata_service
from app.services.quality_assessment_service import quality_assessment_service


class ProcessingService:
    """Service for automatic dataset processing with real-time progress tracking"""

    def __init__(self):
        self.active_processes = {}  # Track active processing tasks
        self.progress_callbacks = {}  # Store progress update callbacks

    def start_processing(self, dataset_id: str, upload_folder: str, progress_callback: Optional[Callable] = None) -> bool:
        """Start automatic processing for a dataset"""
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

            # Store progress callback
            if progress_callback:
                self.progress_callbacks[dataset_id] = progress_callback

            # Start processing in background thread
            thread = threading.Thread(
                target=self._process_dataset_background,
                args=(dataset_id, upload_folder),
                daemon=True
            )
            thread.start()

            # Track the process
            self.active_processes[dataset_id] = {
                'thread': thread,
                'started_at': datetime.utcnow(),
                'status': 'starting'
            }

            return True

        except Exception as e:
            print(f"Error starting processing for dataset {dataset_id}: {e}")
            return False

    def _process_dataset_background(self, dataset_id: str, upload_folder: str):
        """Background processing method"""
        try:
            # Update status to processing
            self._update_progress(dataset_id, 0, 'processing', 'Starting dataset processing...')

            # Get dataset and services
            dataset = Dataset.find_by_id(dataset_id)
            dataset_service = get_dataset_service(upload_folder)

            if not dataset:
                self._update_progress(dataset_id, 0, 'failed', 'Dataset not found')
                return

            # Step 1: Process dataset file (20% progress)
            self._update_progress(dataset_id, 10, 'processing', 'Processing dataset file...')
            processed_data = self._process_dataset_file(dataset, dataset_service)
            self._update_progress(dataset_id, 20, 'processing', 'Dataset file processed')

            # Step 2: NLP Analysis (40% progress)
            self._update_progress(dataset_id, 25, 'processing', 'Performing NLP analysis...')
            nlp_results = self._perform_nlp_analysis(dataset, processed_data)
            self._update_progress(dataset_id, 40, 'processing', 'NLP analysis completed')

            # Step 3: Quality Assessment (60% progress)
            self._update_progress(dataset_id, 45, 'processing', 'Assessing dataset quality...')
            quality_results = self._assess_quality(dataset, processed_data)
            self._update_progress(dataset_id, 60, 'processing', 'Quality assessment completed')

            # Step 4: Metadata Generation (80% progress)
            self._update_progress(dataset_id, 65, 'processing', 'Generating comprehensive metadata...')
            metadata_results = self._generate_metadata(dataset, processed_data)
            self._update_progress(dataset_id, 80, 'processing', 'Metadata generation completed')

            # Step 5: Save Results (90% progress)
            self._update_progress(dataset_id, 85, 'processing', 'Saving processing results...')
            self._save_processing_results(dataset, processed_data, nlp_results, quality_results, metadata_results)
            self._update_progress(dataset_id, 90, 'processing', 'Results saved')

            # Step 6: Generate Visualizations (100% progress)
            self._update_progress(dataset_id, 95, 'processing', 'Generating visualizations...')
            self._generate_visualizations(dataset, processed_data, quality_results)

            # Complete processing
            self._update_progress(dataset_id, 100, 'completed', 'Processing completed successfully')

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"Error processing dataset {dataset_id}: {e}")
            self._update_progress(dataset_id, 0, 'failed', error_msg)

        finally:
            # Clean up
            if dataset_id in self.active_processes:
                del self.active_processes[dataset_id]
            if dataset_id in self.progress_callbacks:
                del self.progress_callbacks[dataset_id]

    def _update_progress(self, dataset_id: str, progress: int, status: str, message: str):
        """Update processing progress"""
        try:
            # Update database
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

            # Update active process status
            if dataset_id in self.active_processes:
                self.active_processes[dataset_id]['status'] = status
                self.active_processes[dataset_id]['progress'] = progress
                self.active_processes[dataset_id]['message'] = message

        except Exception as e:
            print(f"Error updating progress for dataset {dataset_id}: {e}")

    def _process_dataset_file(self, dataset, dataset_service) -> Dict[str, Any]:
        """Process the dataset file"""
        if dataset.file_path:
            file_info = {
                'path': dataset.file_path,
                'format': dataset.format,
                'size': dataset.size
            }
            return dataset_service.process_dataset(file_info, dataset.format)
        return {}

    def _perform_nlp_analysis(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform NLP analysis on dataset content"""
        # Extract text content
        content_text = self._extract_text_content(dataset, processed_data)

        if not content_text:
            return {}

        # Perform various NLP analyses
        results = {
            'keywords': nlp_service.extract_keywords(content_text, 20),
            'suggested_tags': nlp_service.suggest_tags(content_text, 10),
            'entities': nlp_service.extract_entities(content_text),
            'sentiment': nlp_service.analyze_sentiment(content_text),
            'summary': nlp_service.generate_summary(content_text, 3),
            'content_length': len(content_text),
            'word_count': len(content_text.split())
        }

        return results

    def _extract_text_content(self, dataset, processed_data: Dict[str, Any]) -> str:
        """Extract text content for analysis"""
        text_parts = []

        # Add dataset metadata
        if dataset.title:
            text_parts.append(dataset.title)
        if dataset.description:
            text_parts.append(dataset.description)
        if dataset.tags:
            if isinstance(dataset.tags, list):
                text_parts.extend(dataset.tags)
            else:
                text_parts.append(dataset.tags)

        # Add sample data content
        if processed_data and 'sample_data' in processed_data:
            sample_data = processed_data['sample_data']
            if isinstance(sample_data, list):
                for row in sample_data[:10]:  # First 10 rows
                    if isinstance(row, dict):
                        text_parts.extend([str(v) for v in row.values() if isinstance(v, str)])
                    elif isinstance(row, list):
                        text_parts.extend([str(v) for v in row if isinstance(v, str)])

        return ' '.join(text_parts)

    def _assess_quality(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess dataset quality"""
        try:
            # Use the quality assessment service (returns MetadataQuality object)
            metadata_quality = quality_assessment_service.assess_dataset_quality(dataset, processed_data)

            # Convert to dictionary for processing
            assessment = metadata_quality.to_dict()

            # Add additional quality metrics based on processed data
            if processed_data:
                data_structure_assessment = self._assess_data_structure(processed_data)
                assessment['data_structure_quality'] = data_structure_assessment

            return assessment
        except Exception as e:
            print(f"Error in quality assessment: {e}")
            return {'error': str(e)}

    def _assess_data_structure(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of data structure"""
        structure_quality = {
            'has_columns': 'columns' in processed_data,
            'has_sample_data': 'sample_data' in processed_data,
            'record_count': processed_data.get('record_count', 0),
            'format_valid': 'error' not in processed_data
        }

        # Calculate structure score
        score = 0
        if structure_quality['has_columns']: score += 25
        if structure_quality['has_sample_data']: score += 25
        if structure_quality['record_count'] > 0: score += 25
        if structure_quality['format_valid']: score += 25

        structure_quality['score'] = score
        return structure_quality

    def _generate_metadata(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        try:
            return metadata_service.generate_metadata(dataset, processed_data)
        except Exception as e:
            print(f"Error generating metadata: {e}")
            return {'error': str(e)}

    def _save_processing_results(self, dataset, processed_data, nlp_results, quality_results, metadata_results):
        """Save all processing results to database"""
        try:
            # Update dataset with processed information
            update_data = {}

            if processed_data and 'record_count' in processed_data:
                update_data['record_count'] = processed_data['record_count']

            # Add NLP-suggested tags if dataset doesn't have tags
            if not dataset.tags and nlp_results and 'suggested_tags' in nlp_results:
                suggested_tags = nlp_results['suggested_tags'][:5]  # Top 5 suggestions
                if suggested_tags:
                    # Convert list to comma-separated string for StringField
                    update_data['tags'] = ', '.join(suggested_tags)

            if update_data:
                dataset.update(**update_data)

            # Save or update metadata quality record
            metadata_quality = MetadataQuality.objects(dataset=str(dataset.id)).first()

            quality_data = {
                'dataset': dataset,  # Pass the dataset object, not string ID
                'quality_score': quality_results.get('quality_score', 0) if quality_results else 0,
                'completeness': quality_results.get('completeness', 0) if quality_results else 0,
                'consistency': quality_results.get('consistency', 0) if quality_results else 0,
                'accuracy': quality_results.get('accuracy', 0) if quality_results else 0,
                'fair_compliant': quality_results.get('fair_compliant', False) if quality_results else False,
                'schema_org_compliant': True,  # Assume compliant if we generated schema.org metadata
                'assessment_date': datetime.utcnow()
            }

            if metadata_quality:
                metadata_quality.update(**quality_data)
            else:
                MetadataQuality.create(**quality_data)

        except Exception as e:
            print(f"Error saving processing results: {e}")

    def _generate_visualizations(self, dataset, processed_data, quality_results):
        """Generate visualizations for the dataset"""
        # This is a placeholder for visualization generation
        # In a real implementation, you might generate charts, graphs, etc.
        try:
            visualizations = {
                'quality_chart': self._create_quality_chart_data(quality_results),
                'data_overview': self._create_data_overview(processed_data),
                'generated_at': datetime.utcnow().isoformat()
            }

            # Save visualization data (could be stored in database or files)
            # For now, we'll just log that visualizations were generated
            print(f"Generated visualizations for dataset {dataset.id}")

        except Exception as e:
            print(f"Error generating visualizations: {e}")

    def _create_quality_chart_data(self, quality_results) -> Dict[str, Any]:
        """Create data for quality visualization charts"""
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

    def _create_data_overview(self, processed_data) -> Dict[str, Any]:
        """Create data overview visualization"""
        if not processed_data:
            return {}

        return {
            'type': 'data_overview',
            'data': {
                'format': processed_data.get('format', 'unknown'),
                'record_count': processed_data.get('record_count', 0),
                'columns': len(processed_data.get('columns', [])),
                'has_sample': 'sample_data' in processed_data
            }
        }

    def get_processing_status(self, dataset_id: str) -> Dict[str, Any]:
        """Get current processing status for a dataset"""
        # Check active processes first
        if dataset_id in self.active_processes:
            process_info = self.active_processes[dataset_id]
            return {
                'active': True,
                'status': process_info['status'],
                'progress': process_info.get('progress', 0),
                'message': process_info.get('message', ''),
                'started_at': process_info['started_at'].isoformat()
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
                'error': queue_item.error
            }

        return {
            'active': False,
            'status': 'not_started',
            'progress': 0,
            'message': 'Processing not started'
        }


# Global service instance
processing_service = ProcessingService()
