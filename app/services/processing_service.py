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
from app.services.data_cleaning_service import data_cleaning_service
from app.services.ai_standards_service import ai_standards_service
from app.services.persistent_processing_service import persistent_processing_service
from app.services.description_generator import description_generator
from app.services.data_visualization_service import data_visualization_service
from app.services.metadata_completion_service import metadata_completion_service


class ProcessingService:
    """Service for automatic dataset processing with real-time progress tracking"""

    def __init__(self):
        self.active_processes = {}  # Track active processing tasks
        self.progress_callbacks = {}  # Store progress update callbacks

    def start_processing(self, dataset_id: str, upload_folder: str, progress_callback: Optional[Callable] = None) -> bool:
        """Start automatic processing for a dataset with enhanced persistence"""
        # Check if already processing to prevent duplication
        if dataset_id in self.active_processes:
            print(f"⚠️ Dataset {dataset_id} is already being processed")
            return False

        # Check if dataset is already being processed by persistent service
        if hasattr(persistent_processing_service, 'active_processes') and dataset_id in persistent_processing_service.active_processes:
            print(f"⚠️ Dataset {dataset_id} is already being processed by persistent service")
            return False

        try:
            # Try persistent processing first (survives application restarts)
            success = persistent_processing_service.start_persistent_processing(
                dataset_id, upload_folder, progress_callback
            )

            if success:
                print(f"✅ Started persistent processing for dataset {dataset_id}")
                return True
            else:
                print(f"⚠️ Persistent processing failed, falling back to standard processing")
                return self._start_standard_processing(dataset_id, upload_folder, progress_callback)

        except Exception as e:
            print(f"Error with persistent processing, trying background service: {e}")
            # Try background processing service as secondary fallback
            try:
                from app.services.background_processing_service import background_processing_service
                result = background_processing_service.start_background_processing(
                    dataset_id, upload_folder, priority=1
                )
                if result['success']:
                    print(f"✅ Started background processing: {result['method']}")
                    return True
                else:
                    print(f"⚠️ Background processing failed: {result.get('error', 'Unknown error')}")
            except Exception as bg_error:
                print(f"Background processing service unavailable: {bg_error}")

            # Final fallback to standard processing
            return self._start_standard_processing(dataset_id, upload_folder, progress_callback)

    def _start_standard_processing(self, dataset_id: str, upload_folder: str, progress_callback: Optional[Callable] = None) -> bool:
        """Fallback to standard processing if persistent processing fails"""
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

            # Start processing in enhanced background thread (non-daemon for persistence)
            thread = threading.Thread(
                target=self._process_dataset_background,
                args=(dataset_id, upload_folder),
                daemon=False  # Non-daemon thread survives longer
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
            print(f"Error starting standard processing for dataset {dataset_id}: {e}")
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

            # Step 1: Process dataset file (15% progress)
            self._update_progress(dataset_id, 5, 'processing', 'Processing dataset file...')
            processed_data = self._process_dataset_file(dataset, dataset_service)
            self._update_progress(dataset_id, 15, 'processing', 'Dataset file processed')

            # Step 2: Data Cleaning & Restructuring (30% progress)
            self._update_progress(dataset_id, 20, 'processing', 'Cleaning and restructuring data...')
            cleaned_data = self._clean_and_restructure_data(processed_data)
            self._update_progress(dataset_id, 30, 'processing', 'Data cleaning completed')

            # Step 3: NLP Analysis (45% progress)
            self._update_progress(dataset_id, 35, 'processing', 'Performing NLP analysis...')
            nlp_results = self._perform_nlp_analysis(dataset, cleaned_data)
            self._update_progress(dataset_id, 45, 'processing', 'NLP analysis completed')

            # Step 4: Quality Assessment (60% progress)
            self._update_progress(dataset_id, 50, 'processing', 'Assessing dataset quality...')
            quality_results = self._assess_quality(dataset, cleaned_data)
            self._update_progress(dataset_id, 60, 'processing', 'Quality assessment completed')

            # Step 5: AI Standards Compliance (65% progress)
            self._update_progress(dataset_id, 60, 'processing', 'Assessing AI standards compliance...')
            ai_compliance = self._assess_ai_standards(dataset, cleaned_data)
            self._update_progress(dataset_id, 65, 'processing', 'AI standards assessment completed')

            # Step 6: Generate Description (65% progress)
            self._update_progress(dataset_id, 60, 'processing', 'Generating intelligent description...')
            description_results = self._generate_description(dataset, cleaned_data, nlp_results, quality_results)
            self._update_progress(dataset_id, 65, 'processing', 'Description generation completed')

            # Step 7: Complete Metadata (75% progress)
            self._update_progress(dataset_id, 70, 'processing', 'Completing and enhancing metadata...')
            completion_results = self._complete_metadata(dataset, cleaned_data, quality_results)
            self._update_progress(dataset_id, 75, 'processing', 'Metadata completion finished')

            # Step 8: Generate Metadata (85% progress)
            self._update_progress(dataset_id, 80, 'processing', 'Generating comprehensive metadata...')
            metadata_results = self._generate_metadata(dataset, cleaned_data)
            self._update_progress(dataset_id, 85, 'processing', 'Metadata generation completed')

            # Step 9: Save Results (95% progress)
            self._update_progress(dataset_id, 90, 'processing', 'Saving processing results...')
            self._save_processing_results(dataset, cleaned_data, nlp_results, quality_results, metadata_results, ai_compliance)
            self._update_progress(dataset_id, 95, 'processing', 'Results saved')

            # Step 10: Generate Visualizations (100% progress)
            self._update_progress(dataset_id, 98, 'processing', 'Generating visualizations...')
            self._generate_visualizations(dataset, cleaned_data, quality_results)

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
            return dataset_service.process_dataset(file_info, dataset.format, process_full_dataset=True)
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

    def _clean_and_restructure_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and restructure data for better quality and standards compliance"""
        try:
            if not processed_data:
                return processed_data

            # Use the data cleaning service
            cleaned_data = data_cleaning_service.clean_dataset(processed_data)

            return cleaned_data
        except Exception as e:
            print(f"Error in data cleaning: {e}")
            return processed_data

    def _assess_ai_standards(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess AI standards compliance"""
        try:
            # Use the AI standards service
            ai_compliance = ai_standards_service.assess_ai_compliance(dataset, processed_data)

            return ai_compliance
        except Exception as e:
            print(f"Error in AI standards assessment: {e}")
            return {'error': str(e)}

    def _generate_description(self, dataset, processed_data: Dict[str, Any],
                            nlp_results: Dict[str, Any] = None,
                            quality_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate intelligent dataset description"""
        try:
            # Use the description generator service
            generated_description = description_generator.generate_description(
                dataset, processed_data, nlp_results, quality_results
            )

            # Handle structured description response
            if isinstance(generated_description, dict):
                # New structured format
                plain_text = generated_description.get('plain_text', '')
                existing_desc = dataset.description or ""

                # Update if no existing description or existing is very short
                if not existing_desc or len(existing_desc.strip()) < 100:
                    # Store both structured and plain text versions
                    import json
                    update_data = {
                        'description': plain_text,
                        'structured_description': json.dumps(generated_description)
                    }
                    dataset.update(**update_data)
                    print(f"✅ Updated dataset with structured description for {dataset.id}")
                elif "[Auto-generated description" not in existing_desc and "[Enhanced with auto-generated" not in existing_desc:
                    # Enhance existing description if it hasn't been auto-enhanced before
                    enhanced_description = description_generator._enhance_existing_description(
                        existing_desc, dataset, processed_data, nlp_results, quality_results
                    )
                    if enhanced_description != existing_desc:
                        dataset.update(description=enhanced_description)
                        print(f"✅ Enhanced dataset description for {dataset.id}")

                return {
                    'structured_description': generated_description,
                    'description_length': len(plain_text) if plain_text else 0,
                    'updated_dataset': True
                }
            else:
                # Legacy string format (backward compatibility)
                if generated_description and len(generated_description.strip()) > 50:
                    existing_desc = dataset.description or ""

                    if not existing_desc or len(existing_desc.strip()) < 100:
                        dataset.update(description=generated_description)
                        print(f"✅ Updated dataset description for {dataset.id}")

                return {
                    'generated_description': generated_description,
                    'description_length': len(generated_description) if generated_description else 0,
                    'updated_dataset': True
                }

        except Exception as e:
            print(f"Error in description generation: {e}")
            return {'error': str(e)}

    def _complete_metadata(self, dataset, processed_data: Dict[str, Any],
                          quality_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete and enhance dataset metadata automatically."""
        try:
            # Get user information for metadata completion
            user_info = {}
            if hasattr(dataset, 'user') and dataset.user:
                user_info = {
                    'username': getattr(dataset.user, 'username', 'Unknown User'),
                    'organization': getattr(dataset.user, 'organization', None)
                }

            # Complete metadata using the completion service
            metadata_enhancements = metadata_completion_service.complete_dataset_metadata(
                dataset, processed_data, quality_results, user_info
            )

            # Apply enhancements to dataset
            if metadata_enhancements:
                # Filter and convert complex objects for database storage
                filtered_enhancements = {}
                for key, value in metadata_enhancements.items():
                    if value is not None and not callable(value):
                        # Handle Schema.org metadata specially
                        if key == 'schema_org_metadata':
                            import json
                            filtered_enhancements['schema_org_json'] = json.dumps(value)
                            continue

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
                        print(f"✅ Applied {len(filtered_enhancements)} metadata enhancements to dataset {dataset.id}")
                    except Exception as update_error:
                        print(f"⚠️ Error updating dataset with enhancements: {update_error}")
                        # Try updating fields individually to identify problematic ones
                        for key, value in filtered_enhancements.items():
                            try:
                                dataset.update(**{key: value})
                            except Exception as field_error:
                                print(f"⚠️ Could not update field {key}: {field_error}")

            return {
                'enhancements_applied': len(metadata_enhancements) if metadata_enhancements else 0,
                'enhanced_fields': list(metadata_enhancements.keys()) if metadata_enhancements else [],
                'metadata_completed': True
            }

        except Exception as e:
            print(f"❌ Error completing metadata: {e}")
            return {'error': str(e)}

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

    def _save_processing_results(self, dataset, processed_data, nlp_results, quality_results, metadata_results, ai_compliance=None):
        """Save all processing results to database"""
        try:
            # Update dataset with processed information
            update_data = {}

            if processed_data and 'record_count' in processed_data:
                update_data['record_count'] = processed_data['record_count']

                # Add cleaning statistics if available
                if 'cleaning_stats' in processed_data:
                    update_data['cleaning_stats'] = processed_data['cleaning_stats']
                if 'transformation_log' in processed_data:
                    update_data['transformation_log'] = processed_data['transformation_log']

            # Add NLP-suggested tags if dataset doesn't have tags
            if not dataset.tags and nlp_results and 'suggested_tags' in nlp_results:
                suggested_tags = nlp_results['suggested_tags'][:5]  # Top 5 suggestions
                if suggested_tags:
                    # Convert list to comma-separated string for StringField
                    update_data['tags'] = ', '.join(suggested_tags)

            # Add AI compliance data
            if ai_compliance:
                update_data['ai_compliance'] = ai_compliance

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
        """Generate comprehensive visualizations for the dataset"""
        try:
            # Get AI compliance results if available
            ai_compliance = getattr(dataset, 'ai_compliance', None)

            # Generate comprehensive visualizations using the new service
            visualizations = data_visualization_service.generate_comprehensive_visualizations(
                dataset, processed_data, quality_results, ai_compliance
            )

            # Save visualizations to dataset as JSON string
            if visualizations and 'charts' in visualizations:
                import json
                dataset.update(visualizations=json.dumps(visualizations))
                print(f"✅ Generated {len(visualizations['charts'])} visualizations for dataset {dataset.id}")
            else:
                print(f"⚠️ No visualizations generated for dataset {dataset.id}")

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            # Fallback to basic visualization
            try:
                fallback_viz = data_visualization_service._generate_fallback_visualizations(dataset, processed_data)
                import json
                dataset.update(visualizations=json.dumps(fallback_viz))
                print(f"✅ Generated fallback visualizations for dataset {dataset.id}")
            except Exception as fallback_error:
                print(f"❌ Fallback visualization also failed: {fallback_error}")

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
