"""
Routes for dataset management.
"""

import os
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, abort, jsonify, send_file
from flask_login import login_required, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.exceptions import NotFound
from werkzeug.utils import secure_filename

from app.forms import DatasetForm, SearchForm
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.services.dataset_service import get_dataset_service
from app.services.quality_assessment_service import quality_assessment_service
from app.services.processing_service import processing_service
from app.utils.file_utils import validate_zip_contents, format_file_size
import json
import uuid

# Create blueprint
datasets_bp = Blueprint('datasets', __name__)


@datasets_bp.route('/datasets')
def list():
    """List all datasets."""
    search_form = SearchForm(request.args)

    # Get search parameters
    query = request.args.get('query', '')
    category = request.args.get('category', '')
    data_type = request.args.get('data_type', '')

    # Get datasets based on search parameters
    if query or category or data_type:
        datasets_list = Dataset.search(query, category, data_type)
    else:
        datasets_list = Dataset.get_all()

    return render_template('datasets/list.html', datasets=datasets_list, form=search_form)


@datasets_bp.route('/datasets/create', methods=['GET', 'POST'])
@login_required
def create():
    """Create a new dataset."""
    form = DatasetForm()

    if form.validate_on_submit():
        # Handle file upload or URL
        file_path = None
        file_format = None
        file_size = None

        if form.dataset_file.data:
            # Handle file upload
            file = form.dataset_file.data
            if file and file.filename:
                # Create upload directory if it doesn't exist
                upload_dir = os.path.join(current_app.root_path, 'uploads')
                os.makedirs(upload_dir, exist_ok=True)

                # Save file with secure filename
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)

                # Get file info
                file_format = os.path.splitext(filename)[1][1:].lower()
                file_size = os.path.getsize(file_path)

                # Handle ZIP files specially
                if file_format == 'zip':
                    return handle_zip_upload(file_path, form, current_user)

        # Process tags
        tags = form.tags.data

        # Create dataset with enhanced automatic field detection
        dataset = Dataset.create(
            title=form.title.data or None,  # Allow None for auto-generation
            description=form.description.data or None,  # Allow None for auto-generation
            source=form.source.data or None,  # Allow None for auto-detection
            source_url=form.source_url.data,
            data_type=form.data_type.data or None,  # Allow None for auto-detection
            category=form.category.data or None,  # Allow None for auto-detection
            tags=tags or None,  # Allow None for auto-generation
            license=getattr(form, 'license', None) and form.license.data or None,  # Allow None for auto-detection
            author=getattr(form, 'author', None) and form.author.data or current_user.username,  # Default to uploader
            user=current_user,
            file_path=file_path,
            format=file_format,
            size=f"{file_size / 1024 / 1024:.2f} MB" if file_size else None,
            auto_generated_title=not bool(form.title.data)  # Track if title was auto-generated
        )

        # Start automatic processing if we have a file or URL
        if file_path or form.source_url.data:
            try:
                # Use Celery/Redis background processing as primary method
                from app.services.background_processing_service import get_background_processing_service
                bg_service = get_background_processing_service()

                # Start background processing with Celery first, fallback to threading
                result = bg_service.start_background_processing(
                    str(dataset.id),
                    current_app.config['UPLOAD_FOLDER'],
                    priority=1
                )

                if result['success']:
                    method = result.get('method', 'unknown')
                    if method == 'celery':
                        flash('Dataset created successfully! Processing started with Celery background worker.', 'success')
                    else:
                        flash('Dataset created successfully! Processing started with enhanced threading.', 'info')

                    # Also create a queue entry for tracking
                    ProcessingQueue.create(
                        dataset=str(dataset.id),
                        status='processing',
                        priority=1,
                        started_at=datetime.now(),
                        message=f'Processing started with {method}'
                    )
                else:
                    # Fallback to manual queue creation
                    ProcessingQueue.create(
                        dataset=str(dataset.id),
                        status='pending',
                        priority=1
                    )
                    flash('Dataset created successfully and added to processing queue!', 'warning')

            except Exception as e:
                print(f"Warning: Could not start automatic processing: {e}")
                # Create queue entry for manual processing
                ProcessingQueue.create(
                    dataset=str(dataset.id),
                    status='pending',
                    priority=1,
                    error=str(e)
                )
                flash('Dataset created successfully! Processing will be available manually.', 'warning')
        else:
            flash('Dataset created successfully!', 'success')

        return redirect(url_for('datasets.view', dataset_id=dataset.id))

    return render_template('datasets/form.html', form=form)


def handle_zip_upload(file_path, form, user):
    """Handle ZIP file upload and create collection of datasets"""
    try:
        # Validate ZIP file
        is_valid, error_msg, zip_info = validate_zip_contents(file_path)
        if not is_valid:
            flash(f'Invalid ZIP file: {error_msg}', 'danger')
            return redirect(url_for('datasets.create'))

        # Generate collection ID and title
        collection_id = str(uuid.uuid4())
        collection_title = form.title.data if form.title.data else f"Dataset Collection ({zip_info['dataset_files']} files)"

        # Create parent collection dataset
        parent_dataset = Dataset.create(
            title=collection_title,
            description=form.description.data or f"Collection containing {zip_info['dataset_files']} datasets from ZIP archive",
            source=form.source.data or "ZIP Upload",
            source_url=form.source_url.data,
            data_type="collection",
            category=form.category.data or "mixed",
            tags=form.tags.data,
            user=user,
            file_path=file_path,
            format="zip",
            size=format_file_size(zip_info['total_size']),
            is_collection=True,
            collection_id=collection_id,
            auto_generated_title=not bool(form.title.data),
            collection_files=json.dumps(zip_info['dataset_files_list'])
        )

        # Start processing the ZIP file
        try:
            from app.services.background_processing_service import get_background_processing_service
            bg_service = get_background_processing_service()

            result = bg_service.start_background_processing(
                str(parent_dataset.id),
                current_app.config['UPLOAD_FOLDER'],
                priority=1
            )

            if result['success']:
                method = result.get('method', 'unknown')
                flash(f'ZIP collection created successfully! Processing {zip_info["dataset_files"]} datasets with {method}.', 'success')

                ProcessingQueue.create(
                    dataset=str(parent_dataset.id),
                    status='processing',
                    priority=1,
                    started_at=datetime.now(),
                    message=f'ZIP collection processing started with {method}'
                )
            else:
                ProcessingQueue.create(
                    dataset=str(parent_dataset.id),
                    status='pending',
                    priority=1
                )
                flash('ZIP collection created and queued for processing!', 'warning')

        except Exception as e:
            print(f"Warning: Could not start automatic processing: {e}")
            ProcessingQueue.create(
                dataset=str(parent_dataset.id),
                status='pending',
                priority=1,
                error=str(e)
            )
            flash('ZIP collection created! Processing will be available manually.', 'warning')

        return redirect(url_for('datasets.view', dataset_id=parent_dataset.id))

    except Exception as e:
        flash(f'Error processing ZIP file: {str(e)}', 'danger')
        return redirect(url_for('datasets.create'))


@datasets_bp.route('/datasets/<dataset_id>')
def view(dataset_id):
    """View a dataset."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Get metadata quality if available
    metadata_quality = MetadataQuality.get_by_dataset(dataset_id)

    # Get processing status if available
    processing = ProcessingQueue.get_by_dataset(dataset_id)

    # Get feedback summary for the dataset
    from app.models.dataset_feedback import DatasetFeedback
    feedback_summary = DatasetFeedback.get_dataset_feedback_summary(dataset_id)

    return render_template('datasets/view.html',
                           dataset=dataset,
                           quality=metadata_quality,
                           processing=processing,
                           feedback_summary=feedback_summary)


@datasets_bp.route('/datasets/<dataset_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(dataset_id):
    """Edit a dataset."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Authorization check
    if dataset.user and str(dataset.user.id) != str(current_user.id) and not current_user.is_admin:
        flash('You are not authorized to edit this dataset.', 'danger')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))

    form = DatasetForm(obj=dataset)

    if form.validate_on_submit():
        # Update dataset
        dataset.update(
            title=form.title.data,
            description=form.description.data,
            source=form.source.data,
            source_url=form.source_url.data,
            data_type=form.data_type.data,
            category=form.category.data,
            tags=form.tags.data
        )

        flash('Dataset updated successfully.', 'success')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))

    return render_template('datasets/form.html', form=form, dataset=dataset)


@datasets_bp.route('/datasets/<dataset_id>/delete', methods=['POST'])
@login_required
def delete(dataset_id):
    """Delete a dataset."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Authorization check
    if dataset.user and str(dataset.user.id) != str(current_user.id) and not current_user.is_admin:
        flash('You are not authorized to delete this dataset.', 'danger')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))

    dataset.delete()
    flash('Dataset deleted successfully.', 'success')
    return redirect(url_for('datasets.list'))


@datasets_bp.route('/datasets/<dataset_id>/metadata')
def metadata(dataset_id):
    """View dataset metadata."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Get metadata quality if available
    metadata_quality = MetadataQuality.get_by_dataset(dataset_id)

    # Get schema.org metadata if available
    schema_org = None
    if dataset.schema_org:
        import json
        try:
            schema_org = json.loads(dataset.schema_org)
        except json.JSONDecodeError:
            schema_org = None

    # Get feedback summary for the dataset
    from app.models.dataset_feedback import DatasetFeedback
    feedback_summary = DatasetFeedback.get_dataset_feedback_summary(dataset_id)

    return render_template('datasets/metadata.html',
                           dataset=dataset,
                           metadata=metadata_quality,
                           schema_org=schema_org,
                           feedback_summary=feedback_summary)


@datasets_bp.route('/datasets/<dataset_id>/quality')
def quality(dataset_id):
    """View dataset quality assessment."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Get metadata quality if available
    metadata_quality = MetadataQuality.get_by_dataset(dataset_id)

    return render_template('datasets/quality.html',
                           dataset=dataset,
                           quality=metadata_quality)


@datasets_bp.route('/datasets/<dataset_id>/assess-quality')
@login_required
def assess_quality(dataset_id):
    """Trigger quality assessment for a dataset."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Authorization check
    if dataset.user and str(dataset.user.id) != str(current_user.id) and not current_user.is_admin:
        flash('You are not authorized to assess this dataset.', 'danger')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))

    try:
        # Get dataset service
        dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])

        # Process dataset to get data
        processed_data = dataset_service.process_dataset({
            'path': dataset.source_url,
            'format': dataset.format,
            'size': None  # Size will be determined during processing
        })

        if not processed_data:
            flash('Failed to process dataset data.', 'danger')
            return redirect(url_for('datasets.quality', dataset_id=dataset_id))

        # Assess quality
        metadata_quality = quality_assessment_service.assess_dataset_quality(dataset, processed_data)

        # Assess FAIR compliance
        quality_assessment_service.assess_fair_compliance(dataset, metadata_quality)

        # Assess Schema.org compliance
        quality_assessment_service.assess_schema_org_compliance(dataset, metadata_quality)

        # Mark dataset as processed
        dataset.update(processed=True)

        flash('Quality assessment completed successfully.', 'success')

    except Exception as e:
        current_app.logger.error(f"Error assessing quality for dataset {dataset_id}: {str(e)}")
        flash(f'Error assessing quality: {str(e)}', 'danger')

    return redirect(url_for('datasets.quality', dataset_id=dataset_id))


@datasets_bp.route('/datasets/<dataset_id>/visualizations')
def visualizations(dataset_id):
    """View dataset visualizations."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Get visualizations from dataset
    visualizations_json = getattr(dataset, 'visualizations', None)

    if not visualizations_json:
        flash('No visualizations available for this dataset. Please run processing first.', 'info')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))

    # Parse JSON data
    try:
        import json
        visualizations_data = json.loads(visualizations_json)
    except (json.JSONDecodeError, TypeError):
        flash('Error loading visualizations. Please regenerate them.', 'error')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))

    return render_template('datasets/visualizations.html',
                           dataset=dataset,
                           visualizations=visualizations_data)


@datasets_bp.route('/api/datasets/<dataset_id>/visualizations')
def api_visualizations(dataset_id):
    """API endpoint to get dataset visualizations as JSON."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    # Get visualizations from dataset
    visualizations_json = getattr(dataset, 'visualizations', None)

    if not visualizations_json:
        return jsonify({'error': 'No visualizations available'}), 404

    # Parse JSON data
    try:
        import json
        visualizations_data = json.loads(visualizations_json)
        return jsonify(visualizations_data)
    except (json.JSONDecodeError, TypeError):
        return jsonify({'error': 'Error parsing visualization data'}), 500



@datasets_bp.route('/datasets/my-datasets')
@login_required
def my_datasets():
    """View user's datasets."""
    datasets_list = Dataset.get_by_user(str(current_user.id))
    return render_template('datasets/my_datasets.html', datasets=datasets_list)


@datasets_bp.route('/api/datasets/<dataset_id>/progress')
def get_processing_progress(dataset_id):
    """API endpoint to get real-time processing progress"""
    try:
        # Get processing status from the processing service
        status = processing_service.get_processing_status(dataset_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'active': False,
            'status': 'error',
            'progress': 0,
            'message': 'Failed to get processing status'
        }), 500


@datasets_bp.route('/api/datasets/scan-unprocessed', methods=['POST'])
@login_required
def scan_unprocessed_datasets():
    """API endpoint to scan for and queue unprocessed datasets"""
    try:
        from app.services.dataset_monitor_service import get_dataset_monitor_service

        monitor_service = get_dataset_monitor_service()

        # Check if user is admin (optional - you can remove this check if all users should be able to scan)
        # if not current_user.is_admin:
        #     return jsonify({'error': 'Admin access required'}), 403

        # Perform scan
        scan_results = monitor_service.scan_for_unprocessed_datasets()

        # Also get current processing status
        status = monitor_service.get_processing_status()

        return jsonify({
            'success': True,
            'scan_results': scan_results,
            'processing_status': status,
            'message': f'Scan complete: {scan_results.get("queued_for_processing", 0)} datasets queued for processing'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to scan for unprocessed datasets'
        }), 500


@datasets_bp.route('/api/datasets/processing-status')
def get_processing_status():
    """API endpoint to get overall processing status"""
    try:
        from app.services.dataset_monitor_service import get_dataset_monitor_service

        monitor_service = get_dataset_monitor_service()
        status = monitor_service.get_processing_status()

        return jsonify({
            'success': True,
            'status': status
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get processing status'
        }), 500


@datasets_bp.route('/datasets/<dataset_id>/start-processing', methods=['POST'])
@login_required
def start_processing(dataset_id):
    """Manually start processing for a dataset"""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Authorization check
    if dataset.user and str(dataset.user.id) != str(current_user.id) and not current_user.is_admin:
        flash('You are not authorized to process this dataset.', 'danger')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))

    try:
        # Use Celery/Redis background processing as primary method
        from app.services.background_processing_service import get_background_processing_service
        bg_service = get_background_processing_service()

        # Start background processing with Celery first, fallback to threading
        result = bg_service.start_background_processing(
            dataset_id,
            current_app.config['UPLOAD_FOLDER'],
            priority=1
        )

        if result['success']:
            method = result.get('method', 'unknown')
            if method == 'celery':
                flash('Processing started successfully with Celery background worker!', 'success')
            else:
                flash('Processing started successfully with enhanced threading!', 'info')

            # Update or create queue entry
            queue_item = ProcessingQueue.get_by_dataset(dataset_id)
            if queue_item:
                queue_item.update(
                    status='processing',
                    started_at=datetime.now(),
                    message=f'Processing started with {method}'
                )
            else:
                ProcessingQueue.create(
                    dataset=dataset_id,
                    status='processing',
                    priority=1,
                    started_at=datetime.now(),
                    message=f'Processing started with {method}'
                )
        else:
            flash(f'Processing failed to start: {result.get("error", "Unknown error")}', 'warning')

    except Exception as e:
        flash(f'Error starting processing: {str(e)}', 'danger')

    return redirect(url_for('datasets.view', dataset_id=dataset_id))


@datasets_bp.route('/api/datasets/<dataset_id>/fair-metadata')
def api_fair_metadata(dataset_id):
    """API endpoint to get FAIR-compliant metadata in various formats."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    format_type = request.args.get('format', 'json')

    try:
        if format_type == 'dublin-core':
            if dataset.dublin_core:
                import json
                metadata = json.loads(dataset.dublin_core)
                return jsonify(metadata)
            else:
                return jsonify({'error': 'Dublin Core metadata not available'}), 404

        elif format_type == 'dcat':
            if dataset.dcat_metadata:
                import json
                metadata = json.loads(dataset.dcat_metadata)
                return jsonify(metadata)
            else:
                return jsonify({'error': 'DCAT metadata not available'}), 404

        elif format_type == 'json-ld':
            if dataset.json_ld:
                import json
                metadata = json.loads(dataset.json_ld)
                response = jsonify(metadata)
                response.headers['Content-Type'] = 'application/ld+json'
                return response
            else:
                return jsonify({'error': 'JSON-LD metadata not available'}), 404

        elif format_type == 'fair':
            if dataset.fair_metadata:
                import json
                metadata = json.loads(dataset.fair_metadata)
                return jsonify(metadata)
            else:
                return jsonify({'error': 'FAIR metadata not available'}), 404

        else:  # Default JSON format with all available metadata
            metadata = {
                'dataset_id': str(dataset.id),
                'title': dataset.title,
                'description': dataset.description,
                'persistent_identifier': dataset.persistent_identifier,
                'fair_compliant': dataset.fair_compliant,
                'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
                'updated_at': dataset.updated_at.isoformat() if dataset.updated_at else None
            }

            # Add available metadata formats
            if dataset.dublin_core:
                import json
                metadata['dublin_core'] = json.loads(dataset.dublin_core)
            if dataset.dcat_metadata:
                import json
                metadata['dcat'] = json.loads(dataset.dcat_metadata)
            if dataset.json_ld:
                import json
                metadata['json_ld'] = json.loads(dataset.json_ld)
            if dataset.fair_metadata:
                import json
                metadata['fair_metadata'] = json.loads(dataset.fair_metadata)

            return jsonify(metadata)

    except Exception as e:
        return jsonify({'error': f'Error retrieving metadata: {str(e)}'}), 500


@datasets_bp.route('/datasets/<dataset_id>/fair-compliance')
def fair_compliance(dataset_id):
    """View detailed FAIR compliance information."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    # Parse FAIR metadata if available
    fair_data = None
    if dataset.fair_metadata:
        try:
            import json
            fair_data = json.loads(dataset.fair_metadata)
        except:
            fair_data = None

    return render_template('datasets/fair_compliance.html',
                         dataset=dataset,
                         fair_data=fair_data)


@datasets_bp.route('/datasets/<dataset_id>/debug')
def debug_dataset(dataset_id):
    """Debug route to check dataset fields."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    debug_info = {
        'id': str(dataset.id),
        'title': dataset.title,
        'field_names': getattr(dataset, 'field_names', 'NOT SET'),
        'data_types': getattr(dataset, 'data_types', 'NOT SET'),
        'use_cases': getattr(dataset, 'use_cases', 'NOT SET'),
        'tags': getattr(dataset, 'tags', 'NOT SET'),
        'keywords': getattr(dataset, 'keywords', 'NOT SET'),
        'fair_score': getattr(dataset, 'fair_score', 'NOT SET'),
        'record_count': getattr(dataset, 'record_count', 'NOT SET'),
        'field_count': getattr(dataset, 'field_count', 'NOT SET'),
        'status': dataset.status,
        'all_fields': [field for field in dir(dataset) if not field.startswith('_')]
    }

    return jsonify(debug_info)


@datasets_bp.route('/datasets/<dataset_id>/force-process', methods=['POST'])
def force_process_dataset(dataset_id):
    """Force processing of a dataset for testing."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    try:
        # Import and trigger processing
        from celery_app import process_dataset_task

        # Queue the processing task
        task = process_dataset_task.delay(dataset_id)

        return jsonify({
            'status': 'success',
            'message': 'Processing task queued',
            'task_id': task.id,
            'dataset_id': dataset_id
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error queuing processing task: {str(e)}'
        }), 500


@datasets_bp.route('/datasets/<dataset_id>/download')
def download_dataset(dataset_id):
    """Download dataset as a zip file with metadata and reports."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    try:
        from app.services.download_service import get_download_service
        download_service = get_download_service()

        if dataset.is_collection:
            # Download as collection
            zip_path = download_service.create_collection_zip(dataset.collection_id)
        else:
            # Download as single dataset
            zip_path = download_service.create_dataset_zip(dataset_id)

        if not zip_path or not os.path.exists(zip_path):
            flash('Failed to create download package.', 'danger')
            return redirect(url_for('datasets.view', dataset_id=dataset_id))

        # Determine download filename
        if dataset.is_collection:
            download_filename = f"{dataset.title.replace(' ', '_')}_collection.zip"
        else:
            download_filename = f"{dataset.title.replace(' ', '_')}_dataset.zip"

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/zip'
        )

    except Exception as e:
        current_app.logger.error(f"Error creating download: {str(e)}")
        flash(f'Error creating download: {str(e)}', 'danger')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))


@datasets_bp.route('/collections/<collection_id>/download')
def download_collection(collection_id):
    """Download entire collection as a zip file."""
    try:
        from app.services.download_service import get_download_service
        download_service = get_download_service()

        # Find the parent collection dataset
        parent_dataset = Dataset.find_by_collection_id(collection_id)
        if not parent_dataset:
            abort(404)

        zip_path = download_service.create_collection_zip(collection_id)

        if not zip_path or not os.path.exists(zip_path):
            flash('Failed to create collection download package.', 'danger')
            return redirect(url_for('datasets.view', dataset_id=str(parent_dataset.id)))

        download_filename = f"{parent_dataset.title.replace(' ', '_')}_collection.zip"

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/zip'
        )

    except Exception as e:
        current_app.logger.error(f"Error creating collection download: {str(e)}")
        flash(f'Error creating collection download: {str(e)}', 'danger')
        return redirect(url_for('datasets.list'))


@datasets_bp.route('/datasets/<dataset_id>/export/markdown')
def export_metadata_markdown(dataset_id):
    """Export dataset metadata as Markdown file."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    try:
        from app.services.metadata_export_service import get_metadata_export_service
        export_service = get_metadata_export_service()

        file_path = export_service.export_metadata_markdown(dataset_id)

        if not file_path or not os.path.exists(file_path):
            flash('Failed to generate Markdown export.', 'danger')
            return redirect(url_for('datasets.metadata', dataset_id=dataset_id))

        download_filename = f"{dataset.title.replace(' ', '_')}_metadata.md"

        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='text/markdown'
        )

    except Exception as e:
        current_app.logger.error(f"Error exporting metadata to Markdown: {str(e)}")
        flash(f'Error exporting metadata: {str(e)}', 'danger')
        return redirect(url_for('datasets.metadata', dataset_id=dataset_id))


@datasets_bp.route('/datasets/<dataset_id>/export/pdf')
def export_metadata_pdf(dataset_id):
    """Export dataset metadata as PDF file."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    try:
        from app.services.metadata_export_service import get_metadata_export_service
        export_service = get_metadata_export_service()

        file_path = export_service.export_metadata_pdf(dataset_id)

        if not file_path or not os.path.exists(file_path):
            flash('Failed to generate PDF export. Make sure ReportLab is installed.', 'danger')
            return redirect(url_for('datasets.metadata', dataset_id=dataset_id))

        download_filename = f"{dataset.title.replace(' ', '_')}_metadata.pdf"

        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        current_app.logger.error(f"Error exporting metadata to PDF: {str(e)}")
        flash(f'Error exporting metadata: {str(e)}', 'danger')
        return redirect(url_for('datasets.metadata', dataset_id=dataset_id))


@datasets_bp.route('/datasets/<dataset_id>/export/json')
def export_metadata_json(dataset_id):
    """Export dataset metadata as JSON file."""
    dataset = Dataset.find_by_id(dataset_id)

    if not dataset:
        abort(404)

    try:
        from app.services.metadata_export_service import get_metadata_export_service
        export_service = get_metadata_export_service()

        file_path = export_service.export_metadata_json(dataset_id)

        if not file_path or not os.path.exists(file_path):
            flash('Failed to generate JSON export.', 'danger')
            return redirect(url_for('datasets.metadata', dataset_id=dataset_id))

        download_filename = f"{dataset.title.replace(' ', '_')}_metadata.json"

        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/json'
        )

    except Exception as e:
        current_app.logger.error(f"Error exporting metadata to JSON: {str(e)}")
        flash(f'Error exporting metadata: {str(e)}', 'danger')
        return redirect(url_for('datasets.metadata', dataset_id=dataset_id))


@datasets_bp.route('/api/cleanup-completed-queue', methods=['POST'])
def cleanup_completed_queue():
    """Manually clean up all completed queue items."""
    try:
        from app.models.metadata import ProcessingQueue

        # Find all completed queue items
        completed_items = ProcessingQueue.objects(status='completed')
        count = completed_items.count()

        # Delete them
        completed_items.delete()

        return jsonify({
            'status': 'success',
            'message': f'Cleaned up {count} completed queue items',
            'cleaned_count': count
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error cleaning up queue: {str(e)}'
        }), 500


@datasets_bp.route('/api/queue-status')
def queue_status():
    """Check current processing queue status."""
    try:
        from app.models.metadata import ProcessingQueue
        from app.models.dataset import Dataset

        # Get all queue items
        all_items = ProcessingQueue.objects()

        # Group by status
        status_counts = {}
        items_by_status = {}

        for item in all_items:
            status = item.status
            if status not in status_counts:
                status_counts[status] = 0
                items_by_status[status] = []

            # Get dataset info safely - handle DBRef issues
            try:
                # Try to get the dataset ID first without dereferencing
                dataset_id = str(item.dataset.id) if hasattr(item.dataset, 'id') else str(item.dataset)

                # Now try to find the dataset
                dataset = Dataset.find_by_id(dataset_id)
                dataset_title = dataset.title if dataset else 'Dataset Not Found'
                dataset_status = dataset.status if dataset else 'Not Found'
            except Exception as e:
                # Handle cases where dataset reference is broken
                try:
                    dataset_id = str(item.dataset)
                except:
                    dataset_id = 'Unknown'
                dataset_title = 'Dataset Not Found'
                dataset_status = 'Not Found'

            status_counts[status] += 1
            items_by_status[status].append({
                'dataset_id': dataset_id,
                'dataset_title': dataset_title,
                'dataset_status': dataset_status,
                'queue_status': item.status,
                'progress': item.progress,
                'message': item.message,
                'created_at': item.created_at.isoformat() if item.created_at else None,
                'updated_at': item.updated_at.isoformat() if item.updated_at else None
            })

        return jsonify({
            'status': 'success',
            'total_items': all_items.count(),
            'status_counts': status_counts,
            'items_by_status': items_by_status
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': f'Error checking queue status: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@datasets_bp.route('/api/fix-stuck-queue', methods=['POST'])
def fix_stuck_queue():
    """Fix stuck queue items by updating their status based on dataset status."""
    try:
        from app.models.metadata import ProcessingQueue
        from app.models.dataset import Dataset
        from datetime import datetime, timedelta

        fixed_items = []

        # Find queue items that might be stuck
        all_items = ProcessingQueue.objects()

        for item in all_items:
            try:
                # Handle DBRef issues when getting dataset ID
                try:
                    dataset_id = str(item.dataset.id) if hasattr(item.dataset, 'id') else str(item.dataset)
                except:
                    # If we can't even get the dataset ID, delete the queue item
                    item.delete()
                    fixed_items.append({
                        'dataset_id': 'unknown',
                        'action': 'deleted',
                        'reason': 'corrupted dataset reference'
                    })
                    continue

                dataset = Dataset.find_by_id(dataset_id)
                if not dataset:
                    # Dataset doesn't exist, remove queue item
                    item.delete()
                    fixed_items.append({
                        'dataset_id': dataset_id,
                        'action': 'deleted',
                        'reason': 'dataset not found'
                    })
                    continue

                # If dataset is completed but queue item is not
                if dataset.status == 'completed' and item.status != 'completed':
                    item.update(
                        status='completed',
                        progress=100,
                        message='Processing completed successfully',
                        updated_at=datetime.now()
                    )
                    fixed_items.append({
                        'dataset_id': dataset_id,
                        'action': 'updated to completed',
                        'reason': 'dataset was already completed'
                    })

                # If queue item is stuck in processing for more than 1 hour
                elif item.status == 'processing' and item.updated_at:
                    time_diff = datetime.now() - item.updated_at
                    if time_diff > timedelta(hours=1):
                        # Reset to pending for retry
                        item.update(
                            status='pending',
                            progress=0,
                            message='Reset from stuck processing state',
                            updated_at=datetime.now()
                        )
                        fixed_items.append({
                            'dataset_id': dataset_id,
                            'action': 'reset to pending',
                            'reason': 'stuck in processing for over 1 hour'
                        })
            except Exception as item_error:
                # Skip problematic items but continue processing others
                try:
                    error_dataset_id = str(item.dataset.id) if hasattr(item.dataset, 'id') else str(item.dataset)
                except:
                    error_dataset_id = 'unknown'

                fixed_items.append({
                    'dataset_id': error_dataset_id,
                    'action': 'skipped',
                    'reason': f'error processing item: {str(item_error)}'
                })
                continue

        # Check if this is an API request or form submission
        if request.is_json or request.headers.get('Content-Type') == 'application/json':
            return jsonify({
                'status': 'success',
                'message': f'Fixed {len(fixed_items)} queue items',
                'fixed_items': fixed_items
            })
        else:
            # Form submission - redirect with flash message
            if fixed_items:
                flash(f'✅ Fixed {len(fixed_items)} queue items successfully!', 'success')
                for item in fixed_items[:5]:  # Show first 5 items
                    flash(f"Dataset {item['dataset_id']}: {item['action']} ({item['reason']})", 'info')
                if len(fixed_items) > 5:
                    flash(f"... and {len(fixed_items) - 5} more items", 'info')
            else:
                flash('ℹ️ No queue items needed fixing.', 'info')
            return redirect(url_for('main.dashboard'))

    except Exception as e:
        import traceback
        error_msg = f'Error fixing stuck queue: {str(e)}'

        if request.is_json or request.headers.get('Content-Type') == 'application/json':
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'traceback': traceback.format_exc()
            }), 500
        else:
            flash(f'❌ {error_msg}', 'error')
            return redirect(url_for('main.dashboard'))


@datasets_bp.route('/api/datasets/<dataset_id>/enhance-fair', methods=['POST'])
@login_required
def enhance_fair_compliance(dataset_id):
    """Enhance FAIR compliance for a dataset"""
    try:
        dataset = Dataset.objects(id=dataset_id).first()
        if not dataset:
            return jsonify({'success': False, 'message': 'Dataset not found'}), 404

        # Check if user owns the dataset or is admin
        if dataset.user != current_user and not current_user.is_admin:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403

        # Apply FAIR enhancements
        from app.services.fair_enhancement_service import get_fair_enhancement_service
        fair_service = get_fair_enhancement_service()

        result = fair_service.enhance_dataset_fair_compliance(dataset)

        if result['success']:
            return jsonify({
                'success': True,
                'message': f'FAIR compliance enhanced. Score improved by {result["improvement"]:.1f} points.',
                'before_score': result['before_fair_score'],
                'after_score': result['after_fair_score'],
                'improvement': result['improvement'],
                'enhancements': result['enhancements_applied']
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to enhance FAIR compliance: {result.get("error", "Unknown error")}'
            })

    except Exception as e:
        logger.error(f"Error enhancing FAIR compliance for dataset {dataset_id}: {e}")
        return jsonify({'success': False, 'message': 'Error enhancing FAIR compliance'}), 500