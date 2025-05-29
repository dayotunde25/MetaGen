"""
Routes for dataset management.
"""

import os
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, abort, jsonify
from flask_login import login_required, current_user
from werkzeug.exceptions import NotFound
from werkzeug.utils import secure_filename

from app.forms import DatasetForm, SearchForm
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.services.dataset_service import get_dataset_service
from app.services.quality_assessment_service import quality_assessment_service
from app.services.processing_service import processing_service

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

        # Process tags
        tags = form.tags.data

        # Create dataset
        dataset = Dataset.create(
            title=form.title.data,
            description=form.description.data,
            source=form.source.data,
            source_url=form.source_url.data,
            data_type=form.data_type.data,
            category=form.category.data,
            tags=tags,
            user=current_user,
            file_path=file_path,
            format=file_format,
            size=f"{file_size / 1024 / 1024:.2f} MB" if file_size else None
        )

        # Start automatic processing if we have a file or URL
        if file_path or form.source_url.data:
            try:
                # Start automatic processing immediately
                processing_started = processing_service.start_processing(
                    str(dataset.id),
                    current_app.config['UPLOAD_FOLDER']
                )

                if processing_started:
                    flash('Dataset created successfully! Processing started automatically.', 'success')
                else:
                    # Fallback to manual queue creation
                    ProcessingQueue.create(
                        dataset=str(dataset.id),
                        status='pending',
                        priority=1
                    )
                    flash('Dataset created successfully and added to processing queue!', 'info')
            except Exception as e:
                print(f"Warning: Could not start automatic processing: {e}")
                flash('Dataset created successfully! Processing will be available manually.', 'warning')
        else:
            flash('Dataset created successfully!', 'success')

        return redirect(url_for('datasets.view', dataset_id=dataset.id))

    return render_template('datasets/form.html', form=form)


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

    return render_template('datasets/view.html',
                           dataset=dataset,
                           quality=metadata_quality,
                           processing=processing)


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

    return render_template('datasets/metadata.html',
                           dataset=dataset,
                           metadata=metadata_quality,
                           schema_org=schema_org)


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
        # Start processing
        processing_started = processing_service.start_processing(
            dataset_id,
            current_app.config['UPLOAD_FOLDER']
        )

        if processing_started:
            flash('Processing started successfully!', 'success')
        else:
            flash('Processing is already running or failed to start.', 'warning')

    except Exception as e:
        flash(f'Error starting processing: {str(e)}', 'danger')

    return redirect(url_for('datasets.view', dataset_id=dataset_id))