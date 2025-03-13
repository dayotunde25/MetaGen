"""
Routes for dataset management.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, abort
from flask_login import login_required, current_user
from werkzeug.exceptions import NotFound

from app.forms import DatasetForm, SearchForm
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.services.dataset_service import get_dataset_service
from app.services.quality_assessment_service import quality_assessment_service

# Create blueprint
datasets = Blueprint('datasets', __name__)


@datasets.route('/datasets')
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


@datasets.route('/datasets/create', methods=['GET', 'POST'])
@login_required
def create():
    """Create a new dataset."""
    form = DatasetForm()
    
    if form.validate_on_submit():
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
            user_id=current_user.id
        )
        
        flash('Dataset created successfully.', 'success')
        return redirect(url_for('datasets.view', dataset_id=dataset.id))
    
    return render_template('datasets/create.html', form=form)


@datasets.route('/datasets/<int:dataset_id>')
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


@datasets.route('/datasets/<int:dataset_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(dataset_id):
    """Edit a dataset."""
    dataset = Dataset.find_by_id(dataset_id)
    
    if not dataset:
        abort(404)
    
    # Authorization check
    if dataset.user_id and dataset.user_id != current_user.id and not current_user.is_admin:
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
    
    return render_template('datasets/edit.html', form=form, dataset=dataset)


@datasets.route('/datasets/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete(dataset_id):
    """Delete a dataset."""
    dataset = Dataset.find_by_id(dataset_id)
    
    if not dataset:
        abort(404)
    
    # Authorization check
    if dataset.user_id and dataset.user_id != current_user.id and not current_user.is_admin:
        flash('You are not authorized to delete this dataset.', 'danger')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))
    
    dataset.delete()
    flash('Dataset deleted successfully.', 'success')
    return redirect(url_for('datasets.list'))


@datasets.route('/datasets/<int:dataset_id>/metadata')
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
                           quality=metadata_quality,
                           schema_org=schema_org)


@datasets.route('/datasets/<int:dataset_id>/quality')
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


@datasets.route('/datasets/<int:dataset_id>/assess-quality')
@login_required
def assess_quality(dataset_id):
    """Trigger quality assessment for a dataset."""
    dataset = Dataset.find_by_id(dataset_id)
    
    if not dataset:
        abort(404)
    
    # Authorization check
    if dataset.user_id and dataset.user_id != current_user.id and not current_user.is_admin:
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


@datasets.route('/datasets/<int:dataset_id>/process', methods=['POST'])
@login_required
def process_dataset(dataset_id):
    """
    Add dataset to processing queue.
    """
    dataset = Dataset.find_by_id(dataset_id)
    
    if not dataset:
        abort(404)
    
    # Authorization check
    if dataset.user_id and dataset.user_id != current_user.id and not current_user.is_admin:
        flash('You are not authorized to process this dataset.', 'danger')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))
    
    # Check if already in queue
    existing_queue = ProcessingQueue.get_by_dataset(dataset_id)
    
    if existing_queue and existing_queue.status in ['pending', 'processing']:
        flash('Dataset is already in the processing queue.', 'info')
        return redirect(url_for('datasets.view', dataset_id=dataset_id))
    
    # Add to queue
    if existing_queue:
        existing_queue.update(
            status='pending',
            progress=0.0,
            error=None
        )
    else:
        ProcessingQueue.create(
            dataset_id=dataset_id,
            status='pending',
            priority=5  # Default priority
        )
    
    flash('Dataset has been added to the processing queue.', 'success')
    return redirect(url_for('datasets.view', dataset_id=dataset_id))


@datasets.route('/datasets/my-datasets')
@login_required
def my_datasets():
    """View user's datasets."""
    datasets_list = Dataset.get_by_user(current_user.id)
    return render_template('datasets/my_datasets.html', datasets=datasets_list)