from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.exceptions import NotFound
import json

from app.models import db
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.forms import DatasetForm
from app.services.dataset_service import get_dataset_service
from app.services.metadata_service import metadata_service

# Create blueprint
datasets_bp = Blueprint('datasets', __name__)

@datasets_bp.route('/datasets')
@login_required
def list_datasets():
    """List all datasets for the current user"""
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    return render_template('datasets/list.html', 
                          title='My Datasets',
                          datasets=user_datasets)

@datasets_bp.route('/datasets/add', methods=['GET', 'POST'])
@login_required
def add_dataset():
    """Add a new dataset"""
    form = DatasetForm()
    
    if form.validate_on_submit():
        # Process tags
        tags = form.tags.data.split(',') if form.tags.data else []
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        # Create dataset
        dataset = Dataset(
            title=form.title.data,
            description=form.description.data,
            source_url=form.source_url.data,
            source=form.source.data,
            data_type=form.data_type.data,
            category=form.category.data,
            user_id=current_user.id
        )
        dataset.tags_list = tags
        
        db.session.add(dataset)
        db.session.commit()
        
        flash('Dataset added successfully!', 'success')
        
        # Add to processing queue if URL is provided
        if form.source_url.data:
            dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
            dataset_service.add_to_processing_queue(dataset.id)
            flash('Dataset has been added to the processing queue.', 'info')
            
        return redirect(url_for('datasets.view_dataset', dataset_id=dataset.id))
    
    return render_template('datasets/add.html', 
                          title='Add Dataset',
                          form=form)

@datasets_bp.route('/datasets/<int:dataset_id>')
@login_required
def view_dataset(dataset_id):
    """View a specific dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access
    if dataset.user_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to view this dataset.', 'error')
        return redirect(url_for('datasets.list_datasets'))
    
    # Get metadata quality
    metadata_quality = MetadataQuality.query.filter_by(dataset_id=dataset_id).first()
    
    # Get processing status
    processing = ProcessingQueue.query.filter_by(dataset_id=dataset_id).first()
    
    return render_template('datasets/view.html', 
                          title=dataset.title,
                          dataset=dataset,
                          metadata=metadata_quality,
                          processing=processing)

@datasets_bp.route('/datasets/<int:dataset_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_dataset(dataset_id):
    """Edit a dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access
    if dataset.user_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to edit this dataset.', 'error')
        return redirect(url_for('datasets.list_datasets'))
    
    form = DatasetForm()
    
    if request.method == 'GET':
        # Populate form with dataset data
        form.title.data = dataset.title
        form.description.data = dataset.description
        form.source_url.data = dataset.source_url
        form.source.data = dataset.source
        form.data_type.data = dataset.data_type
        form.category.data = dataset.category
        form.tags.data = ','.join(dataset.tags_list) if dataset.tags_list else ''
    
    if form.validate_on_submit():
        # Process tags
        tags = form.tags.data.split(',') if form.tags.data else []
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        # Update dataset
        dataset.title = form.title.data
        dataset.description = form.description.data
        dataset.source_url = form.source_url.data
        dataset.source = form.source.data
        dataset.data_type = form.data_type.data
        dataset.category = form.category.data
        dataset.tags_list = tags
        
        db.session.commit()
        
        flash('Dataset updated successfully!', 'success')
        return redirect(url_for('datasets.view_dataset', dataset_id=dataset.id))
    
    return render_template('datasets/edit.html', 
                          title='Edit Dataset',
                          form=form,
                          dataset=dataset)

@datasets_bp.route('/datasets/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    """Delete a dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access
    if dataset.user_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to delete this dataset.', 'error')
        return redirect(url_for('datasets.list_datasets'))
    
    # Delete related data
    MetadataQuality.query.filter_by(dataset_id=dataset_id).delete()
    ProcessingQueue.query.filter_by(dataset_id=dataset_id).delete()
    
    # Delete dataset
    db.session.delete(dataset)
    db.session.commit()
    
    flash('Dataset deleted successfully!', 'success')
    return redirect(url_for('datasets.list_datasets'))

@datasets_bp.route('/datasets/<int:dataset_id>/process', methods=['POST'])
@login_required
def process_dataset(dataset_id):
    """Add dataset to processing queue"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access
    if dataset.user_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to process this dataset.', 'error')
        return redirect(url_for('datasets.list_datasets'))
    
    dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
    queue_item = dataset_service.add_to_processing_queue(dataset_id)
    
    if queue_item:
        flash('Dataset has been added to the processing queue.', 'success')
    else:
        flash('Failed to add dataset to processing queue.', 'error')
    
    return redirect(url_for('datasets.view_dataset', dataset_id=dataset_id))

@datasets_bp.route('/datasets/<int:dataset_id>/metadata')
@login_required
def dataset_metadata(dataset_id):
    """View dataset metadata"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user has access
    if dataset.user_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to view this dataset.', 'error')
        return redirect(url_for('datasets.list_datasets'))
    
    # Get metadata quality
    metadata_quality = MetadataQuality.query.filter_by(dataset_id=dataset_id).first()
    
    if not metadata_quality:
        flash('No metadata available for this dataset.', 'info')
        return redirect(url_for('datasets.view_dataset', dataset_id=dataset_id))
    
    return render_template('datasets/metadata.html', 
                          title=f'Metadata: {dataset.title}',
                          dataset=dataset,
                          metadata=metadata_quality)