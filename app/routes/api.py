from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from sqlalchemy import desc
import json

from app.models import db
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.services.dataset_service import get_dataset_service
from app.services.metadata_service import metadata_service
from app.services.nlp_service import nlp_service

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/stats')
def get_stats():
    """Get system statistics"""
    stats = {
        'totalDatasets': Dataset.query.count(),
        'processedDatasets': Dataset.query.filter_by(status='completed').count(),
        'fairCompliantDatasets': MetadataQuality.query.filter_by(fair_compliant=True).count(),
        'queuedDatasets': ProcessingQueue.query.filter(ProcessingQueue.status.in_(['queued', 'processing'])).count()
    }
    return jsonify(stats)

@api_bp.route('/datasets')
def get_datasets():
    """Get all datasets"""
    # Check if user specific filter is required
    user_id = request.args.get('user_id', None, type=int)
    
    query = Dataset.query
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    
    # Get optional limit
    limit = request.args.get('limit', None, type=int)
    if limit:
        query = query.limit(limit)
    
    datasets = query.order_by(desc(Dataset.created_at)).all()
    
    # Convert to JSON serializable format
    result = [dataset.to_dict() for dataset in datasets]
    
    return jsonify(result)

@api_bp.route('/datasets/<int:dataset_id>')
def get_dataset(dataset_id):
    """Get a specific dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Add metadata quality if available
    metadata = MetadataQuality.query.filter_by(dataset_id=dataset_id).first()
    processing = ProcessingQueue.query.filter_by(dataset_id=dataset_id).first()
    
    result = dataset.to_dict()
    
    if metadata:
        result['metadata'] = metadata.to_dict()
    
    if processing:
        result['processing'] = processing.to_dict()
    
    return jsonify(result)

@api_bp.route('/datasets/<int:dataset_id>', methods=['PUT'])
@login_required
def update_dataset(dataset_id):
    """Update a dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check permission
    if dataset.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Permission denied'}), 403
    
    data = request.get_json()
    
    # Update fields
    if 'title' in data:
        dataset.title = data['title']
    if 'description' in data:
        dataset.description = data['description']
    if 'source' in data:
        dataset.source = data['source']
    if 'source_url' in data:
        dataset.source_url = data['source_url']
    if 'data_type' in data:
        dataset.data_type = data['data_type']
    if 'category' in data:
        dataset.category = data['category']
    if 'tags' in data:
        dataset.tags_list = data['tags']
    
    db.session.commit()
    
    return jsonify(dataset.to_dict())

@api_bp.route('/datasets/<int:dataset_id>', methods=['DELETE'])
@login_required
def delete_dataset(dataset_id):
    """Delete a dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check permission
    if dataset.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Permission denied'}), 403
    
    # Delete related records
    MetadataQuality.query.filter_by(dataset_id=dataset_id).delete()
    ProcessingQueue.query.filter_by(dataset_id=dataset_id).delete()
    
    db.session.delete(dataset)
    db.session.commit()
    
    return jsonify({'success': True})

@api_bp.route('/datasets', methods=['POST'])
@login_required
def create_dataset():
    """Create a new dataset"""
    data = request.get_json()
    
    # Validate required fields
    if 'title' not in data:
        return jsonify({'error': 'Title is required'}), 400
    
    # Process tags
    tags = data.get('tags', [])
    
    # Create dataset
    dataset = Dataset(
        title=data['title'],
        description=data.get('description'),
        source_url=data.get('source_url'),
        source=data.get('source'),
        data_type=data.get('data_type'),
        category=data.get('category'),
        user_id=current_user.id
    )
    dataset.tags_list = tags
    
    db.session.add(dataset)
    db.session.commit()
    
    # Add to processing queue if URL is provided
    if data.get('source_url'):
        dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
        dataset_service.add_to_processing_queue(dataset.id)
    
    return jsonify(dataset.to_dict()), 201

@api_bp.route('/processing-queue')
def get_processing_queue():
    """Get processing queue"""
    # Optional user filter
    user_id = request.args.get('user_id', None, type=int)
    
    if user_id:
        # Join with datasets to filter by user
        queue_items = db.session.query(ProcessingQueue, Dataset)\
            .join(Dataset, ProcessingQueue.dataset_id == Dataset.id)\
            .filter(Dataset.user_id == user_id)\
            .all()
        
        result = []
        for queue, dataset in queue_items:
            item = queue.to_dict()
            item['dataset'] = dataset.to_dict()
            result.append(item)
    else:
        # All queue items
        queue_items = ProcessingQueue.query.all()
        result = [item.to_dict() for item in queue_items]
    
    return jsonify(result)

@api_bp.route('/process-dataset/<int:dataset_id>', methods=['POST'])
@login_required
def process_dataset(dataset_id):
    """Process a dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check permission
    if dataset.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Permission denied'}), 403
    
    dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
    queue_item = dataset_service.add_to_processing_queue(dataset_id)
    
    if queue_item:
        return jsonify(queue_item.to_dict())
    else:
        return jsonify({'error': 'Failed to add to processing queue'}), 500

@api_bp.route('/metadata-quality/<int:dataset_id>')
def get_metadata_quality(dataset_id):
    """Get metadata quality for a dataset"""
    metadata = MetadataQuality.query.filter_by(dataset_id=dataset_id).first()
    
    if not metadata:
        return jsonify({'error': 'No metadata found for this dataset'}), 404
    
    return jsonify(metadata.to_dict())

@api_bp.route('/search')
def search_datasets():
    """Search datasets with semantic search"""
    query = request.args.get('q', '')
    category = request.args.get('category')
    data_type = request.args.get('data_type')
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Build filters
    filters = {}
    if category:
        filters['category'] = category
    if data_type:
        filters['data_type'] = data_type
    
    # Get dataset service
    dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
    
    # Perform search
    results = dataset_service.search_datasets(
        query=query,
        filters=filters,
        limit=limit,
        offset=offset
    )
    
    # Convert to dict
    result_dicts = []
    for dataset in results:
        dataset_dict = dataset.to_dict()
        if hasattr(dataset, 'similarity'):
            dataset_dict['similarity'] = dataset.similarity
        result_dicts.append(dataset_dict)
    
    return jsonify({
        'query': query,
        'results': result_dicts,
        'total': len(results),
        'limit': limit,
        'offset': offset
    })

@api_bp.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    """Extract keywords from text"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Extract keywords
    keywords = nlp_service.extract_keywords(text)
    
    return jsonify({
        'keywords': keywords
    })

@api_bp.route('/suggest-tags', methods=['POST'])
def suggest_tags():
    """Suggest tags for a dataset"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Suggest tags
    tags = nlp_service.suggest_tags(text)
    
    return jsonify({
        'tags': tags
    })