from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from sqlalchemy import desc
import json

from app.models import db
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.forms import SearchForm
from app.services.dataset_service import get_dataset_service

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html', title='Home')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    # Get stats
    total_datasets = Dataset.query.count()
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).count()
    processing_datasets = ProcessingQueue.query.filter_by(status='processing').count()
    completed_datasets = Dataset.query.filter_by(status='completed').count()
    
    # Get recent datasets
    recent_datasets = Dataset.query.order_by(desc(Dataset.created_at)).limit(5).all()
    
    # Get processing queue for user
    processing_queue = db.session.query(ProcessingQueue, Dataset)\
        .join(Dataset, ProcessingQueue.dataset_id == Dataset.id)\
        .filter(Dataset.user_id == current_user.id)\
        .filter(ProcessingQueue.status.in_(['queued', 'processing']))\
        .all()
    
    return render_template('dashboard.html', 
                          title='Dashboard',
                          total_datasets=total_datasets,
                          user_datasets=user_datasets,
                          processing_datasets=processing_datasets,
                          completed_datasets=completed_datasets,
                          recent_datasets=recent_datasets,
                          processing_queue=processing_queue)

@main_bp.route('/statistics')
@login_required
def statistics():
    """Statistics page"""
    # Get overall stats
    stats = {
        'total_datasets': Dataset.query.count(),
        'completed_datasets': Dataset.query.filter_by(status='completed').count(),
        'fair_compliant': MetadataQuality.query.filter_by(fair_compliant=True).count(),
        'schema_org_compliant': MetadataQuality.query.filter_by(schema_org_compliant=True).count(),
    }
    
    # Get category stats
    categories = db.session.query(Dataset.category, db.func.count(Dataset.id))\
        .group_by(Dataset.category)\
        .all()
    
    category_stats = [{'name': cat or 'Uncategorized', 'count': count} for cat, count in categories]
    
    # Get data type stats
    data_types = db.session.query(Dataset.data_type, db.func.count(Dataset.id))\
        .group_by(Dataset.data_type)\
        .all()
    
    data_type_stats = [{'name': dt or 'Unknown', 'count': count} for dt, count in data_types]
    
    # Get FAIR scores
    fair_scores = db.session.query(
        db.func.avg(MetadataQuality.findable_score),
        db.func.avg(MetadataQuality.accessible_score),
        db.func.avg(MetadataQuality.interoperable_score),
        db.func.avg(MetadataQuality.reusable_score)
    ).first()
    
    fair_stats = {
        'findable': round(fair_scores[0] or 0, 2),
        'accessible': round(fair_scores[1] or 0, 2),
        'interoperable': round(fair_scores[2] or 0, 2),
        'reusable': round(fair_scores[3] or 0, 2),
        'overall': round(sum([fair_scores[0] or 0, fair_scores[1] or 0, 
                             fair_scores[2] or 0, fair_scores[3] or 0]) / 4, 2)
    }
    
    return render_template('statistics.html', 
                          title='Statistics',
                          stats=stats,
                          category_stats=category_stats,
                          data_type_stats=data_type_stats,
                          fair_stats=fair_stats)

@main_bp.route('/search', methods=['GET', 'POST'])
def search():
    """Search datasets"""
    form = SearchForm()
    results = []
    
    if request.method == 'GET' and request.args.get('query'):
        form.query.data = request.args.get('query')
        form.category.data = request.args.get('category', '')
        form.data_type.data = request.args.get('data_type', '')
    
    if form.validate_on_submit() or (request.method == 'GET' and request.args.get('query')):
        # Prepare filters
        filters = {}
        if form.category.data:
            filters['category'] = form.category.data
        if form.data_type.data:
            filters['data_type'] = form.data_type.data
            
        # Get dataset service
        dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
        
        # Perform search
        results = dataset_service.search_datasets(
            query=form.query.data,
            filters=filters,
            sort_by='created_at',
            limit=20,
            offset=0
        )
        
    return render_template('search.html', 
                          title='Search Datasets',
                          form=form,
                          results=results)

@main_bp.route('/docs')
def documentation():
    """Documentation page"""
    return render_template('documentation.html', title='Documentation')