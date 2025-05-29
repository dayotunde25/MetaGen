from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required, current_user

from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.forms import SearchForm
from app.services.semantic_search_service import get_semantic_search_service

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('main/index.html', title='Home')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    # Get user-specific stats (only datasets uploaded by current user)
    user_datasets = Dataset.objects(user=current_user)
    user_dataset_ids = [str(d.id) for d in user_datasets]

    # User-specific statistics
    stats = {
        'totalDatasets': user_datasets.count(),
        'processedDatasets': user_datasets.filter(status='completed').count(),
        'fairCompliantDatasets': MetadataQuality.objects(
            dataset__in=user_dataset_ids,
            fair_compliant=True
        ).count(),
        'queuedDatasets': ProcessingQueue.objects(
            dataset__in=user_dataset_ids,
            status__in=['pending', 'processing']
        ).count()
    }

    # Get user's recent datasets (only their own)
    recent_datasets = list(user_datasets.order_by('-created_at').limit(5))

    # Get featured datasets (curated selection from all users)
    # Featured datasets are high-quality, FAIR-compliant datasets from the platform
    featured_quality_records = MetadataQuality.objects(
        fair_compliant=True,
        quality_score__gte=80  # High quality threshold
    ).order_by('-quality_score').limit(10)

    # Extract the actual dataset objects and filter for completed ones
    featured_datasets = []
    for quality_record in featured_quality_records:
        if quality_record.dataset and quality_record.dataset.status == 'completed':
            featured_datasets.append(quality_record.dataset)
            if len(featured_datasets) >= 6:  # Limit to 6 featured datasets
                break

    # Get processing queue for user
    processing_queue = list(ProcessingQueue.objects(
        dataset__in=user_dataset_ids,
        status__in=['pending', 'processing']
    ))

    return render_template('main/dashboard.html',
                          title='Dashboard',
                          stats=stats,
                          recent_datasets=recent_datasets,
                          featured_datasets=featured_datasets,
                          processing_queue=processing_queue)

@main_bp.route('/statistics')
@login_required
def statistics():
    """Statistics page"""
    # Get overall stats
    stats = {
        'total_datasets': Dataset.objects.count(),
        'completed_datasets': Dataset.objects(status='completed').count(),
        'fair_compliant': MetadataQuality.objects(fair_compliant=True).count(),
        'schema_org_compliant': MetadataQuality.objects(schema_org_compliant=True).count(),
    }

    # Get category stats using aggregation
    categories = Dataset.objects.aggregate([
        {"$group": {"_id": "$category", "count": {"$sum": 1}}}
    ])
    category_stats = [{'name': cat['_id'] or 'Uncategorized', 'count': cat['count']} for cat in categories]

    # Get data type stats using aggregation
    data_types = Dataset.objects.aggregate([
        {"$group": {"_id": "$data_type", "count": {"$sum": 1}}}
    ])
    data_type_stats = [{'name': dt['_id'] or 'Unknown', 'count': dt['count']} for dt in data_types]

    # Get FAIR scores using aggregation
    fair_aggregation = MetadataQuality.objects.aggregate([
        {"$group": {
            "_id": None,
            "avg_findable": {"$avg": "$findable_score"},
            "avg_accessible": {"$avg": "$accessible_score"},
            "avg_interoperable": {"$avg": "$interoperable_score"},
            "avg_reusable": {"$avg": "$reusable_score"}
        }}
    ])

    fair_result = list(fair_aggregation)
    if fair_result:
        fair_data = fair_result[0]
        fair_stats = {
            'findable': round(fair_data.get('avg_findable', 0), 2),
            'accessible': round(fair_data.get('avg_accessible', 0), 2),
            'interoperable': round(fair_data.get('avg_interoperable', 0), 2),
            'reusable': round(fair_data.get('avg_reusable', 0), 2),
        }
        fair_stats['overall'] = round(sum(fair_stats.values()) / 4, 2)
    else:
        fair_stats = {'findable': 0, 'accessible': 0, 'interoperable': 0, 'reusable': 0, 'overall': 0}

    return render_template('main/statistics.html',
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
    total_results = 0
    page = 1
    per_page = 10

    # Get page number from request
    try:
        page = int(request.args.get('page', 1))
        if page < 1:
            page = 1
    except (ValueError, TypeError):
        page = 1

    # Handle GET parameters for search
    if request.method == 'GET':
        form.query.data = request.args.get('query', '')
        form.category.data = request.args.get('category', '')
        form.data_type.data = request.args.get('data_type', '')

    # Always perform search (show all datasets if no filters)
    # Build MongoDB query for filtering
    query_filters = {}

    # Add category filter
    if form.category.data:
        query_filters['category'] = form.category.data

    # Add data type filter
    if form.data_type.data:
        query_filters['data_type'] = form.data_type.data

    # Get all datasets that match filters
    filtered_datasets = list(Dataset.objects(**query_filters))

    # Use semantic search if query provided, otherwise use filtered datasets
    if form.query.data and form.query.data.strip():
        search_term = form.query.data.strip()

        # Try semantic search first
        try:
            semantic_search_service = get_semantic_search_service()

            # Index datasets if not already done (this is cached)
            semantic_search_service.index_datasets(filtered_datasets)

            # Perform semantic search
            search_results = semantic_search_service.search(
                query=search_term,
                datasets=filtered_datasets,
                limit=1000  # Get more results for pagination
            )

            # Extract datasets from search results (already scored and sorted)
            all_results = [result['dataset'] for result in search_results]

        except Exception as e:
            # Fall back to basic text search if semantic search fails
            from mongoengine import Q
            datasets_query = Dataset.objects(**query_filters)
            text_query = (
                Q(title__icontains=search_term) |
                Q(description__icontains=search_term) |
                Q(tags__icontains=search_term) |
                Q(source__icontains=search_term)
            )
            all_results = list(datasets_query.filter(text_query).order_by('-created_at'))
    else:
        # No search query - show all filtered datasets
        all_results = sorted(filtered_datasets, key=lambda d: d.created_at, reverse=True)

    # Get total count for pagination
    total_results = len(all_results)

    # Calculate pagination
    offset = (page - 1) * per_page
    total_pages = (total_results + per_page - 1) // per_page  # Ceiling division

    # Apply pagination to results
    results = all_results[offset:offset + per_page]

    # Pagination info
    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total_results,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_num': page - 1 if page > 1 else None,
        'next_num': page + 1 if page < total_pages else None
    }

    return render_template('main/search.html',
                          title='Search Datasets',
                          form=form,
                          results=results,
                          pagination=pagination,
                          total_results=total_results)


@main_bp.route('/admin/reindex-search')
@login_required
def reindex_search():
    """Admin route to reindex all datasets for semantic search"""
    # Simple admin check (you might want to implement proper admin roles)
    if not current_user.username == 'admin':
        return redirect(url_for('main.dashboard'))

    try:
        from app.services.dataset_service import get_dataset_service
        from flask import current_app

        dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
        success = dataset_service.reindex_all_datasets()

        if success:
            return f"<h2>✅ Search Index Updated</h2><p>All datasets have been reindexed for semantic search.</p><a href='{url_for('main.search')}'>Go to Search</a>"
        else:
            return f"<h2>⚠️ Indexing Failed</h2><p>Failed to reindex datasets. Check logs for details.</p><a href='{url_for('main.dashboard')}'>Back to Dashboard</a>"

    except Exception as e:
        return f"<h2>❌ Error</h2><p>Error during reindexing: {e}</p><a href='{url_for('main.dashboard')}'>Back to Dashboard</a>"

@main_bp.route('/docs')
def documentation():
    """Documentation page"""
    return render_template('main/documentation.html', title='Documentation')

@main_bp.route('/favicon.ico')
def favicon():
    """Favicon route"""
    return '', 204  # No content