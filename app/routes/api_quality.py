"""
API routes for dataset quality assessment.
"""

import json
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user

from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality, ProcessingQueue
from app.services.quality_assessment_service import quality_assessment_service
from app.services.dataset_service import get_dataset_service

# Create blueprint
api_quality = Blueprint('api_quality', __name__)


@api_quality.route('/api/quality/assess/<int:dataset_id>', methods=['POST'])
@login_required
def assess_dataset_quality(dataset_id):
    """
    Trigger quality assessment for a dataset.
    
    Args:
        dataset_id: ID of the dataset to assess
        
    Returns:
        JSON response with assessment results or error
    """
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        # Check authorization
        if dataset.user_id and dataset.user_id != current_user.id and not current_user.is_admin:
            return jsonify({"error": "Unauthorized to assess this dataset"}), 403
        
        # Get processed data
        dataset_service = get_dataset_service(current_app.config['UPLOAD_FOLDER'])
        processed_data = dataset_service.process_dataset({
            'path': dataset.source_url,
            'format': dataset.format,
            'size': None  # Size will be determined during processing
        })
        
        if not processed_data:
            return jsonify({"error": "Failed to process dataset data"}), 400
        
        # Assess quality
        metadata_quality = quality_assessment_service.assess_dataset_quality(dataset, processed_data)
        
        # Assess FAIR compliance
        fair_results = quality_assessment_service.assess_fair_compliance(dataset, metadata_quality)
        
        # Assess Schema.org compliance
        schema_org_results = quality_assessment_service.assess_schema_org_compliance(dataset, metadata_quality)
        
        # Mark dataset as processed
        dataset.update(processed=True)
        
        # Return results
        return jsonify({
            "success": True,
            "quality_score": metadata_quality.quality_score,
            "dimension_scores": metadata_quality.dimension_scores,
            "fair_scores": fair_results,
            "schema_org_score": schema_org_results["schema_org_score"],
            "recommendations": metadata_quality.recommendations_list
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error assessing quality for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_quality.route('/api/quality/dataset/<int:dataset_id>', methods=['GET'])
def get_dataset_quality(dataset_id):
    """
    Get quality assessment for a dataset.
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        JSON response with quality assessment data
    """
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        assessment = quality_assessment_service.get_quality_assessment(dataset_id)
        
        if not assessment:
            return jsonify({"error": "Quality assessment not found"}), 404
        
        return jsonify(assessment), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting quality for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_quality.route('/api/quality/recommendations/<int:dataset_id>', methods=['GET'])
def get_quality_recommendations(dataset_id):
    """
    Get quality improvement recommendations for a dataset.
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        JSON response with recommendations
    """
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        recommendations = quality_assessment_service.get_improvement_recommendations(dataset_id)
        
        return jsonify({"recommendations": recommendations}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting recommendations for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_quality.route('/api/quality/dimensions/<int:dataset_id>', methods=['GET'])
def get_quality_dimensions(dataset_id):
    """
    Get dimension scores for a dataset.
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        JSON response with dimension scores
    """
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        dimensions = quality_assessment_service.get_dimension_scores(dataset_id)
        
        if not dimensions:
            return jsonify({"error": "Quality dimensions not found"}), 404
        
        return jsonify({"dimensions": dimensions}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting dimensions for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_quality.route('/api/quality/fair/<int:dataset_id>', methods=['GET'])
def get_fair_scores(dataset_id):
    """
    Get FAIR scores for a dataset.
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        JSON response with FAIR scores
    """
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        fair_scores = quality_assessment_service.get_fair_scores(dataset_id)
        
        if not fair_scores:
            return jsonify({"error": "FAIR scores not found"}), 404
        
        return jsonify({"fair_scores": fair_scores}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting FAIR scores for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_quality.route('/api/quality/batch-assess', methods=['POST'])
@login_required
def batch_assess_quality():
    """
    Trigger quality assessment for multiple datasets.
    
    Request JSON:
        dataset_ids: List of dataset IDs to assess
        
    Returns:
        JSON response with batch processing status
    """
    try:
        data = request.get_json()
        if not data or 'dataset_ids' not in data:
            return jsonify({"error": "Missing dataset_ids in request"}), 400
        
        dataset_ids = data['dataset_ids']
        
        if not isinstance(dataset_ids, list):
            return jsonify({"error": "dataset_ids must be a list"}), 400
        
        # Queue datasets for assessment
        queued_count = 0
        for dataset_id in dataset_ids:
            dataset = Dataset.find_by_id(dataset_id)
            
            if not dataset:
                continue
                
            # Check authorization
            if dataset.user_id and dataset.user_id != current_user.id and not current_user.is_admin:
                continue
            
            # Add to processing queue if not already queued
            queue_item = ProcessingQueue.get_by_dataset(dataset_id)
            
            if not queue_item or queue_item.status in ['completed', 'failed']:
                # Create or update queue item
                if queue_item:
                    queue_item.update(
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
                
                queued_count += 1
        
        return jsonify({
            "success": True,
            "message": f"Queued {queued_count} datasets for quality assessment",
            "queued_count": queued_count
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in batch quality assessment: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_quality.route('/api/quality/issues/<int:dataset_id>', methods=['GET'])
def get_quality_issues(dataset_id):
    """
    Get quality issues for a dataset.
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        JSON response with quality issues
    """
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        issues = quality_assessment_service.get_quality_issues(dataset_id)
        
        return jsonify({"issues": issues}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting issues for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_quality.route('/api/quality/score/<int:dataset_id>', methods=['GET'])
def get_quality_score(dataset_id):
    """
    Get overall quality score for a dataset.
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        JSON response with quality score
    """
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        score = quality_assessment_service.get_overall_quality_score(dataset_id)
        
        return jsonify({"quality_score": score}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting quality score for dataset {dataset_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500