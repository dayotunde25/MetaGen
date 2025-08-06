"""
Dataset Feedback Routes

This module handles routes for dataset feedback, ratings, and comments.
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models.dataset import Dataset
from app.models.dataset_feedback import DatasetFeedback, DatasetFeedbackHelpful
from app.models.user import User
from app.forms import FeedbackForm
import logging

logger = logging.getLogger(__name__)

feedback_bp = Blueprint('feedback', __name__)


@feedback_bp.route('/dataset/<dataset_id>/feedback', methods=['GET'])
def view_feedback(dataset_id):
    """View all feedback for a dataset"""
    try:
        dataset = Dataset.objects(id=dataset_id).first()
        if not dataset:
            flash('Dataset not found', 'error')
            return redirect(url_for('main.index'))

        # Get feedback summary
        feedback_summary = DatasetFeedback.get_dataset_feedback_summary(dataset_id)

        # Get all feedback for display
        all_feedback = DatasetFeedback.objects(dataset=dataset_id).order_by('-created_at')

        return render_template('feedback/view_feedback.html',
                             dataset=dataset,
                             feedback_summary=feedback_summary,
                             all_feedback=all_feedback)

    except Exception as e:
        logger.error(f"Error viewing feedback for dataset {dataset_id}: {e}")
        flash('Error loading feedback', 'error')
        return redirect(url_for('main.index'))


@feedback_bp.route('/dataset/<dataset_id>/feedback/api', methods=['GET'])
def get_feedback_api(dataset_id):
    """API endpoint to get feedback data for a dataset"""
    try:
        dataset = Dataset.objects(id=dataset_id).first()
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404

        # Get feedback summary
        feedback_summary = DatasetFeedback.get_dataset_feedback_summary(dataset_id)

        # Get recent feedback for display
        recent_feedback = DatasetFeedback.objects(dataset=dataset_id).order_by('-created_at').limit(5)

        # Format feedback data
        feedback_data = []
        for feedback in recent_feedback:
            feedback_data.append({
                'id': str(feedback.id),
                'rating': feedback.rating,
                'comment': feedback.comment,
                'user_name': feedback.user.username if feedback.user and not feedback.is_anonymous else None,
                'is_anonymous': feedback.is_anonymous,
                'created_at': feedback.created_at.isoformat(),
                'helpful_count': feedback.helpful_count or 0
            })

        return jsonify({
            'success': True,
            'feedback_summary': feedback_summary,
            'recent_feedback': feedback_data,
            'user_authenticated': current_user.is_authenticated
        })

    except Exception as e:
        logger.error(f"Error getting feedback API for dataset {dataset_id}: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500





@feedback_bp.route('/dataset/<dataset_id>/feedback/add', methods=['GET', 'POST'])
@login_required
def add_feedback(dataset_id):
    """Add feedback for a dataset"""
    try:
        dataset = Dataset.objects(id=dataset_id).first()
        if not dataset:
            flash('Dataset not found', 'error')
            return redirect(url_for('main.index'))

        form = FeedbackForm()
        existing_feedback = DatasetFeedback.get_user_feedback_for_dataset(current_user.id, dataset_id)

        if request.method == 'GET':
            if existing_feedback:
                # Populate form with existing feedback data
                form.rating.data = str(existing_feedback.rating)
                form.satisfaction.data = str(existing_feedback.satisfaction) if existing_feedback.satisfaction else ''
                form.usefulness.data = str(existing_feedback.usefulness) if existing_feedback.usefulness else ''
                form.quality.data = str(existing_feedback.quality) if existing_feedback.quality else ''
                form.comment.data = existing_feedback.comment
                form.feedback_type.data = existing_feedback.feedback_type
                form.is_anonymous.data = existing_feedback.is_anonymous

            return render_template('feedback/add_feedback.html',
                                   dataset=dataset,
                                   form=form,
                                   existing_feedback=existing_feedback)

        if form.validate_on_submit():
            rating = int(form.rating.data)
            satisfaction = form.satisfaction.data
            usefulness = form.usefulness.data
            quality = form.quality.data
            comment = form.comment.data.strip() if form.comment.data else ''
            feedback_type = form.feedback_type.data
            is_anonymous = form.is_anonymous.data

            if existing_feedback:
                # Update existing feedback
                existing_feedback.rating = rating
                existing_feedback.satisfaction = int(satisfaction) if satisfaction else None
                existing_feedback.usefulness = int(usefulness) if usefulness else None
                existing_feedback.quality = int(quality) if quality else None
                existing_feedback.comment = comment
                existing_feedback.feedback_type = feedback_type
                existing_feedback.is_anonymous = is_anonymous
                existing_feedback.save()

                flash('Your feedback has been updated successfully!', 'success')
            else:
                # Create new feedback
                feedback_data = {
                    'dataset': dataset,
                    'user': current_user._get_current_object(),
                    'rating': rating,
                    'feedback_type': feedback_type,
                    'is_anonymous': is_anonymous
                }

                if satisfaction:
                    feedback_data['satisfaction'] = int(satisfaction)
                if usefulness:
                    feedback_data['usefulness'] = int(usefulness)
                if quality:
                    feedback_data['quality'] = int(quality)
                if comment:
                    feedback_data['comment'] = comment

                DatasetFeedback.create(**feedback_data)
                flash('Thank you for your feedback!', 'success')

            return redirect(url_for('datasets.metadata', dataset_id=dataset_id))
        else:
            # Form validation failed
            flash('Please correct the errors in the form.', 'error')
            return render_template('feedback/add_feedback.html',
                                   dataset=dataset,
                                   form=form,
                                   existing_feedback=existing_feedback)

    except Exception as e:
        logger.error(f"Error adding feedback for dataset {dataset_id}: {e}")
        flash('Error submitting feedback. Please try again.', 'error')
        return redirect(url_for('datasets.metadata', dataset_id=dataset_id))


@feedback_bp.route('/feedback/<feedback_id>/helpful', methods=['POST'])
@login_required
def mark_helpful(feedback_id):
    """Mark feedback as helpful"""
    try:
        feedback = DatasetFeedback.objects(id=feedback_id).first()
        if not feedback:
            return jsonify({'success': False, 'message': 'Feedback not found'})
        
        # Check if user already marked this as helpful
        existing_helpful = DatasetFeedbackHelpful.objects(
            feedback=feedback, 
            user=current_user._get_current_object()
        ).first()
        
        if existing_helpful:
            # Remove helpful mark
            existing_helpful.delete()
            feedback.helpful_count = max(0, feedback.helpful_count - 1)
            feedback.save()
            
            return jsonify({
                'success': True, 
                'action': 'removed',
                'helpful_count': feedback.helpful_count
            })
        else:
            # Add helpful mark
            DatasetFeedbackHelpful.create(
                feedback=feedback,
                user=current_user._get_current_object()
            )
            feedback.helpful_count += 1
            feedback.save()
            
            return jsonify({
                'success': True, 
                'action': 'added',
                'helpful_count': feedback.helpful_count
            })
    
    except Exception as e:
        logger.error(f"Error marking feedback {feedback_id} as helpful: {e}")
        return jsonify({'success': False, 'message': 'Error processing request'})


@feedback_bp.route('/api/dataset/<dataset_id>/rating', methods=['GET'])
def get_dataset_rating(dataset_id):
    """Get dataset rating summary via API"""
    try:
        feedback_summary = DatasetFeedback.get_dataset_feedback_summary(dataset_id)
        return jsonify({
            'success': True,
            'data': feedback_summary
        })
    except Exception as e:
        logger.error(f"Error getting rating for dataset {dataset_id}: {e}")
        return jsonify({'success': False, 'message': 'Error loading rating'})



@feedback_bp.route('/feedback/<feedback_id>/delete', methods=['POST'])
@login_required
def delete_feedback(feedback_id):
    """Delete user's own feedback"""
    try:
        feedback = DatasetFeedback.objects(
            id=feedback_id, 
            user=current_user._get_current_object()
        ).first()
        
        if not feedback:
            flash('Feedback not found or you do not have permission to delete it', 'error')
            return redirect(url_for('main.index'))
        
        dataset_id = str(feedback.dataset.id)
        feedback.delete()
        
        flash('Your feedback has been deleted', 'success')
        return redirect(url_for('datasets.metadata', dataset_id=dataset_id))
    
    except Exception as e:
        logger.error(f"Error deleting feedback {feedback_id}: {e}")
        flash('Error deleting feedback', 'error')
        return redirect(url_for('main.index'))


@feedback_bp.route('/feedback/quick-rate/<dataset_id>', methods=['POST'])
@login_required
def quick_rate(dataset_id):
    """Quick rating endpoint for AJAX requests."""
    try:
        dataset = Dataset.find_by_id(dataset_id)
        if not dataset:
            return jsonify({'success': False, 'message': 'Dataset not found'}), 404

        data = request.get_json()
        if not data or 'rating' not in data:
            return jsonify({'success': False, 'message': 'Rating is required'}), 400

        rating = int(data['rating'])
        if rating < 1 or rating > 5:
            return jsonify({'success': False, 'message': 'Rating must be between 1 and 5'}), 400

        # Check if user already rated this dataset
        existing_feedback = DatasetFeedback.get_user_feedback_for_dataset(current_user.id, dataset_id)

        if existing_feedback:
            # Update existing rating
            existing_feedback.rating = rating
            existing_feedback.save()
            message = 'Rating updated successfully'
        else:
            # Create new quick rating (no comment)
            feedback = DatasetFeedback.create(
                dataset=dataset,
                user=current_user,
                rating=rating,
                comment="",  # Quick rating without comment
                feedback_type='rating'
            )
            message = 'Rating submitted successfully'

        # Get updated summary
        feedback_summary = DatasetFeedback.get_dataset_feedback_summary(dataset_id)

        return jsonify({
            'success': True,
            'message': message,
            'new_average': feedback_summary.average_rating if feedback_summary else rating,
            'total_reviews': feedback_summary.total_reviews if feedback_summary else 1
        })

    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid rating value'}), 400
    except Exception as e:
        logger.error(f"Error in quick rating: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500
