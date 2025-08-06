"""
Dataset Feedback and Rating Model

This module defines the DatasetFeedback model for storing user feedback,
ratings, and comments on datasets.
"""

from mongoengine import Document, StringField, IntField, DateTimeField, ReferenceField, BooleanField
from datetime import datetime


class DatasetFeedback(Document):
    """
    Model for storing user feedback and ratings on datasets.
    
    Attributes:
        dataset: Reference to the dataset being rated
        user: Reference to the user providing feedback
        rating: Rating score (1-5 stars)
        satisfaction: Satisfaction level (1-5)
        usefulness: How useful the dataset is (1-5)
        quality: Data quality rating (1-5)
        comment: Optional text comment
        feedback_type: Type of feedback (rating, comment, suggestion)
        is_helpful: Whether this feedback was marked as helpful by others
        helpful_count: Number of users who found this feedback helpful
        created_at: When the feedback was created
        updated_at: When the feedback was last updated
        is_verified: Whether the feedback is from a verified user
        is_anonymous: Whether the feedback should be displayed anonymously
    """
    
    # Core references
    dataset = ReferenceField('Dataset', required=True)
    user = ReferenceField('User', required=True)
    
    # Rating fields (1-5 scale)
    rating = IntField(min_value=1, max_value=5, required=True)
    satisfaction = IntField(min_value=1, max_value=5)
    usefulness = IntField(min_value=1, max_value=5)
    quality = IntField(min_value=1, max_value=5)
    
    # Text feedback
    comment = StringField(max_length=1000)
    feedback_type = StringField(max_length=20, choices=['rating', 'comment', 'suggestion', 'issue'], default='rating')
    
    # Feedback metadata
    is_helpful = BooleanField(default=False)
    helpful_count = IntField(default=0)
    is_verified = BooleanField(default=False)
    is_anonymous = BooleanField(default=False)
    
    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'dataset_feedback',
        'indexes': [
            'dataset',
            'user',
            'rating',
            'created_at',
            ('dataset', 'user'),  # Compound index for unique user-dataset feedback
            'helpful_count'
        ]
    }
    
    def save(self, *args, **kwargs):
        """Override save to update timestamp"""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    @classmethod
    def create(cls, **kwargs):
        """
        Create a new dataset feedback record.

        Args:
            **kwargs: Keyword arguments with feedback attributes

        Returns:
            New DatasetFeedback instance
        """
        feedback = cls(**kwargs)
        feedback.save()
        return feedback

    @classmethod
    def create(cls, **kwargs):
        """
        Create a new dataset feedback record.

        Args:
            **kwargs: Keyword arguments with feedback attributes

        Returns:
            New DatasetFeedback instance
        """
        feedback = cls(**kwargs)
        feedback.save()
        return feedback
    
    @classmethod
    def get_dataset_average_rating(cls, dataset_id):
        """Get average rating for a dataset"""
        try:
            from mongoengine import Q
            feedbacks = cls.objects(dataset=dataset_id)
            if feedbacks:
                total_rating = sum(feedback.rating for feedback in feedbacks)
                return round(total_rating / len(feedbacks), 1)
            return 0.0
        except Exception:
            return 0.0
    
    @classmethod
    def get_dataset_rating_distribution(cls, dataset_id):
        """Get rating distribution for a dataset"""
        try:
            distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            feedbacks = cls.objects(dataset=dataset_id)
            
            for feedback in feedbacks:
                distribution[feedback.rating] += 1
            
            total = sum(distribution.values())
            if total > 0:
                # Convert to percentages
                for rating in distribution:
                    distribution[rating] = round((distribution[rating] / total) * 100, 1)
            
            return distribution
        except Exception:
            return {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    @classmethod
    def get_user_feedback_for_dataset(cls, user_id, dataset_id):
        """Get user's existing feedback for a dataset"""
        try:
            return cls.objects(user=user_id, dataset=dataset_id).first()
        except Exception:
            return None
    
    @classmethod
    def get_dataset_feedback(cls, dataset_id):
        """Get all feedback for a dataset"""
        try:
            return cls.objects(dataset=dataset_id).order_by('-created_at')
        except Exception:
            return []

    @classmethod
    def get_dataset_feedback_summary(cls, dataset_id):
        """Get comprehensive feedback summary for a dataset"""
        try:
            feedbacks = cls.objects(dataset=dataset_id)
            
            if not feedbacks:
                return {
                    'total_reviews': 0,
                    'average_rating': 0.0,
                    'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                    'average_satisfaction': 0.0,
                    'average_usefulness': 0.0,
                    'average_quality': 0.0,
                    'recent_comments': []
                }
            
            # Calculate averages
            total_reviews = len(feedbacks)
            avg_rating = sum(f.rating for f in feedbacks) / total_reviews
            
            # Calculate other averages (only for feedbacks that have these fields)
            satisfaction_feedbacks = [f for f in feedbacks if f.satisfaction]
            usefulness_feedbacks = [f for f in feedbacks if f.usefulness]
            quality_feedbacks = [f for f in feedbacks if f.quality]
            
            avg_satisfaction = sum(f.satisfaction for f in satisfaction_feedbacks) / len(satisfaction_feedbacks) if satisfaction_feedbacks else 0
            avg_usefulness = sum(f.usefulness for f in usefulness_feedbacks) / len(usefulness_feedbacks) if usefulness_feedbacks else 0
            avg_quality = sum(f.quality for f in quality_feedbacks) / len(quality_feedbacks) if quality_feedbacks else 0
            
            # Get recent comments and feedback
            recent_comments = feedbacks.filter(comment__ne=None).filter(comment__ne='').order_by('-created_at')[:5]
            recent_feedback = feedbacks.order_by('-created_at')[:3]

            return {
                'total_reviews': total_reviews,
                'average_rating': round(avg_rating, 1),
                'rating_distribution': cls.get_dataset_rating_distribution(dataset_id),
                'average_satisfaction': round(avg_satisfaction, 1),
                'average_usefulness': round(avg_usefulness, 1),
                'average_quality': round(avg_quality, 1),
                'recent_comments': [
                    {
                        'comment': comment.comment,
                        'rating': comment.rating,
                        'user': comment.user.username if not comment.is_anonymous else 'Anonymous',
                        'created_at': comment.created_at,
                        'helpful_count': comment.helpful_count
                    }
                    for comment in recent_comments
                ],
                'recent_feedback': recent_feedback
            }
        except Exception as e:
            print(f"Error getting feedback summary: {e}")
            return {
                'total_reviews': 0,
                'average_rating': 0.0,
                'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'average_satisfaction': 0.0,
                'average_usefulness': 0.0,
                'average_quality': 0.0,
                'recent_comments': []
            }


class DatasetFeedbackHelpful(Document):
    """
    Model for tracking which users found feedback helpful.
    """
    
    feedback = ReferenceField('DatasetFeedback', required=True)
    user = ReferenceField('User', required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'dataset_feedback_helpful',
        'indexes': [
            ('feedback', 'user'),  # Unique index to prevent duplicate helpful votes
        ]
    }
