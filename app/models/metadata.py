"""
Metadata models for datasets.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from mongoengine import Document, StringField, FloatField, BooleanField, DateTimeField, ReferenceField, IntField


class MetadataQuality(Document):
    """
    MetadataQuality model representing quality assessment for a dataset.

    Attributes:
        id: Unique identifier
        dataset_id: ID of the dataset this quality assessment relates to
        quality_score: Overall quality score (0-100)
        completeness: Completeness score (0-100)
        consistency: Consistency score (0-100)
        accuracy: Accuracy score (0-100)
        timeliness: Timeliness score (0-100)
        conformity: Conformity score (0-100)
        integrity: Integrity score (0-100)
        findable_score: FAIR Findable score (0-100)
        accessible_score: FAIR Accessible score (0-100)
        interoperable_score: FAIR Interoperable score (0-100)
        reusable_score: FAIR Reusable score (0-100)
        fair_compliant: Whether the dataset is FAIR compliant
        schema_org_compliant: Whether the dataset is Schema.org compliant
        issues: List of identified issues (JSON or comma-separated string)
        recommendations: List of recommendations for improvement
        assessment_date: Date of the quality assessment
    """

    dataset = ReferenceField('Dataset', required=True)
    quality_score = FloatField(default=0.0)
    completeness = FloatField(default=0.0)
    consistency = FloatField(default=0.0)
    accuracy = FloatField(default=0.0)
    timeliness = FloatField(default=0.0)
    conformity = FloatField(default=0.0)
    integrity = FloatField(default=0.0)
    findable_score = FloatField(default=0.0)
    accessible_score = FloatField(default=0.0)
    interoperable_score = FloatField(default=0.0)
    reusable_score = FloatField(default=0.0)
    fair_compliant = BooleanField(default=False)
    schema_org_compliant = BooleanField(default=False)
    issues = StringField()
    recommendations = StringField()
    assessment_date = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'metadata_quality',
        'indexes': ['dataset', 'assessment_date']
    }

    @property
    def fair_scores(self) -> Dict[str, float]:
        """
        Get FAIR scores as a dictionary.

        Returns:
            Dictionary with FAIR component scores
        """
        return {
            'findable': self.findable_score,
            'accessible': self.accessible_score,
            'interoperable': self.interoperable_score,
            'reusable': self.reusable_score,
            'overall': (self.findable_score + self.accessible_score +
                        self.interoperable_score + self.reusable_score) / 4.0
        }

    @property
    def dimension_scores(self) -> Dict[str, float]:
        """
        Get dimension scores as a dictionary.

        Returns:
            Dictionary with dimension scores
        """
        return {
            'completeness': self.completeness,
            'consistency': self.consistency,
            'accuracy': self.accuracy,
            'timeliness': self.timeliness,
            'conformity': self.conformity,
            'integrity': self.integrity
        }

    @property
    def issues_list(self) -> List[str]:
        """
        Get issues as a list.

        Returns:
            List of issue strings
        """
        import json

        if not self.issues:
            return []

        # Try to parse as JSON first
        try:
            return json.loads(self.issues)
        except json.JSONDecodeError:
            # Fall back to comma-separated string
            return [issue.strip() for issue in self.issues.split(',')]

    @property
    def recommendations_list(self) -> List[str]:
        """
        Get recommendations as a list.

        Returns:
            List of recommendation strings
        """
        import json

        if not self.recommendations:
            return []

        # Try to parse as JSON first
        try:
            return json.loads(self.recommendations)
        except json.JSONDecodeError:
            # Fall back to comma-separated string
            return [rec.strip() for rec in self.recommendations.split(',')]

    @classmethod
    def get_by_dataset(cls, dataset_id: str) -> Optional['MetadataQuality']:
        """
        Get metadata quality for a dataset.

        Args:
            dataset_id: ID of the dataset

        Returns:
            MetadataQuality instance or None
        """
        try:
            return cls.objects(dataset=dataset_id).first()
        except cls.DoesNotExist:
            return None

    @classmethod
    def create(cls, **kwargs) -> 'MetadataQuality':
        """
        Create a new metadata quality record.

        Args:
            **kwargs: Keyword arguments with attributes

        Returns:
            New MetadataQuality instance
        """
        quality = cls(**kwargs)
        quality.save()
        return quality

    def update(self, **kwargs) -> 'MetadataQuality':
        """
        Update the metadata quality record.

        Args:
            **kwargs: Keyword arguments with updated attributes

        Returns:
            Updated MetadataQuality instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.save()
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'id': str(self.id),
            'dataset_id': str(self.dataset.id) if self.dataset else None,
            'quality_score': self.quality_score,
            'completeness': self.completeness,
            'consistency': self.consistency,
            'accuracy': self.accuracy,
            'timeliness': self.timeliness,
            'conformity': self.conformity,
            'integrity': self.integrity,
            'fair_scores': self.fair_scores,
            'fair_compliant': self.fair_compliant,
            'schema_org_compliant': self.schema_org_compliant,
            'issues': self.issues_list,
            'recommendations': self.recommendations_list,
            'assessment_date': self.assessment_date
        }


class ProcessingQueue(Document):
    """
    ProcessingQueue model for tracking dataset processing status.

    Attributes:
        dataset: Reference to the dataset in the queue
        status: Processing status (pending, processing, completed, failed)
        priority: Priority level (1-10, higher is more important)
        started_at: When processing started
        completed_at: When processing completed
        error: Error message if processing failed
        progress: Processing progress percentage (0-100)
        message: Current processing status message
        estimated_completion_time: Estimated time to completion
    """

    dataset = ReferenceField('Dataset', required=True)
    status = StringField(max_length=20, default='pending')
    priority = IntField(default=5)
    queued_at = DateTimeField(default=datetime.utcnow)
    started_at = DateTimeField()
    completed_at = DateTimeField()
    error = StringField()
    progress = FloatField(default=0.0)
    message = StringField()  # Current processing message
    estimated_completion_time = StringField(max_length=50)

    meta = {
        'collection': 'processing_queue',
        'indexes': ['dataset', 'status', 'priority', 'queued_at']
    }

    @classmethod
    def get_queue(cls) -> List['ProcessingQueue']:
        """
        Get all items in the processing queue.

        Returns:
            List of ProcessingQueue items
        """
        return list(cls.objects.order_by('-priority', 'queued_at'))

    @classmethod
    def get_by_dataset(cls, dataset_id: str) -> Optional['ProcessingQueue']:
        """
        Get queue item for a dataset.

        Args:
            dataset_id: ID of the dataset

        Returns:
            ProcessingQueue instance or None
        """
        try:
            return cls.objects(dataset=dataset_id).first()
        except cls.DoesNotExist:
            return None

    @classmethod
    def create(cls, **kwargs) -> 'ProcessingQueue':
        """
        Create a new processing queue item.

        Args:
            **kwargs: Keyword arguments with attributes

        Returns:
            New ProcessingQueue instance
        """
        queue_item = cls(**kwargs)
        queue_item.save()
        return queue_item

    def update(self, **kwargs) -> 'ProcessingQueue':
        """
        Update the queue item.

        Args:
            **kwargs: Keyword arguments with updated attributes

        Returns:
            Updated ProcessingQueue instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.save()
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'id': str(self.id),
            'dataset_id': str(self.dataset.id) if self.dataset else None,
            'status': self.status,
            'priority': self.priority,
            'queued_at': self.queued_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'error': self.error,
            'progress': self.progress,
            'message': self.message,
            'estimated_completion_time': self.estimated_completion_time
        }