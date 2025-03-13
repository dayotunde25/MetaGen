"""
Metadata models for datasets.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.models.base import Base, db


class MetadataQuality(Base):
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
    
    __tablename__ = 'metadata_quality'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    quality_score = Column(Float, nullable=False, default=0.0)
    completeness = Column(Float, nullable=False, default=0.0)
    consistency = Column(Float, nullable=False, default=0.0)
    accuracy = Column(Float, nullable=False, default=0.0)
    timeliness = Column(Float, nullable=False, default=0.0)
    conformity = Column(Float, nullable=False, default=0.0)
    integrity = Column(Float, nullable=False, default=0.0)
    findable_score = Column(Float, nullable=False, default=0.0)
    accessible_score = Column(Float, nullable=False, default=0.0)
    interoperable_score = Column(Float, nullable=False, default=0.0)
    reusable_score = Column(Float, nullable=False, default=0.0)
    fair_compliant = Column(Boolean, nullable=False, default=False)
    schema_org_compliant = Column(Boolean, nullable=False, default=False)
    issues = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship('Dataset', back_populates='metadata_quality')
    
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
    def get_by_dataset(cls, dataset_id: int) -> Optional['MetadataQuality']:
        """
        Get metadata quality for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            MetadataQuality instance or None
        """
        return db.session.query(cls).filter_by(dataset_id=dataset_id).first()
    
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
        db.session.add(quality)
        db.session.commit()
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
        
        db.session.commit()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
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


class ProcessingQueue(Base):
    """
    ProcessingQueue model for tracking dataset processing status.
    
    Attributes:
        id: Unique identifier
        dataset_id: ID of the dataset in the queue
        status: Processing status (pending, processing, completed, failed)
        priority: Priority level (1-10, higher is more important)
        started_at: When processing started
        completed_at: When processing completed
        error: Error message if processing failed
        progress: Processing progress percentage (0-100)
        estimated_completion_time: Estimated time to completion
    """
    
    __tablename__ = 'processing_queue'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    status = Column(String(20), default='pending')
    priority = Column(Integer, default=5)
    queued_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)
    progress = Column(Float, default=0.0)
    estimated_completion_time = Column(String(50), nullable=True)
    
    # Relationships
    dataset = relationship('Dataset', back_populates='processing_queue')
    
    @classmethod
    def get_queue(cls) -> List['ProcessingQueue']:
        """
        Get all items in the processing queue.
        
        Returns:
            List of ProcessingQueue items
        """
        return db.session.query(cls).order_by(cls.priority.desc(), cls.queued_at.asc()).all()
    
    @classmethod
    def get_by_dataset(cls, dataset_id: int) -> Optional['ProcessingQueue']:
        """
        Get queue item for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            ProcessingQueue instance or None
        """
        return db.session.query(cls).filter_by(dataset_id=dataset_id).first()
    
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
        db.session.add(queue_item)
        db.session.commit()
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
        
        db.session.commit()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'status': self.status,
            'priority': self.priority,
            'queued_at': self.queued_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'error': self.error,
            'progress': self.progress,
            'estimated_completion_time': self.estimated_completion_time
        }