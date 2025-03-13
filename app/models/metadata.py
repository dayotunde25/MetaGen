from datetime import datetime
import json
from app.models import db

class MetadataQuality(db.Model):
    """Model to track metadata quality metrics and FAIR compliance"""
    __tablename__ = 'metadata_quality'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    
    # Core metadata quality metrics
    quality_score = db.Column(db.Float)  # Overall quality score (0-100)
    completeness = db.Column(db.Float)   # Completeness score (0-100)
    consistency = db.Column(db.Float)    # Consistency score (0-100)
    
    # FAIR compliance scores
    findable_score = db.Column(db.Float)       # Findable score (0-100)
    accessible_score = db.Column(db.Float)     # Accessible score (0-100)
    interoperable_score = db.Column(db.Float)  # Interoperable score (0-100)
    reusable_score = db.Column(db.Float)       # Reusable score (0-100)
    fair_compliant = db.Column(db.Boolean, default=False)
    
    # Schema.org compliance
    schema_org_compliant = db.Column(db.Boolean, default=False)
    schema_org_metadata = db.Column(db.Text)  # JSON representation of Schema.org metadata
    
    # Issues and recommendations
    issues = db.Column(db.Text)  # JSON list of issues
    recommendations = db.Column(db.Text)  # JSON list of recommendations
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, dataset_id, quality_score=0, completeness=0, consistency=0,
                 findable_score=0, accessible_score=0, interoperable_score=0, reusable_score=0,
                 fair_compliant=False, schema_org_compliant=False, schema_org_metadata=None,
                 issues=None, recommendations=None):
        self.dataset_id = dataset_id
        self.quality_score = quality_score
        self.completeness = completeness
        self.consistency = consistency
        self.findable_score = findable_score
        self.accessible_score = accessible_score
        self.interoperable_score = interoperable_score
        self.reusable_score = reusable_score
        self.fair_compliant = fair_compliant
        self.schema_org_compliant = schema_org_compliant
        self.schema_org_metadata = schema_org_metadata
        self.issues = json.dumps(issues) if issues else '[]'
        self.recommendations = json.dumps(recommendations) if recommendations else '[]'
    
    @property
    def issues_list(self):
        """Return issues as a list"""
        if not self.issues:
            return []
        return json.loads(self.issues)
    
    @issues_list.setter
    def issues_list(self, issues):
        """Set issues from a list"""
        self.issues = json.dumps(issues) if issues else '[]'
    
    @property
    def recommendations_list(self):
        """Return recommendations as a list"""
        if not self.recommendations:
            return []
        return json.loads(self.recommendations)
    
    @recommendations_list.setter
    def recommendations_list(self, recommendations):
        """Set recommendations from a list"""
        self.recommendations = json.dumps(recommendations) if recommendations else '[]'
    
    @property
    def schema_org_json(self):
        """Return schema.org metadata as JSON object"""
        if not self.schema_org_metadata:
            return {}
        return json.loads(self.schema_org_metadata)
    
    @schema_org_json.setter
    def schema_org_json(self, data):
        """Set schema.org metadata from JSON object"""
        self.schema_org_metadata = json.dumps(data) if data else '{}'
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'quality_score': self.quality_score,
            'completeness': self.completeness,
            'consistency': self.consistency,
            'findable_score': self.findable_score,
            'accessible_score': self.accessible_score,
            'interoperable_score': self.interoperable_score,
            'reusable_score': self.reusable_score,
            'fair_compliant': self.fair_compliant,
            'schema_org_compliant': self.schema_org_compliant,
            'issues': self.issues_list,
            'recommendations': self.recommendations_list,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f'<MetadataQuality for dataset_id={self.dataset_id}>'


class ProcessingQueue(db.Model):
    """Model to track dataset processing queue"""
    __tablename__ = 'processing_queue'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    status = db.Column(db.String(64), default='queued')  # queued, processing, completed, failed
    progress = db.Column(db.Float, default=0)  # Processing progress (0-100)
    priority = db.Column(db.Integer, default=1)  # Processing priority (higher is more important)
    error_message = db.Column(db.Text)  # Error message if processing failed
    started_at = db.Column(db.DateTime)  # When processing started
    completed_at = db.Column(db.DateTime)  # When processing completed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, dataset_id, status='queued', priority=1):
        self.dataset_id = dataset_id
        self.status = status
        self.priority = priority
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'status': self.status,
            'progress': self.progress,
            'priority': self.priority,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f'<ProcessingQueue for dataset_id={self.dataset_id}>'