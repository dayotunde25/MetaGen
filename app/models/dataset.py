from datetime import datetime
import json
from app.models import db

class Dataset(db.Model):
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    source_url = db.Column(db.String(512))
    source = db.Column(db.String(128))
    format = db.Column(db.String(64))
    data_type = db.Column(db.String(64))
    category = db.Column(db.String(64))
    tags = db.Column(db.String(255))
    size = db.Column(db.String(64))
    record_count = db.Column(db.Integer)
    file_path = db.Column(db.String(512))
    status = db.Column(db.String(64), default='pending')  # pending, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    metadata = db.relationship('MetadataQuality', backref='dataset', lazy='joined', uselist=False)
    processing = db.relationship('ProcessingQueue', backref='dataset', lazy='joined', uselist=False)
    
    def __init__(self, title, description=None, source_url=None, source=None, 
                format=None, data_type=None, category=None, tags=None, size=None, 
                record_count=None, user_id=None):
        self.title = title
        self.description = description
        self.source_url = source_url
        self.source = source
        self.format = format
        self.data_type = data_type
        self.category = category
        self.tags = json.dumps(tags) if isinstance(tags, list) else tags
        self.size = size
        self.record_count = record_count
        self.user_id = user_id
    
    @property
    def tags_list(self):
        """Return tags as a list"""
        if not self.tags:
            return []
        try:
            return json.loads(self.tags)
        except:
            return self.tags.split(',')
    
    @tags_list.setter
    def tags_list(self, tags):
        """Set tags from a list"""
        if isinstance(tags, list):
            self.tags = json.dumps(tags)
        else:
            self.tags = tags
    
    def to_dict(self):
        """Convert dataset to a dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'source_url': self.source_url,
            'source': self.source,
            'format': self.format,
            'data_type': self.data_type,
            'category': self.category,
            'tags': self.tags_list,
            'size': self.size,
            'record_count': self.record_count,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id
        }
    
    def __repr__(self):
        return f'<Dataset {self.title}>'