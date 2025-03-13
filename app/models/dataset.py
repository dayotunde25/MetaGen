"""
Dataset model and related database operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import Base, db


class Dataset(Base):
    """
    Dataset model representing metadata for a dataset.
    
    Attributes:
        id: Unique identifier for the dataset
        title: Title of the dataset
        description: Description of the dataset
        source: Source of the dataset (organization or entity)
        source_url: URL where the dataset can be accessed
        data_type: Type of data (tabular, text, image, etc.)
        category: Category or domain of the dataset
        format: Format of the dataset (CSV, JSON, etc.)
        size: Size of the dataset (e.g., "5 MB")
        record_count: Number of records in the dataset
        created_at: Timestamp when the dataset was created
        updated_at: Timestamp when the dataset was last updated
        status: Processing status of the dataset
        license: License under which the dataset is available
        author: Author of the dataset
        publisher: Publisher of the dataset
        temporal_coverage: Time period covered by the dataset
        spatial_coverage: Geographic area covered by the dataset
        tags: Tags associated with the dataset (comma-separated)
        keywords: Keywords describing the dataset content
        schema_org: Schema.org metadata as JSON
        processed: Whether the dataset has been processed
        user_id: ID of the user who added the dataset
    """
    
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String(128), nullable=False)
    source_url = Column(String(512), nullable=True)
    data_type = Column(String(50), nullable=True)
    category = Column(String(50), nullable=True)
    format = Column(String(20), nullable=True)
    size = Column(String(20), nullable=True)
    record_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), default='pending')
    license = Column(String(100), nullable=True)
    author = Column(String(128), nullable=True)
    publisher = Column(String(128), nullable=True)
    temporal_coverage = Column(String(100), nullable=True)
    spatial_coverage = Column(Text, nullable=True)
    tags = Column(String(255), nullable=True)
    keywords = Column(Text, nullable=True)
    schema_org = Column(Text, nullable=True)
    processed = Column(Boolean, default=False)
    update_frequency = Column(String(50), nullable=True)
    standards = Column(String(255), nullable=True)
    identifier = Column(String(100), nullable=True)
    indexed = Column(Boolean, default=False)
    vocabulary = Column(String(100), nullable=True)
    related_datasets = Column(Text, nullable=True)
    provenance = Column(Text, nullable=True)
    
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='datasets')
    metadata_quality = relationship('MetadataQuality', back_populates='dataset', uselist=False)
    processing_queue = relationship('ProcessingQueue', back_populates='dataset', uselist=False)
    
    @property
    def tags_list(self) -> List[str]:
        """
        Get tags as a list.
        
        Returns:
            List of tag strings
        """
        if not self.tags:
            return []
        return [tag.strip() for tag in self.tags.split(',')]
    
    @property
    def keywords_list(self) -> List[str]:
        """
        Get keywords as a list.
        
        Returns:
            List of keyword strings
        """
        if not self.keywords:
            return []
        return [kw.strip() for kw in self.keywords.split(',')]
    
    @property
    def schema_org_dict(self) -> Optional[Dict[str, Any]]:
        """
        Get Schema.org metadata as a dictionary.
        
        Returns:
            Dictionary of Schema.org metadata or None
        """
        import json
        if not self.schema_org:
            return None
        try:
            return json.loads(self.schema_org)
        except json.JSONDecodeError:
            return None
    
    @classmethod
    def create(cls, **kwargs) -> 'Dataset':
        """
        Create a new dataset.
        
        Args:
            **kwargs: Keyword arguments with dataset attributes
            
        Returns:
            New Dataset instance
        """
        dataset = cls(**kwargs)
        db.session.add(dataset)
        db.session.commit()
        return dataset
    
    @classmethod
    def find_by_id(cls, dataset_id: int) -> Optional['Dataset']:
        """
        Find a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset to find
            
        Returns:
            Dataset instance or None
        """
        return db.session.query(cls).filter_by(id=dataset_id).first()
    
    @classmethod
    def get_all(cls) -> List['Dataset']:
        """
        Get all datasets.
        
        Returns:
            List of all datasets
        """
        return db.session.query(cls).order_by(cls.created_at.desc()).all()
    
    @classmethod
    def get_by_user(cls, user_id: int) -> List['Dataset']:
        """
        Get datasets for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of datasets
        """
        return db.session.query(cls).filter_by(user_id=user_id).order_by(cls.created_at.desc()).all()
    
    @classmethod
    def search(cls, query: str, category: Optional[str] = None, data_type: Optional[str] = None) -> List['Dataset']:
        """
        Search datasets.
        
        Args:
            query: Search query string
            category: Filter by category
            data_type: Filter by data type
            
        Returns:
            List of matching datasets
        """
        search_query = db.session.query(cls)
        
        if query:
            search_query = search_query.filter(
                (cls.title.ilike(f'%{query}%')) |
                (cls.description.ilike(f'%{query}%')) |
                (cls.tags.ilike(f'%{query}%')) |
                (cls.keywords.ilike(f'%{query}%'))
            )
        
        if category:
            search_query = search_query.filter(cls.category == category)
        
        if data_type:
            search_query = search_query.filter(cls.data_type == data_type)
        
        return search_query.order_by(cls.created_at.desc()).all()
    
    def update(self, **kwargs) -> 'Dataset':
        """
        Update the dataset.
        
        Args:
            **kwargs: Keyword arguments with updated attributes
            
        Returns:
            Updated Dataset instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        db.session.commit()
        return self
    
    def delete(self) -> bool:
        """
        Delete the dataset.
        
        Returns:
            True if successful
        """
        db.session.delete(self)
        db.session.commit()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'source': self.source,
            'source_url': self.source_url,
            'data_type': self.data_type,
            'category': self.category,
            'format': self.format,
            'size': self.size,
            'record_count': self.record_count,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'status': self.status,
            'license': self.license,
            'author': self.author,
            'publisher': self.publisher,
            'temporal_coverage': self.temporal_coverage,
            'spatial_coverage': self.spatial_coverage,
            'tags': self.tags_list,
            'keywords': self.keywords_list,
            'processed': self.processed,
            'update_frequency': self.update_frequency,
            'standards': self.standards,
            'user_id': self.user_id
        }