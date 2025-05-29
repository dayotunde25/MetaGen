"""
Dataset model and related database operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from mongoengine import Document, StringField, IntField, DateTimeField, BooleanField, ReferenceField
from bson import ObjectId

class Dataset(Document):
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
        file_path: Path to the uploaded dataset file
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

    title = StringField(max_length=255, required=True)
    description = StringField()
    source = StringField(max_length=128, required=True)
    source_url = StringField(max_length=512)
    data_type = StringField(max_length=50)
    category = StringField(max_length=50)
    format = StringField(max_length=20)
    size = StringField(max_length=20)
    record_count = IntField()
    file_path = StringField(max_length=512)  # Path to uploaded file
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    status = StringField(max_length=20, default='pending')
    license = StringField(max_length=100)
    author = StringField(max_length=128)
    publisher = StringField(max_length=128)
    temporal_coverage = StringField(max_length=100)
    spatial_coverage = StringField()
    tags = StringField(max_length=255)
    keywords = StringField()
    schema_org = StringField()
    processed = BooleanField(default=False)
    update_frequency = StringField(max_length=50)
    standards = StringField(max_length=255)
    identifier = StringField(max_length=100)
    indexed = BooleanField(default=False)
    vocabulary = StringField(max_length=100)
    related_datasets = StringField()
    provenance = StringField()

    # Reference to User
    user = ReferenceField('User')

    meta = {
        'collection': 'datasets',
        'indexes': ['title', 'category', 'data_type', 'created_at']
    }

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
        dataset.save()
        return dataset

    @classmethod
    def find_by_id(cls, dataset_id: str) -> Optional['Dataset']:
        """
        Find a dataset by ID.

        Args:
            dataset_id: ID of the dataset to find

        Returns:
            Dataset instance or None
        """
        try:
            return cls.objects(id=dataset_id).first()
        except cls.DoesNotExist:
            return None

    @classmethod
    def get_all(cls) -> List['Dataset']:
        """
        Get all datasets.

        Returns:
            List of all datasets
        """
        return list(cls.objects.order_by('-created_at'))

    @classmethod
    def get_by_user(cls, user_id: str) -> List['Dataset']:
        """
        Get datasets for a specific user.

        Args:
            user_id: ID of the user

        Returns:
            List of datasets
        """
        return list(cls.objects(user=user_id).order_by('-created_at'))

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
        from mongoengine import Q

        search_filter = Q()

        if query:
            search_filter = (
                Q(title__icontains=query) |
                Q(description__icontains=query) |
                Q(tags__icontains=query) |
                Q(keywords__icontains=query)
            )

        if category:
            search_filter = search_filter & Q(category=category)

        if data_type:
            search_filter = search_filter & Q(data_type=data_type)

        return list(cls.objects(search_filter).order_by('-created_at'))

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

        self.updated_at = datetime.utcnow()
        self.save()
        return self

    def delete(self) -> bool:
        """
        Delete the dataset.

        Returns:
            True if successful
        """
        super().delete()
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'id': str(self.id),
            'title': self.title,
            'description': self.description,
            'source': self.source,
            'source_url': self.source_url,
            'data_type': self.data_type,
            'category': self.category,
            'format': self.format,
            'size': self.size,
            'record_count': self.record_count,
            'file_path': self.file_path,
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
            'user_id': str(self.user.id) if self.user else None
        }