"""
Quality Assessment Service - Integrates quality scoring algorithms with application.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality
from app.services.quality_scoring import (
    QualityScorer,
    CompletenessScorer,
    ConsistencyScorer,
    AccuracyScorer,
    TimelinessScorer,
    ConformityScorer,
    IntegrityScorer
)

logger = logging.getLogger(__name__)


class QualityAssessmentService:
    """
    Service for assessing dataset quality and managing quality metadata.
    
    This service:
    1. Initializes and configures quality scoring algorithms
    2. Processes datasets to generate quality assessments
    3. Manages storage and retrieval of quality metadata
    4. Provides recommendations for improving dataset quality
    """
    
    def __init__(self):
        """Initialize quality assessment service."""
        # Create and configure quality scorer
        self.quality_scorer = self._initialize_quality_scorer()
        
    def _initialize_quality_scorer(self) -> QualityScorer:
        """
        Initialize and configure the quality scorer with dimension scorers.
        
        Returns:
            Configured QualityScorer instance
        """
        # Create main scorer with default dimension weights
        quality_scorer = QualityScorer()
        
        # Register dimension scorers
        quality_scorer.register_dimension_scorer('completeness', CompletenessScorer())
        quality_scorer.register_dimension_scorer('consistency', ConsistencyScorer())
        quality_scorer.register_dimension_scorer('accuracy', AccuracyScorer())
        quality_scorer.register_dimension_scorer('timeliness', TimelinessScorer())
        quality_scorer.register_dimension_scorer('conformity', ConformityScorer())
        quality_scorer.register_dimension_scorer('integrity', IntegrityScorer())
        
        return quality_scorer
    
    def assess_dataset_quality(self, dataset: Dataset, processed_data: Any) -> MetadataQuality:
        """
        Assess dataset quality and store results.
        
        Args:
            dataset: The dataset to assess
            processed_data: Processed data from the dataset
            
        Returns:
            MetadataQuality instance with assessment results
        """
        try:
            # Generate quality assessment
            quality_assessment = self.quality_scorer.generate_quality_assessment(dataset, processed_data)
            
            # Get or create metadata quality record
            metadata_quality = MetadataQuality.get_by_dataset(dataset.id)
            
            if metadata_quality:
                # Update existing record
                metadata_quality.update(
                    quality_score=quality_assessment['quality_score'],
                    completeness=quality_assessment['dimension_scores'].get('completeness', 0.0),
                    consistency=quality_assessment['dimension_scores'].get('consistency', 0.0),
                    accuracy=quality_assessment['dimension_scores'].get('accuracy', 0.0),
                    timeliness=quality_assessment['dimension_scores'].get('timeliness', 0.0),
                    conformity=quality_assessment['dimension_scores'].get('conformity', 0.0),
                    integrity=quality_assessment['dimension_scores'].get('integrity', 0.0),
                    issues=json.dumps(quality_assessment['issues']),
                    recommendations=json.dumps(quality_assessment['recommendations'])
                )
            else:
                # Create new record
                metadata_quality = MetadataQuality.create(
                    dataset_id=dataset.id,
                    quality_score=quality_assessment['quality_score'],
                    completeness=quality_assessment['dimension_scores'].get('completeness', 0.0),
                    consistency=quality_assessment['dimension_scores'].get('consistency', 0.0),
                    accuracy=quality_assessment['dimension_scores'].get('accuracy', 0.0),
                    timeliness=quality_assessment['dimension_scores'].get('timeliness', 0.0),
                    conformity=quality_assessment['dimension_scores'].get('conformity', 0.0),
                    integrity=quality_assessment['dimension_scores'].get('integrity', 0.0),
                    issues=json.dumps(quality_assessment['issues']),
                    recommendations=json.dumps(quality_assessment['recommendations'])
                )
            
            logger.info(f"Quality assessment completed for dataset {dataset.id} with score {quality_assessment['quality_score']}")
            return metadata_quality
            
        except Exception as e:
            logger.error(f"Error assessing quality for dataset {dataset.id}: {str(e)}")
            raise
    
    def assess_fair_compliance(self, dataset: Dataset, metadata_quality: MetadataQuality) -> Dict[str, Any]:
        """
        Assess FAIR compliance for a dataset.
        
        Args:
            dataset: The dataset to assess
            metadata_quality: Existing metadata quality record
            
        Returns:
            Dictionary with FAIR assessment results
        """
        try:
            # Create a ConformityScorer just for FAIR assessment
            conformity_scorer = ConformityScorer()
            
            # Calculate FAIR compliance
            fair_score, fair_details = conformity_scorer._calculate_fair_compliance(dataset)
            
            # Determine if FAIR compliant (minimum threshold of 70%)
            fair_compliant = fair_score >= 70.0
            
            # Update metadata quality record
            metadata_quality.update(
                findable_score=fair_details['findable'],
                accessible_score=fair_details['accessible'],
                interoperable_score=fair_details['interoperable'],
                reusable_score=fair_details['reusable'],
                fair_compliant=fair_compliant
            )
            
            return {
                'fair_score': fair_score,
                'fair_details': fair_details,
                'fair_compliant': fair_compliant
            }
            
        except Exception as e:
            logger.error(f"Error assessing FAIR compliance for dataset {dataset.id}: {str(e)}")
            raise
    
    def assess_schema_org_compliance(self, dataset: Dataset, metadata_quality: MetadataQuality) -> Dict[str, Any]:
        """
        Assess Schema.org compliance for a dataset.
        
        Args:
            dataset: The dataset to assess
            metadata_quality: Existing metadata quality record
            
        Returns:
            Dictionary with Schema.org assessment results
        """
        try:
            # Create a ConformityScorer just for Schema.org assessment
            conformity_scorer = ConformityScorer()
            
            # Calculate Schema.org compliance
            schema_org_score = conformity_scorer._calculate_schema_org_compliance(dataset)
            
            # Determine if Schema.org compliant (minimum threshold of 70%)
            schema_org_compliant = schema_org_score >= 70.0
            
            # Update metadata quality record
            metadata_quality.update(
                schema_org_compliant=schema_org_compliant
            )
            
            return {
                'schema_org_score': schema_org_score,
                'schema_org_compliant': schema_org_compliant
            }
            
        except Exception as e:
            logger.error(f"Error assessing Schema.org compliance for dataset {dataset.id}: {str(e)}")
            raise
    
    def get_improvement_recommendations(self, dataset_id: int) -> List[str]:
        """
        Get recommendations for improving dataset quality.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            List of recommendation strings
        """
        metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
        
        if not metadata_quality:
            return ["Generate a quality assessment first to get recommendations."]
        
        return metadata_quality.recommendations_list
    
    def get_quality_assessment(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """
        Get quality assessment for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary with quality assessment data or None
        """
        metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
        
        if not metadata_quality:
            return None
        
        return metadata_quality.to_dict()
    
    def get_fair_scores(self, dataset_id: int) -> Optional[Dict[str, float]]:
        """
        Get FAIR scores for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary with FAIR scores or None
        """
        metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
        
        if not metadata_quality:
            return None
        
        return metadata_quality.fair_scores
    
    def get_dimension_scores(self, dataset_id: int) -> Optional[Dict[str, float]]:
        """
        Get dimension scores for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary with dimension scores or None
        """
        metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
        
        if not metadata_quality:
            return None
        
        return metadata_quality.dimension_scores
        
    def get_overall_quality_score(self, dataset_id: int) -> float:
        """
        Get overall quality score for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Quality score (0-100) or 0 if not assessed
        """
        metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
        
        if not metadata_quality:
            return 0.0
        
        return metadata_quality.quality_score
    
    def get_quality_issues(self, dataset_id: int) -> List[str]:
        """
        Get quality issues for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            List of issue strings
        """
        metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
        
        if not metadata_quality:
            return []
        
        return metadata_quality.issues_list


# Singleton instance for application-wide use
quality_assessment_service = QualityAssessmentService()