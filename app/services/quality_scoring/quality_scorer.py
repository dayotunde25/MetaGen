"""
Main quality scoring class for comprehensive dataset quality assessment.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality

logger = logging.getLogger(__name__)

class QualityScorer:
    """
    Main quality scorer class that orchestrates the assessment of dataset quality
    across multiple dimensions.

    This class:
    1. Coordinates different dimension scorers
    2. Calculates the overall quality score using weighted dimensions
    3. Generates quality reports with detailed metrics
    4. Provides recommendations for quality improvement
    """

    def __init__(self, dimension_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the quality scorer with dimension weights.

        Args:
            dimension_weights: Dictionary mapping dimension names to their weights.
                If None, default weights are used.
        """
        # Default weights for quality dimensions
        self.dimension_weights = dimension_weights or {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.15,
            'timeliness': 0.10,
            'conformity': 0.15,
            'integrity': 0.15
        }

        # Ensure weights sum to 1.0
        weight_sum = sum(self.dimension_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            logger.warning(f"Dimension weights sum to {weight_sum}, normalizing to 1.0")
            self.dimension_weights = {k: v/weight_sum for k, v in self.dimension_weights.items()}

        # Initialize dimension scorers
        self.dimension_scorers = {}

        # Metrics tracked for detailed reporting
        self.metrics = {}
        self.issues = []
        self.recommendations = []

    def register_dimension_scorer(self, dimension: str, scorer) -> None:
        """
        Register a scorer for a specific quality dimension.

        Args:
            dimension: Name of the quality dimension
            scorer: Scorer instance for that dimension
        """
        self.dimension_scorers[dimension] = scorer

    def score_dataset(self, dataset: Dataset, processed_data: Any) -> Dict[str, Any]:
        """
        Calculate quality scores for a dataset across all dimensions.

        Args:
            dataset: The dataset model instance
            processed_data: Processed data from the dataset

        Returns:
            Dictionary containing dimension scores, overall score, and recommendations
        """
        self.metrics = {}
        self.issues = []
        self.recommendations = []

        # Calculate scores for each dimension
        dimension_scores = {}

        for dimension, scorer in self.dimension_scorers.items():
            try:
                score, dimension_metrics = scorer.score(dataset, processed_data)
                dimension_scores[dimension] = score

                # Store detailed metrics
                self.metrics[dimension] = dimension_metrics

                # Collect issues and recommendations
                if hasattr(scorer, 'get_issues') and callable(scorer.get_issues):
                    self.issues.extend(scorer.get_issues())

                if hasattr(scorer, 'get_recommendations') and callable(scorer.get_recommendations):
                    self.recommendations.extend(scorer.get_recommendations())

            except Exception as e:
                logger.error(f"Error scoring {dimension} dimension: {str(e)}")
                dimension_scores[dimension] = 0.0

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)

        # Generate final result
        result = {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'metrics': self.metrics,
            'issues': self.issues,
            'recommendations': self.recommendations
        }

        return result

    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score from dimension scores.

        Args:
            dimension_scores: Dictionary of scores for each dimension

        Returns:
            Weighted overall quality score (0-100)
        """
        overall_score = 0.0

        for dimension, score in dimension_scores.items():
            if dimension in self.dimension_weights:
                overall_score += score * self.dimension_weights[dimension]

        return min(100.0, max(0.0, overall_score))

    def generate_quality_assessment(self, dataset: Dataset, processed_data: Any) -> Dict[str, Any]:
        """
        Generate a comprehensive quality assessment for a dataset.

        Args:
            dataset: The dataset model instance
            processed_data: Processed data from the dataset

        Returns:
            Dictionary with quality assessment results
        """
        # Score the dataset
        scoring_result = self.score_dataset(dataset, processed_data)

        # Add metadata for the assessment
        assessment = {
            'dataset_id': dataset.id,
            'assessment_timestamp': datetime.utcnow().isoformat(),
            'quality_score': scoring_result['overall_score'],
            'dimension_scores': scoring_result['dimension_scores'],
            'metrics': scoring_result['metrics'],
            'issues': scoring_result['issues'],
            'recommendations': scoring_result['recommendations']
        }

        return assessment

    def prepare_metadata_quality(self, dataset: Dataset, processed_data: Any) -> MetadataQuality:
        """
        Prepare a MetadataQuality instance based on quality assessment.

        Args:
            dataset: The dataset model instance
            processed_data: Processed data from the dataset

        Returns:
            Populated MetadataQuality instance ready to be saved
        """
        assessment = self.generate_quality_assessment(dataset, processed_data)

        # Create or update metadata quality record
        metadata_quality = MetadataQuality(
            dataset=dataset,  # Pass dataset object, not dataset.id
            quality_score=assessment['quality_score'],
            completeness=assessment['dimension_scores'].get('completeness', 0.0),
            consistency=assessment['dimension_scores'].get('consistency', 0.0),
            findable_score=0.0,  # Will be set in FAIR assessment
            accessible_score=0.0,  # Will be set in FAIR assessment
            interoperable_score=0.0,  # Will be set in FAIR assessment
            reusable_score=0.0,  # Will be set in FAIR assessment
            fair_compliant=False,  # Will be set in FAIR assessment
            schema_org_compliant=False,  # Will be set in Schema.org validation
            issues=str(assessment['issues']),
            recommendations=str(assessment['recommendations'])
        )

        return metadata_quality