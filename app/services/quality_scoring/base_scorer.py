"""
Base class for all quality dimension scorers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from app.models.dataset import Dataset


class BaseDimensionScorer(ABC):
    """
    Abstract base class that all dimension scorers must implement.
    
    This class defines the interface that all quality dimension scorers
    must adhere to, ensuring consistency across different scoring dimensions.
    """
    
    def __init__(self):
        """Initialize the scorer with any necessary configuration."""
        self.issues = []
        self.recommendations = []
    
    @abstractmethod
    def score(self, dataset: Dataset, processed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Score the dataset on this quality dimension.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data from the dataset
            
        Returns:
            Tuple of (dimension_score, detailed_metrics)
            - dimension_score: Float between 0-100
            - detailed_metrics: Dictionary of detailed metrics specific to this dimension
        """
        pass
    
    def get_issues(self) -> List[str]:
        """
        Get the list of issues identified during scoring.
        
        Returns:
            List of issue descriptions
        """
        return self.issues
    
    def get_recommendations(self) -> List[str]:
        """
        Get the list of recommendations to improve quality.
        
        Returns:
            List of recommendation descriptions
        """
        return self.recommendations
    
    def add_issue(self, issue: str) -> None:
        """
        Add an issue to the list.
        
        Args:
            issue: Description of the issue
        """
        self.issues.append(issue)
    
    def add_recommendation(self, recommendation: str) -> None:
        """
        Add a recommendation to the list.
        
        Args:
            recommendation: Description of the recommendation
        """
        self.recommendations.append(recommendation)
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize a score to be between 0 and 100.
        
        Args:
            score: Raw score
            
        Returns:
            Normalized score between 0 and 100
        """
        return min(100.0, max(0.0, score))