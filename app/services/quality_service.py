"""
Quality Service - Advanced algorithms for dataset quality assessment
"""
import json
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality


class QualityService:
    """Service for advanced dataset quality assessment"""

    def __init__(self):
        """Initialize quality service"""
        # Define weights for different quality dimensions
        self.dimension_weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.15,
            'timeliness': 0.10,
            'fairness': 0.30
        }
        
        # Define weights for FAIR principles
        self.fair_weights = {
            'findable': 0.25,
            'accessible': 0.25,
            'interoperable': 0.25,
            'reusable': 0.25
        }
        
        # Define essential metadata fields and their importance
        self.essential_fields = {
            'title': 1.0,
            'description': 0.9,
            'source': 0.8,
            'source_url': 0.7,
            'data_type': 0.8,
            'category': 0.8,
            'tags': 0.7,
            'format': 0.6,
            'size': 0.5,
            'record_count': 0.5
        }
        
        # Schema.org required fields for Dataset type
        self.schema_required_fields = [
            'name',
            'description',
            'url',
            'creator',
            'datePublished'
        ]
        
        # Schema.org recommended fields for Dataset type
        self.schema_recommended_fields = [
            'keywords',
            'license',
            'temporalCoverage',
            'spatialCoverage',
            'variableMeasured',
            'distribution',
            'publisher',
            'version'
        ]

    def assess_quality(self, dataset: Dataset, processed_data: Any = None, 
                       schema_org_metadata: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive quality assessment on a dataset
        
        Args:
            dataset: The dataset to assess
            processed_data: The processed dataset content (optional)
            schema_org_metadata: Generated Schema.org metadata (optional)
            
        Returns:
            Dictionary with quality scores and assessment details
        """
        # Initialize quality assessment
        assessment = {
            'quality_score': 0,
            'completeness': 0,
            'consistency': 0,
            'accuracy': 0,
            'timeliness': 0,
            'findable_score': 0,
            'accessible_score': 0,
            'interoperable_score': 0,
            'reusable_score': 0,
            'fair_compliant': False,
            'schema_org_compliant': False,
            'issues': [],
            'recommendations': []
        }
        
        # Assess completeness
        completeness_score, missing_fields = self._assess_completeness(dataset)
        assessment['completeness'] = completeness_score
        
        # Add issues and recommendations for completeness
        if missing_fields:
            assessment['issues'].append(f"Missing essential metadata fields: {', '.join(missing_fields)}")
            assessment['recommendations'].append(f"Add values for missing fields: {', '.join(missing_fields)}")
        
        # Assess consistency if processed data is available
        if processed_data:
            consistency_score, consistency_issues = self._assess_consistency(dataset, processed_data)
            assessment['consistency'] = consistency_score
            
            # Add consistency issues
            for issue in consistency_issues:
                assessment['issues'].append(issue)
                assessment['recommendations'].append(self._generate_recommendation_for_issue(issue))
        else:
            assessment['consistency'] = 50  # Default value when data not available
            assessment['issues'].append("Dataset content not available for consistency assessment")
            assessment['recommendations'].append("Process the dataset content to enable detailed consistency analysis")
        
        # Assess accuracy
        accuracy_score, accuracy_issues = self._assess_accuracy(dataset, processed_data)
        assessment['accuracy'] = accuracy_score
        
        for issue in accuracy_issues:
            assessment['issues'].append(issue)
            assessment['recommendations'].append(self._generate_recommendation_for_issue(issue))
        
        # Assess timeliness
        timeliness_score = self._assess_timeliness(dataset)
        assessment['timeliness'] = timeliness_score
        
        # Assess FAIR principles
        fair_assessment = self._assess_fair_principles(dataset, schema_org_metadata)
        assessment.update(fair_assessment)
        
        # Check Schema.org compliance
        if schema_org_metadata:
            schema_compliant, schema_issues = self._assess_schema_org_compliance(schema_org_metadata)
            assessment['schema_org_compliant'] = schema_compliant
            
            for issue in schema_issues:
                assessment['issues'].append(issue)
                assessment['recommendations'].append(self._generate_recommendation_for_issue(issue))
        
        # Calculate overall quality score (weighted average of all dimensions)
        assessment['quality_score'] = self._calculate_overall_score(assessment)
        
        # Determine FAIR compliance (requires minimum scores in all areas)
        assessment['fair_compliant'] = self._is_fair_compliant(assessment)
        
        return assessment

    def _assess_completeness(self, dataset: Dataset) -> Tuple[float, List[str]]:
        """
        Assess metadata completeness based on essential fields
        
        Returns:
            Tuple of (completeness score, list of missing fields)
        """
        missing_fields = []
        total_weight = sum(self.essential_fields.values())
        current_weight = 0
        
        # Convert dataset to dictionary
        dataset_dict = dataset.to_dict() if hasattr(dataset, 'to_dict') else vars(dataset)
        
        for field, weight in self.essential_fields.items():
            if field in dataset_dict and dataset_dict[field]:
                current_weight += weight
            else:
                missing_fields.append(field)
        
        completeness_score = (current_weight / total_weight) * 100
        return completeness_score, missing_fields

    def _assess_consistency(self, dataset: Dataset, processed_data: Any) -> Tuple[float, List[str]]:
        """
        Assess data consistency based on processed data
        
        Returns:
            Tuple of (consistency score, list of consistency issues)
        """
        issues = []
        
        # Skip consistency checks if no processed data
        if not processed_data:
            return 50, ["Cannot assess consistency without processed data"]
        
        # Initialize sub-scores
        internal_consistency = 100
        metadata_data_consistency = 100
        schema_consistency = 100
        
        # Check if processed data format matches dataset.format
        if hasattr(dataset, 'format') and dataset.format:
            if processed_data.get('detected_format') and dataset.format.lower() != processed_data.get('detected_format').lower():
                issues.append(f"Format inconsistency: Metadata states '{dataset.format}' but detected format is '{processed_data.get('detected_format')}'")
                metadata_data_consistency -= 25
        
        # Check if record count is consistent
        if hasattr(dataset, 'record_count') and dataset.record_count:
            if processed_data.get('record_count') and dataset.record_count != processed_data.get('record_count'):
                issues.append(f"Record count inconsistency: Metadata states {dataset.record_count} records but actual count is {processed_data.get('record_count')}")
                metadata_data_consistency -= 25
        
        # Check for internal data consistency issues
        if processed_data.get('structure_issues'):
            for issue in processed_data.get('structure_issues'):
                issues.append(f"Data structure issue: {issue}")
                internal_consistency -= (20 * len(processed_data.get('structure_issues')))
                internal_consistency = max(0, internal_consistency)
        
        # Calculate final consistency score
        consistency_score = (internal_consistency * 0.4) + (metadata_data_consistency * 0.4) + (schema_consistency * 0.2)
        
        return consistency_score, issues

    def _assess_accuracy(self, dataset: Dataset, processed_data: Any) -> Tuple[float, List[str]]:
        """
        Assess data accuracy based on data quality checks
        
        Returns:
            Tuple of (accuracy score, list of accuracy issues)
        """
        issues = []
        accuracy_score = 80  # Start with a default score
        
        # Skip accuracy checks if no processed data
        if not processed_data:
            return accuracy_score, ["Cannot assess accuracy without processed data"]
        
        # Check for missing values if we have processed data with statistics
        if processed_data.get('statistics') and 'missing_value_rate' in processed_data['statistics']:
            missing_rate = processed_data['statistics']['missing_value_rate']
            if missing_rate > 0.2:  # More than 20% missing values
                issues.append(f"High missing value rate: {missing_rate:.1%} of values are missing")
                accuracy_score -= 20
            elif missing_rate > 0.1:  # 10-20% missing values
                issues.append(f"Moderate missing value rate: {missing_rate:.1%} of values are missing")
                accuracy_score -= 10
                
        # Check for outliers if we have processed data with statistics
        if processed_data.get('statistics') and 'outlier_rate' in processed_data['statistics']:
            outlier_rate = processed_data['statistics']['outlier_rate']
            if outlier_rate > 0.1:  # More than 10% outliers
                issues.append(f"High outlier rate: {outlier_rate:.1%} of values are potential outliers")
                accuracy_score -= 15
            elif outlier_rate > 0.05:  # 5-10% outliers
                issues.append(f"Moderate outlier rate: {outlier_rate:.1%} of values are potential outliers")
                accuracy_score -= 10
        
        # Check for data type consistency if available
        if processed_data.get('statistics') and 'type_consistency' in processed_data['statistics']:
            type_consistency = processed_data['statistics']['type_consistency']
            if type_consistency < 0.9:  # Less than 90% type consistency
                issues.append(f"Data type inconsistency: {(1-type_consistency):.1%} of values have unexpected types")
                accuracy_score -= 20
                
        # Ensure score stays in range 0-100
        accuracy_score = max(0, min(100, accuracy_score))
        
        return accuracy_score, issues

    def _assess_timeliness(self, dataset: Dataset) -> float:
        """
        Assess timeliness of the dataset based on creation and update dates
        
        Returns:
            Timeliness score (0-100)
        """
        # Default score
        timeliness_score = 70
        
        # Check if creation date exists
        if not hasattr(dataset, 'created_at') or not dataset.created_at:
            return timeliness_score
            
        # Get current date
        now = datetime.utcnow()
        
        # Calculate dataset age in days
        dataset_age = (now - dataset.created_at).days
        
        # Adjust score based on age (newer datasets get higher scores)
        if dataset_age < 30:  # Less than a month old
            timeliness_score = 100
        elif dataset_age < 90:  # Less than 3 months old
            timeliness_score = 90
        elif dataset_age < 180:  # Less than 6 months old
            timeliness_score = 80
        elif dataset_age < 365:  # Less than a year old
            timeliness_score = 70
        elif dataset_age < 730:  # Less than 2 years old
            timeliness_score = 60
        else:  # More than 2 years old
            timeliness_score = 50
            
        # Check if update date exists and is more recent than creation date
        if hasattr(dataset, 'updated_at') and dataset.updated_at:
            update_age = (now - dataset.updated_at).days
            if update_age < dataset_age:
                # Boost score if recently updated
                if update_age < 30:
                    timeliness_score = min(100, timeliness_score + 20)
                elif update_age < 90:
                    timeliness_score = min(100, timeliness_score + 15)
                elif update_age < 180:
                    timeliness_score = min(100, timeliness_score + 10)
        
        return timeliness_score

    def _assess_fair_principles(self, dataset: Dataset, schema_org_metadata: Optional[Dict] = None) -> Dict:
        """
        Assess compliance with FAIR principles
        
        Returns:
            Dictionary with FAIR assessment scores
        """
        assessment = {
            'findable_score': 0,
            'accessible_score': 0,
            'interoperable_score': 0,
            'reusable_score': 0
        }
        
        # Assess Findable
        assessment['findable_score'] = self._assess_findability(dataset, schema_org_metadata)
        
        # Assess Accessible
        assessment['accessible_score'] = self._assess_accessibility(dataset, schema_org_metadata)
        
        # Assess Interoperable
        assessment['interoperable_score'] = self._assess_interoperability(dataset, schema_org_metadata)
        
        # Assess Reusable
        assessment['reusable_score'] = self._assess_reusability(dataset, schema_org_metadata)
        
        return assessment
        
    def _assess_findability(self, dataset: Dataset, schema_org_metadata: Optional[Dict] = None) -> float:
        """
        Assess findability score based on metadata completeness and identifiers
        
        Returns:
            Findability score (0-100)
        """
        score = 0
        max_score = 100
        
        # Check for essential metadata that affects findability
        if hasattr(dataset, 'title') and dataset.title:
            score += 20
        
        if hasattr(dataset, 'description') and dataset.description:
            # Add more points for longer, more detailed descriptions
            if len(dataset.description) > 100:
                score += 15
            else:
                score += 10
        
        # Check for keywords/tags
        if hasattr(dataset, 'tags') and dataset.tags:
            # More points for multiple tags
            tags_list = dataset.tags.split(',') if isinstance(dataset.tags, str) else []
            if len(tags_list) >= 3:
                score += 15
            else:
                score += 10
        
        # Check for category
        if hasattr(dataset, 'category') and dataset.category:
            score += 10
            
        # Check for data type
        if hasattr(dataset, 'data_type') and dataset.data_type:
            score += 10
            
        # Check for persistent identifier
        if hasattr(dataset, 'doi') and dataset.doi:
            score += 20
        elif hasattr(dataset, 'source_url') and dataset.source_url:
            # URL is not as good as DOI but better than nothing
            score += 10
            
        # Check Schema.org metadata for additional findability factors
        if schema_org_metadata:
            # Check for additional identifiers in schema
            if 'identifier' in schema_org_metadata:
                score += 5
                
            # Check for keywords in schema if not already checked
            if 'keywords' in schema_org_metadata and not (hasattr(dataset, 'tags') and dataset.tags):
                score += 5
                
            # Check for structured fields that improve findability
            if 'variableMeasured' in schema_org_metadata:
                score += 5
                
        # Ensure score doesn't exceed maximum
        return min(score, max_score)
        
    def _assess_accessibility(self, dataset: Dataset, schema_org_metadata: Optional[Dict] = None) -> float:
        """
        Assess accessibility score based on access methods and protocols
        
        Returns:
            Accessibility score (0-100)
        """
        score = 0
        max_score = 100
        
        # Check for URL to access the dataset
        if hasattr(dataset, 'source_url') and dataset.source_url:
            # Basic access URL
            score += 30
            
            # Check if URL uses HTTPS (more secure protocol)
            if dataset.source_url.startswith('https://'):
                score += 10
            
            # Check if URL is from a known repository domain
            known_repositories = ['zenodo.org', 'figshare.com', 'datadryad.org', 'dataverse.org', 
                                 'osf.io', 'data.gov', 'kaggle.com', 'github.com']
            if any(repo in dataset.source_url for repo in known_repositories):
                score += 15
        else:
            # No URL severely impacts accessibility
            score += 10  # Small base score
        
        # Check for access information in metadata
        if hasattr(dataset, 'access_rights') and dataset.access_rights:
            score += 15
        
        # Check for documentation about how to access
        if hasattr(dataset, 'access_instructions') and dataset.access_instructions:
            score += 15
        
        # Check Schema.org metadata for distribution information
        if schema_org_metadata:
            if 'distribution' in schema_org_metadata:
                # Distribution info provides access details
                score += 10
                
                # Check if distribution has contentUrl
                if isinstance(schema_org_metadata['distribution'], dict) and 'contentUrl' in schema_org_metadata['distribution']:
                    score += 5
                    
                # Check if distribution has multiple formats
                if isinstance(schema_org_metadata['distribution'], list) and len(schema_org_metadata['distribution']) > 1:
                    score += 5
            
            # Check for license info which affects access rights
            if 'license' in schema_org_metadata:
                score += 10
                
        # Ensure score doesn't exceed maximum
        return min(score, max_score)
        
    def _assess_interoperability(self, dataset: Dataset, schema_org_metadata: Optional[Dict] = None) -> float:
        """
        Assess interoperability score based on formats and standards
        
        Returns:
            Interoperability score (0-100)
        """
        score = 0
        max_score = 100
        
        # Check for format information
        if hasattr(dataset, 'format') and dataset.format:
            # Base points for having format info
            score += 15
            
            # More points for standard, interoperable formats
            standard_formats = ['csv', 'json', 'xml', 'rdf', 'turtle', 'n3', 'jsonld']
            if dataset.format.lower() in standard_formats:
                score += 20
            elif dataset.format.lower() in ['xlsx', 'xls', 'ods']:
                # Spreadsheet formats are somewhat interoperable
                score += 10
            elif dataset.format.lower() in ['pdf', 'doc', 'docx']:
                # Less interoperable formats
                score += 5
        
        # Check for data structure information
        if hasattr(dataset, 'data_structure') and dataset.data_structure:
            score += 15
        
        # Check for data dictionary or schema
        if hasattr(dataset, 'data_dictionary') and dataset.data_dictionary:
            score += 20
        
        # Check Schema.org metadata for interoperability factors
        if schema_org_metadata:
            # Schema.org itself enhances interoperability
            score += 15
            
            # Check for standard vocabularies
            if '@context' in schema_org_metadata and schema_org_metadata['@context'] == 'https://schema.org/':
                score += 5
                
            # Check for structured variable descriptions
            if 'variableMeasured' in schema_org_metadata:
                score += 10
                
        # If the dataset is in JSON-LD format it's highly interoperable
        if hasattr(dataset, 'format') and dataset.format and dataset.format.lower() == 'jsonld':
            score += 25
            
        # Ensure score doesn't exceed maximum
        return min(score, max_score)
        
    def _assess_reusability(self, dataset: Dataset, schema_org_metadata: Optional[Dict] = None) -> float:
        """
        Assess reusability score based on licensing, provenance and completeness
        
        Returns:
            Reusability score (0-100)
        """
        score = 0
        max_score = 100
        
        # Check for license information (critical for reuse)
        if hasattr(dataset, 'license') and dataset.license:
            score += 25
            
            # Check for standard open licenses
            open_licenses = ['cc0', 'public domain', 'cc-by', 'mit', 'apache', 'gpl']
            if any(license_type in dataset.license.lower() for license_type in open_licenses):
                score += 10
        
        # Check for provenance information
        if hasattr(dataset, 'source') and dataset.source:
            score += 15
        
        # Check for rich metadata that helps understand and reuse the data
        if hasattr(dataset, 'description') and dataset.description and len(dataset.description) > 200:
            score += 10
        
        # Check for methodology information
        if hasattr(dataset, 'methodology') and dataset.methodology:
            score += 15
        
        # Check for date information
        if hasattr(dataset, 'created_at') and dataset.created_at:
            score += 5
            
        # Check for author/creator information
        if hasattr(dataset, 'author') and dataset.author:
            score += 10
            
        # Check Schema.org metadata for reusability factors
        if schema_org_metadata:
            # Check for license in schema
            if 'license' in schema_org_metadata and not (hasattr(dataset, 'license') and dataset.license):
                score += 15
                
            # Check for creator info in schema
            if 'creator' in schema_org_metadata:
                score += 10
                
            # Check for citation information
            if 'citation' in schema_org_metadata:
                score += 10
            
            # Check for temporal and spatial coverage (context for reuse)
            if 'temporalCoverage' in schema_org_metadata:
                score += 5
                
            if 'spatialCoverage' in schema_org_metadata:
                score += 5
                
        # Ensure score doesn't exceed maximum
        return min(score, max_score)

    def _assess_schema_org_compliance(self, schema_org_metadata: Dict) -> Tuple[bool, List[str]]:
        """
        Assess compliance with Schema.org Dataset standard
        
        Returns:
            Tuple of (is_compliant, list of compliance issues)
        """
        issues = []
        
        # Check for required context
        if '@context' not in schema_org_metadata:
            issues.append("Missing @context in Schema.org metadata")
        elif 'schema.org' not in str(schema_org_metadata['@context']):
            issues.append("@context does not reference schema.org")
            
        # Check for correct type
        if '@type' not in schema_org_metadata:
            issues.append("Missing @type in Schema.org metadata")
        elif schema_org_metadata['@type'] != 'Dataset':
            issues.append(f"@type should be 'Dataset', found '{schema_org_metadata['@type']}'")
            
        # Check for required fields
        for field in self.schema_required_fields:
            if field not in schema_org_metadata:
                issues.append(f"Missing required Schema.org field: {field}")
                
        # Generate warnings for recommended fields
        for field in self.schema_recommended_fields:
            if field not in schema_org_metadata:
                issues.append(f"Missing recommended Schema.org field: {field}")
                
        # Schema is compliant if there are no issues with required fields
        is_compliant = all(not issue.startswith("Missing required") for issue in issues)
        
        return is_compliant, issues

    def _calculate_overall_score(self, assessment: Dict) -> float:
        """
        Calculate overall quality score as weighted average of dimension scores
        
        Returns:
            Overall quality score (0-100)
        """
        # Extract dimension scores
        completeness = assessment.get('completeness', 0)
        consistency = assessment.get('consistency', 0)
        accuracy = assessment.get('accuracy', 0)
        timeliness = assessment.get('timeliness', 0)
        
        # Calculate fair score as average of fair dimensions
        fair_score = (
            assessment.get('findable_score', 0) * self.fair_weights['findable'] +
            assessment.get('accessible_score', 0) * self.fair_weights['accessible'] +
            assessment.get('interoperable_score', 0) * self.fair_weights['interoperable'] +
            assessment.get('reusable_score', 0) * self.fair_weights['reusable']
        )
        
        # Calculate weighted average
        overall_score = (
            completeness * self.dimension_weights['completeness'] +
            consistency * self.dimension_weights['consistency'] +
            accuracy * self.dimension_weights['accuracy'] +
            timeliness * self.dimension_weights['timeliness'] +
            fair_score * self.dimension_weights['fairness']
        )
        
        # Round to integer
        return round(overall_score)

    def _is_fair_compliant(self, assessment: Dict) -> bool:
        """
        Determine if a dataset meets minimum FAIR compliance standards
        
        Returns:
            Boolean indicating FAIR compliance
        """
        # Set minimum thresholds for each FAIR principle
        min_findable = 70
        min_accessible = 70
        min_interoperable = 60
        min_reusable = 70
        
        # Dataset is FAIR compliant if it meets all minimum thresholds
        return (
            assessment.get('findable_score', 0) >= min_findable and
            assessment.get('accessible_score', 0) >= min_accessible and
            assessment.get('interoperable_score', 0) >= min_interoperable and
            assessment.get('reusable_score', 0) >= min_reusable
        )

    def _generate_recommendation_for_issue(self, issue: str) -> str:
        """
        Generate a recommendation based on a specific issue
        
        Returns:
            A recommendation string
        """
        # Dictionary mapping issue patterns to recommendations
        issue_recommendations = {
            "Missing essential metadata fields": "Add complete metadata information to improve dataset discoverability",
            "Missing required Schema.org field": "Add the missing Schema.org field to improve metadata compliance",
            "Format inconsistency": "Correct the format information to match the actual dataset format",
            "Record count inconsistency": "Update the record count to reflect the actual number of records",
            "High missing value rate": "Address the high rate of missing values in the dataset",
            "Moderate missing value rate": "Consider addressing missing values in the dataset",
            "High outlier rate": "Investigate and document potential outliers in the dataset",
            "Data type inconsistency": "Review and correct data type inconsistencies",
            "Missing @context": "Add proper Schema.org @context to the metadata",
            "Missing @type": "Specify Dataset as the @type in Schema.org metadata",
            "Metadata states": "Update metadata to accurately reflect the dataset's characteristics"
        }
        
        # Find matching recommendation
        for pattern, recommendation in issue_recommendations.items():
            if pattern in issue:
                return recommendation
                
        # Default recommendation
        return "Address this issue to improve dataset quality"

    def generate_improvement_plan(self, assessment: Dict) -> List[Dict]:
        """
        Generate a prioritized improvement plan based on quality assessment
        
        Args:
            assessment: The quality assessment dictionary
            
        Returns:
            List of improvement actions in priority order
        """
        improvements = []
        
        # Add recommendations from the assessment
        for i, recommendation in enumerate(assessment.get('recommendations', [])):
            improvements.append({
                'priority': i+1,
                'action': recommendation,
                'impact': self._estimate_recommendation_impact(recommendation, assessment)
            })
            
        # Add general improvements based on dimension scores
        dimension_improvements = self._generate_dimension_improvements(assessment)
        improvements.extend(dimension_improvements)
        
        # Sort by impact (high to low)
        improvements.sort(key=lambda x: x['impact'], reverse=True)
        
        # Update priorities based on sorted order
        for i, improvement in enumerate(improvements):
            improvement['priority'] = i+1
            
        return improvements
        
    def _estimate_recommendation_impact(self, recommendation: str, assessment: Dict) -> str:
        """Estimate the impact of a recommendation on quality score"""
        # High impact recommendations
        high_impact_keywords = ['required', 'missing essential', 'high rate', 'license', 
                              'format', 'identifier', 'schema']
                              
        # Medium impact recommendations                
        medium_impact_keywords = ['recommended', 'moderate', 'update', 'improve', 
                                'consistency', 'outlier']
        
        # Check for high impact keywords
        if any(keyword in recommendation.lower() for keyword in high_impact_keywords):
            return "high"
            
        # Check for medium impact keywords
        if any(keyword in recommendation.lower() for keyword in medium_impact_keywords):
            return "medium"
            
        # Default impact
        return "low"
        
    def _generate_dimension_improvements(self, assessment: Dict) -> List[Dict]:
        """Generate improvements based on dimension scores"""
        improvements = []
        
        # Check completeness
        if assessment.get('completeness', 0) < 70:
            improvements.append({
                'priority': 0,
                'action': "Improve metadata completeness by adding missing fields",
                'impact': "high"
            })
            
        # Check findability
        if assessment.get('findable_score', 0) < 70:
            improvements.append({
                'priority': 0,
                'action': "Enhance dataset findability with better keywords, description and identifiers",
                'impact': "high"
            })
            
        # Check accessibility
        if assessment.get('accessible_score', 0) < 70:
            improvements.append({
                'priority': 0,
                'action': "Improve dataset accessibility by ensuring clear access methods and permissions",
                'impact': "high"
            })
            
        # Check interoperability
        if assessment.get('interoperable_score', 0) < 60:
            improvements.append({
                'priority': 0,
                'action': "Enhance interoperability by using standard formats and providing data dictionary",
                'impact': "medium"
            })
            
        # Check reusability
        if assessment.get('reusable_score', 0) < 70:
            improvements.append({
                'priority': 0,
                'action': "Improve reusability by adding license information and usage documentation",
                'impact': "high"
            })
            
        return improvements

    def compare_datasets(self, datasets: List[Dataset], quality_assessments: List[Dict]) -> Dict:
        """
        Compare multiple datasets based on their quality assessments
        
        Args:
            datasets: List of datasets to compare
            quality_assessments: List of quality assessments for each dataset
            
        Returns:
            Dictionary with comparison results
        """
        if len(datasets) != len(quality_assessments):
            raise ValueError("Number of datasets must match number of quality assessments")
            
        comparison = {
            'datasets': [],
            'dimensions': {
                'quality_score': [],
                'completeness': [],
                'consistency': [],
                'findable_score': [],
                'accessible_score': [],
                'interoperable_score': [],
                'reusable_score': []
            },
            'best_overall': None,
            'best_by_dimension': {}
        }
        
        # Collect data for each dataset
        for i, (dataset, assessment) in enumerate(zip(datasets, quality_assessments)):
            dataset_info = {
                'id': dataset.id if hasattr(dataset, 'id') else i,
                'title': dataset.title if hasattr(dataset, 'title') else f"Dataset {i+1}",
                'quality_score': assessment.get('quality_score', 0),
                'completeness': assessment.get('completeness', 0),
                'consistency': assessment.get('consistency', 0),
                'findable_score': assessment.get('findable_score', 0),
                'accessible_score': assessment.get('accessible_score', 0),
                'interoperable_score': assessment.get('interoperable_score', 0),
                'reusable_score': assessment.get('reusable_score', 0)
            }
            
            comparison['datasets'].append(dataset_info)
            
            # Add scores to dimensions
            for dim in comparison['dimensions'].keys():
                comparison['dimensions'][dim].append(dataset_info.get(dim, 0))
                
        # Find best dataset overall
        if comparison['datasets']:
            best_idx = np.argmax([d['quality_score'] for d in comparison['datasets']])
            comparison['best_overall'] = comparison['datasets'][best_idx]['id']
            
        # Find best dataset by dimension
        for dim in comparison['dimensions'].keys():
            if comparison['dimensions'][dim]:
                best_idx = np.argmax(comparison['dimensions'][dim])
                comparison['best_by_dimension'][dim] = comparison['datasets'][best_idx]['id']
                
        return comparison


# Singleton instance
_quality_service = None

def get_quality_service():
    """Get singleton instance of QualityService"""
    global _quality_service
    if _quality_service is None:
        _quality_service = QualityService()
    return _quality_service