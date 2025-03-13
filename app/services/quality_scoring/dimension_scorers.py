"""
Scorer implementations for individual quality dimensions.

This module contains scorer classes for each quality dimension:
- Completeness: Assesses the presence of required metadata fields
- Consistency: Measures data format and schema consistency
- Accuracy: Evaluates correctness of data content
- Timeliness: Assesses freshness and update frequency
- Conformity: Measures adherence to standards (Schema.org, FAIR)
- Integrity: Evaluates referential integrity and data validity
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional, Set, Union

from app.models.dataset import Dataset
from app.services.quality_scoring.base_scorer import BaseDimensionScorer

logger = logging.getLogger(__name__)


class CompletenessScorer(BaseDimensionScorer):
    """
    Scorer for the completeness dimension.
    
    Assesses how complete the dataset metadata is based on:
    - Required fields presence (title, description, etc.)
    - Optional fields presence
    - Richness of metadata fields
    """
    
    def __init__(self):
        """Initialize the completeness scorer."""
        super().__init__()
        
        # Define required and recommended fields
        self.required_fields = {
            'title', 'description', 'source'
        }
        
        self.recommended_fields = {
            'data_type', 'category', 'format', 'license', 
            'keywords', 'tags', 'author', 'publisher',
            'sample_data', 'schema', 'size'
        }
        
        # Field weighting
        self.required_weight = 0.7  # 70% importance
        self.recommended_weight = 0.3  # 30% importance
    
    def score(self, dataset: Dataset, processed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Score the dataset on metadata completeness.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data from the dataset
            
        Returns:
            Tuple of (completeness_score, detailed_metrics)
        """
        present_required = 0
        missing_required = []
        
        # Check required fields
        for field in self.required_fields:
            if hasattr(dataset, field) and getattr(dataset, field):
                present_required += 1
            else:
                missing_required.append(field)
                self.add_issue(f"Missing required field: {field}")
                self.add_recommendation(f"Add {field} to improve dataset completeness")
        
        # Calculate required fields score
        if len(self.required_fields) > 0:
            required_score = (present_required / len(self.required_fields)) * 100
        else:
            required_score = 100
        
        # Check recommended fields
        present_recommended = 0
        present_recommended_fields = []
        missing_recommended = []
        
        for field in self.recommended_fields:
            if hasattr(dataset, field) and getattr(dataset, field):
                present_recommended += 1
                present_recommended_fields.append(field)
            else:
                missing_recommended.append(field)
        
        # Calculate recommended fields score
        if len(self.recommended_fields) > 0:
            recommended_score = (present_recommended / len(self.recommended_fields)) * 100
        else:
            recommended_score = 100
        
        # Calculate overall completeness score
        completeness_score = (
            required_score * self.required_weight +
            recommended_score * self.recommended_weight
        )
        
        # Create recommendations for missing recommended fields
        if len(missing_recommended) > 0:
            self.add_recommendation(
                f"Consider adding these fields to enhance completeness: {', '.join(missing_recommended[:3])}"
            )
        
        # Create detailed metrics
        metrics = {
            'required_fields_present': present_required,
            'required_fields_total': len(self.required_fields),
            'required_fields_score': required_score,
            'recommended_fields_present': present_recommended,
            'recommended_fields_total': len(self.recommended_fields),
            'recommended_fields_score': recommended_score,
            'present_fields': list(present_recommended_fields),
            'missing_required_fields': missing_required,
            'missing_recommended_fields': missing_recommended
        }
        
        return self._normalize_score(completeness_score), metrics


class ConsistencyScorer(BaseDimensionScorer):
    """
    Scorer for the consistency dimension.
    
    Assesses how consistent the dataset is based on:
    - Data format consistency
    - Schema consistency
    - Value pattern consistency
    - Unit consistency
    """
    
    def score(self, dataset: Dataset, processed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Score the dataset on data consistency.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data from the dataset
            
        Returns:
            Tuple of (consistency_score, detailed_metrics)
        """
        # Default metrics
        metrics = {
            'format_consistency': 0.0,
            'schema_consistency': 0.0,
            'value_consistency': 0.0,
            'inconsistent_fields': []
        }
        
        # If no processed data, return minimal score
        if not processed_data:
            self.add_issue("No processed data available for consistency assessment")
            self.add_recommendation("Process the dataset to enable consistency scoring")
            return 20.0, metrics
        
        # Analyze schema consistency if data has schema
        schema_consistency = 0.0
        if hasattr(processed_data, 'schema') and processed_data.schema:
            schema_consistency = self._check_schema_consistency(processed_data.schema)
        elif isinstance(processed_data, dict) and 'schema' in processed_data:
            schema_consistency = self._check_schema_consistency(processed_data['schema'])
        
        # Analyze format consistency
        format_consistency = self._check_format_consistency(dataset, processed_data)
        
        # Analyze value pattern consistency
        value_consistency = self._check_value_consistency(processed_data)
        
        # Calculate overall consistency score
        # Weightings: schema (40%), format (30%), values (30%)
        consistency_score = (
            schema_consistency * 0.4 +
            format_consistency * 0.3 +
            value_consistency * 0.3
        )
        
        # Update metrics
        metrics['format_consistency'] = format_consistency
        metrics['schema_consistency'] = schema_consistency
        metrics['value_consistency'] = value_consistency
        
        return self._normalize_score(consistency_score), metrics
    
    def _check_schema_consistency(self, schema: Union[Dict, List, str]) -> float:
        """
        Check consistency of the dataset schema.
        
        Args:
            schema: The schema definition for the dataset
            
        Returns:
            Schema consistency score (0-100)
        """
        # Parse schema if it's a string
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError:
                self.add_issue("Schema is not valid JSON")
                return 0.0
        
        # If no schema, return 0
        if not schema:
            self.add_issue("No schema definition found")
            self.add_recommendation("Define a schema for the dataset")
            return 0.0
        
        # Check schema structure
        score = 0.0
        
        if isinstance(schema, dict):
            # For object schemas, check field definitions
            if all(isinstance(v, dict) and 'type' in v for v in schema.values()):
                score = 100.0
                
                # Check for inconsistent field definitions
                for field, definition in schema.items():
                    if not all(k in ['type', 'format', 'description', 'required'] for k in definition.keys()):
                        score -= 10.0
                        self.add_issue(f"Inconsistent field definition for {field}")
            else:
                score = 50.0
                self.add_issue("Schema structure is not consistent")
                self.add_recommendation("Standardize schema field definitions")
        elif isinstance(schema, list):
            # For array schemas, check item consistency
            if len(schema) > 0:
                item_types = set(type(item) for item in schema)
                
                if len(item_types) == 1:
                    score = 90.0
                else:
                    score = 70.0
                    self.add_issue("Schema array contains inconsistent item types")
        
        return score
    
    def _check_format_consistency(self, dataset: Dataset, processed_data: Any) -> float:
        """
        Check consistency of data formats.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data
            
        Returns:
            Format consistency score (0-100)
        """
        # Default score
        score = 50.0
        
        # Check if format is explicitly defined
        if hasattr(dataset, 'format') and dataset.format:
            score += 20.0
        
        # If processed data is a dictionary with samples
        if isinstance(processed_data, dict) and 'samples' in processed_data:
            samples = processed_data['samples']
            if isinstance(samples, list) and len(samples) > 0:
                # Check if all samples have the same structure
                if all(isinstance(sample, dict) for sample in samples):
                    # Get fields from first sample
                    first_sample_keys = set(samples[0].keys())
                    
                    # Check if all samples have the same fields
                    if all(set(sample.keys()) == first_sample_keys for sample in samples):
                        score += 30.0
                    else:
                        score -= 10.0
                        self.add_issue("Inconsistent fields across data samples")
        
        return score
    
    def _check_value_consistency(self, processed_data: Any) -> float:
        """
        Check consistency of data values.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Value consistency score (0-100)
        """
        # Default score
        score = 50.0
        inconsistent_fields = []
        
        # If processed data is a dictionary with samples
        if isinstance(processed_data, dict) and 'samples' in processed_data:
            samples = processed_data['samples']
            if isinstance(samples, list) and len(samples) > 1:
                # Get fields from first sample
                if isinstance(samples[0], dict):
                    first_sample = samples[0]
                    
                    # Check type consistency for each field
                    for field in first_sample:
                        field_types = set()
                        
                        for sample in samples:
                            if field in sample:
                                field_types.add(type(sample[field]))
                        
                        if len(field_types) > 1:
                            inconsistent_fields.append(field)
                            self.add_issue(f"Field '{field}' has inconsistent value types")
                    
                    # Calculate score based on consistency
                    if inconsistent_fields:
                        consistency_ratio = 1.0 - (len(inconsistent_fields) / len(first_sample))
                        score = 50.0 + (consistency_ratio * 50.0)
                    else:
                        score = 100.0
        
        return score


class AccuracyScorer(BaseDimensionScorer):
    """
    Scorer for the accuracy dimension.
    
    Assesses how accurate the dataset content is based on:
    - Data validation against expected formats
    - Range checking for numeric values
    - Outlier detection
    - Error rate estimation
    """
    
    def score(self, dataset: Dataset, processed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Score the dataset on data accuracy.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data from the dataset
            
        Returns:
            Tuple of (accuracy_score, detailed_metrics)
        """
        # Default metrics
        metrics = {
            'validation_score': 0.0,
            'range_score': 0.0,
            'outlier_score': 0.0,
            'overall_error_rate': 0.0,
            'validation_errors': []
        }
        
        # If no processed data, return minimal score
        if not processed_data:
            self.add_issue("No processed data available for accuracy assessment")
            self.add_recommendation("Process the dataset to enable accuracy scoring")
            return 30.0, metrics
        
        # Validate data against expected formats
        validation_score, validation_errors = self._validate_data_formats(processed_data)
        
        # Check numeric ranges
        range_score = self._check_numeric_ranges(processed_data)
        
        # Detect outliers
        outlier_score = self._detect_outliers(processed_data)
        
        # Calculate estimated error rate
        error_rate = self._estimate_error_rate(validation_score, range_score, outlier_score)
        
        # Calculate overall accuracy score
        # Weightings: validation (40%), range (30%), outliers (30%)
        accuracy_score = (
            validation_score * 0.4 +
            range_score * 0.3 +
            outlier_score * 0.3
        )
        
        # Update metrics
        metrics['validation_score'] = validation_score
        metrics['range_score'] = range_score
        metrics['outlier_score'] = outlier_score
        metrics['overall_error_rate'] = error_rate
        metrics['validation_errors'] = validation_errors
        
        return self._normalize_score(accuracy_score), metrics
    
    def _validate_data_formats(self, processed_data: Any) -> Tuple[float, List[str]]:
        """
        Validate data against expected formats.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Tuple of (validation_score, validation_errors)
        """
        validation_score = 70.0  # Default score
        validation_errors = []
        
        # If processed data is a dictionary with samples and schema
        if isinstance(processed_data, dict):
            samples = processed_data.get('samples', [])
            schema = processed_data.get('schema', {})
            
            if samples and schema and isinstance(schema, dict):
                # Track validation success rate
                total_validations = 0
                successful_validations = 0
                
                for sample in samples:
                    if isinstance(sample, dict):
                        for field, value in sample.items():
                            if field in schema:
                                total_validations += 1
                                
                                # Get expected type from schema
                                expected_type = schema[field].get('type') if isinstance(schema[field], dict) else None
                                
                                # Validate based on expected type
                                if expected_type:
                                    valid = self._validate_value_by_type(value, expected_type)
                                    
                                    if valid:
                                        successful_validations += 1
                                    else:
                                        error = f"Field '{field}' value '{value}' does not match expected type '{expected_type}'"
                                        validation_errors.append(error)
                
                # Calculate validation score
                if total_validations > 0:
                    validation_score = (successful_validations / total_validations) * 100
                    
                    if validation_score < 80:
                        self.add_issue(f"Data validation rate is low ({validation_score:.1f}%)")
                        self.add_recommendation("Review and correct data format issues")
        
        return validation_score, validation_errors
    
    def _validate_value_by_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate a value against an expected type.
        
        Args:
            value: The value to validate
            expected_type: The expected type name
            
        Returns:
            True if valid, False otherwise
        """
        if expected_type.lower() == 'string' or expected_type.lower() == 'text':
            return isinstance(value, str)
        elif expected_type.lower() == 'number' or expected_type.lower() == 'float':
            return isinstance(value, (int, float))
        elif expected_type.lower() == 'integer' or expected_type.lower() == 'int':
            return isinstance(value, int)
        elif expected_type.lower() == 'boolean' or expected_type.lower() == 'bool':
            return isinstance(value, bool)
        elif expected_type.lower() == 'array' or expected_type.lower() == 'list':
            return isinstance(value, list)
        elif expected_type.lower() == 'object' or expected_type.lower() == 'dict':
            return isinstance(value, dict)
        elif expected_type.lower() == 'date':
            if isinstance(value, str):
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                    try:
                        datetime.strptime(value, fmt)
                        return True
                    except ValueError:
                        continue
            return False
        
        # Default case
        return True
    
    def _check_numeric_ranges(self, processed_data: Any) -> float:
        """
        Check numeric values for reasonable ranges.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Range check score (0-100)
        """
        range_score = 80.0  # Default score
        
        # If processed data is a dictionary with samples
        if isinstance(processed_data, dict):
            samples = processed_data.get('samples', [])
            
            if samples:
                # Track numeric fields and their values
                numeric_values = {}
                
                for sample in samples:
                    if isinstance(sample, dict):
                        for field, value in sample.items():
                            if isinstance(value, (int, float)):
                                if field not in numeric_values:
                                    numeric_values[field] = []
                                numeric_values[field].append(value)
                
                # Check ranges for each numeric field
                out_of_range_fields = 0
                
                for field, values in numeric_values.items():
                    if len(values) > 1:
                        min_val = min(values)
                        max_val = max(values)
                        avg_val = sum(values) / len(values)
                        
                        # Check for suspicious ranges - extremely wide ranges often indicate errors
                        if max_val > avg_val * 100 or min_val < avg_val / 100:
                            out_of_range_fields += 1
                            self.add_issue(f"Field '{field}' has suspicious value range: {min_val} to {max_val}")
                
                # Adjust score based on out-of-range fields
                if numeric_values:
                    range_score -= (out_of_range_fields / len(numeric_values)) * 30.0
        
        return range_score
    
    def _detect_outliers(self, processed_data: Any) -> float:
        """
        Detect outliers in the dataset.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Outlier score (0-100)
        """
        outlier_score = 90.0  # Default score
        
        # If processed data is a dictionary with samples
        if isinstance(processed_data, dict):
            samples = processed_data.get('samples', [])
            
            if samples:
                # Track numeric fields and their values
                numeric_values = {}
                
                for sample in samples:
                    if isinstance(sample, dict):
                        for field, value in sample.items():
                            if isinstance(value, (int, float)):
                                if field not in numeric_values:
                                    numeric_values[field] = []
                                numeric_values[field].append(value)
                
                # Check for outliers in each numeric field
                total_fields = len(numeric_values)
                fields_with_outliers = 0
                
                for field, values in numeric_values.items():
                    if len(values) > 4:  # Need enough values to detect outliers
                        # Calculate mean and standard deviation
                        mean = sum(values) / len(values)
                        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                        
                        # Define outlier threshold (> 3 standard deviations from mean)
                        lower_bound = mean - 3 * std_dev
                        upper_bound = mean + 3 * std_dev
                        
                        # Count outliers
                        outliers = [v for v in values if v < lower_bound or v > upper_bound]
                        
                        if outliers:
                            fields_with_outliers += 1
                            outlier_ratio = len(outliers) / len(values)
                            
                            if outlier_ratio > 0.05:  # More than 5% outliers
                                self.add_issue(f"Field '{field}' has {len(outliers)} outliers ({outlier_ratio:.1%})")
                
                # Adjust score based on fields with outliers
                if total_fields > 0:
                    outlier_score -= (fields_with_outliers / total_fields) * 30.0
        
        return outlier_score
    
    def _estimate_error_rate(self, validation_score: float, range_score: float, outlier_score: float) -> float:
        """
        Estimate overall error rate based on validation, range, and outlier scores.
        
        Args:
            validation_score: Score from format validation
            range_score: Score from range checking
            outlier_score: Score from outlier detection
            
        Returns:
            Estimated error rate as a percentage
        """
        # Convert scores to error rates (100 - score)
        validation_error_rate = 100 - validation_score
        range_error_rate = 100 - range_score
        outlier_error_rate = 100 - outlier_score
        
        # Weighted average of error rates
        error_rate = (
            validation_error_rate * 0.5 +
            range_error_rate * 0.3 +
            outlier_error_rate * 0.2
        )
        
        return error_rate


class TimelinessScorer(BaseDimensionScorer):
    """
    Scorer for the timeliness dimension.
    
    Assesses how timely the dataset is based on:
    - Creation date
    - Last update date
    - Update frequency
    - Temporal coverage
    """
    
    def score(self, dataset: Dataset, processed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Score the dataset on timeliness.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data from the dataset
            
        Returns:
            Tuple of (timeliness_score, detailed_metrics)
        """
        # Default metrics
        metrics = {
            'freshness_score': 0.0,
            'update_frequency_score': 0.0,
            'temporal_coverage_score': 0.0,
            'days_since_update': None,
            'update_frequency': None
        }
        
        # Calculate freshness score
        freshness_score = self._calculate_freshness_score(dataset)
        metrics['freshness_score'] = freshness_score
        
        # Calculate update frequency score
        update_frequency_score = self._calculate_update_frequency_score(dataset)
        metrics['update_frequency_score'] = update_frequency_score
        
        # Calculate temporal coverage score
        temporal_coverage_score = self._calculate_temporal_coverage_score(dataset, processed_data)
        metrics['temporal_coverage_score'] = temporal_coverage_score
        
        # Calculate days since last update
        days_since_update = None
        if hasattr(dataset, 'updated_at') and dataset.updated_at:
            days_since_update = (datetime.utcnow() - dataset.updated_at).days
            metrics['days_since_update'] = days_since_update
        
        # Calculate overall timeliness score
        # Weightings: freshness (50%), update frequency (30%), temporal coverage (20%)
        timeliness_score = (
            freshness_score * 0.5 +
            update_frequency_score * 0.3 +
            temporal_coverage_score * 0.2
        )
        
        return self._normalize_score(timeliness_score), metrics
    
    def _calculate_freshness_score(self, dataset: Dataset) -> float:
        """
        Calculate freshness score based on creation and update dates.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Freshness score (0-100)
        """
        # Default score
        score = 50.0
        
        # Check creation date
        if hasattr(dataset, 'created_at') and dataset.created_at:
            age_days = (datetime.utcnow() - dataset.created_at).days
            
            # Newer datasets get higher scores
            if age_days < 30:  # Less than a month old
                score = 90.0
            elif age_days < 90:  # Less than 3 months old
                score = 80.0
            elif age_days < 180:  # Less than 6 months old
                score = 70.0
            elif age_days < 365:  # Less than a year old
                score = 60.0
            else:  # More than a year old
                score = 50.0
        
        # Check update date if available
        if hasattr(dataset, 'updated_at') and dataset.updated_at:
            days_since_update = (datetime.utcnow() - dataset.updated_at).days
            
            # Recently updated datasets get higher scores
            if days_since_update < 7:  # Updated within a week
                score = max(score, 95.0)
            elif days_since_update < 30:  # Updated within a month
                score = max(score, 85.0)
            elif days_since_update < 90:  # Updated within 3 months
                score = max(score, 75.0)
            
            # Old datasets with recent updates still get good scores
            if hasattr(dataset, 'created_at') and dataset.created_at:
                age_days = (datetime.utcnow() - dataset.created_at).days
                if age_days > 365 and days_since_update < 90:
                    score = max(score, 70.0)
                    
            # Provide issues and recommendations for outdated datasets
            if days_since_update > 180:  # More than 6 months since update
                self.add_issue(f"Dataset hasn't been updated in {days_since_update} days")
                self.add_recommendation("Update the dataset to improve timeliness")
        
        return score
    
    def _calculate_update_frequency_score(self, dataset: Dataset) -> float:
        """
        Calculate update frequency score.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Update frequency score (0-100)
        """
        # Default score
        score = 60.0
        
        # Check if update frequency is specified
        if hasattr(dataset, 'update_frequency') and dataset.update_frequency:
            # Higher score for datasets with defined update frequency
            score = 80.0
            
            # Parse update frequency - look for patterns like "daily", "weekly", "monthly"
            freq = dataset.update_frequency.lower()
            if "daily" in freq or "day" in freq:
                score = 100.0
            elif "weekly" in freq or "week" in freq:
                score = 90.0
            elif "monthly" in freq or "month" in freq:
                score = 80.0
            elif "quarterly" in freq or "quarter" in freq:
                score = 70.0
            elif "yearly" in freq or "year" in freq or "annual" in freq:
                score = 60.0
            elif "never" in freq or "static" in freq:
                score = 40.0
        else:
            self.add_recommendation("Specify update frequency to improve timeliness assessment")
        
        return score
    
    def _calculate_temporal_coverage_score(self, dataset: Dataset, processed_data: Any) -> float:
        """
        Calculate temporal coverage score.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data
            
        Returns:
            Temporal coverage score (0-100)
        """
        # Default score
        score = 50.0
        
        # Check if temporal coverage is specified
        if hasattr(dataset, 'temporal_coverage') and dataset.temporal_coverage:
            score = 90.0
        
        # Check processed data for temporal indicators
        if isinstance(processed_data, dict) and 'samples' in processed_data:
            samples = processed_data['samples']
            
            # Look for date-like fields in the first few samples
            date_fields = set()
            
            for sample in samples[:10]:  # Check first 10 samples
                if isinstance(sample, dict):
                    for field, value in sample.items():
                        # Check if field name suggests a date
                        date_indicators = ['date', 'time', 'year', 'month', 'day']
                        if any(indicator in field.lower() for indicator in date_indicators):
                            date_fields.add(field)
                        
                        # Check if value looks like a date string
                        if isinstance(value, str):
                            date_patterns = [
                                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
                                r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
                            ]
                            
                            if any(re.match(pattern, value) for pattern in date_patterns):
                                date_fields.add(field)
            
            # If date fields found but temporal coverage not specified
            if date_fields and not (hasattr(dataset, 'temporal_coverage') and dataset.temporal_coverage):
                self.add_recommendation(
                    f"Consider adding temporal coverage information based on fields: {', '.join(date_fields)}"
                )
                score = 70.0
        
        return score


class ConformityScorer(BaseDimensionScorer):
    """
    Scorer for the conformity dimension.
    
    Assesses how well the dataset conforms to standards:
    - Schema.org compliance
    - FAIR principles compliance
    - Domain-specific standards compliance
    """
    
    def score(self, dataset: Dataset, processed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Score the dataset on conformity to standards.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data from the dataset
            
        Returns:
            Tuple of (conformity_score, detailed_metrics)
        """
        # Default metrics
        metrics = {
            'schema_org_score': 0.0,
            'fair_score': 0.0,
            'domain_standards_score': 0.0,
            'fair_details': {
                'findable': 0.0,
                'accessible': 0.0,
                'interoperable': 0.0,
                'reusable': 0.0
            }
        }
        
        # Calculate Schema.org compliance score
        schema_org_score = self._calculate_schema_org_compliance(dataset)
        metrics['schema_org_score'] = schema_org_score
        
        # Calculate FAIR principles compliance score
        fair_score, fair_details = self._calculate_fair_compliance(dataset)
        metrics['fair_score'] = fair_score
        metrics['fair_details'] = fair_details
        
        # Calculate domain-specific standards compliance
        domain_standards_score = self._calculate_domain_standards_compliance(dataset)
        metrics['domain_standards_score'] = domain_standards_score
        
        # Calculate overall conformity score
        # Weightings: Schema.org (40%), FAIR (40%), domain standards (20%)
        conformity_score = (
            schema_org_score * 0.4 +
            fair_score * 0.4 +
            domain_standards_score * 0.2
        )
        
        return self._normalize_score(conformity_score), metrics
    
    def _calculate_schema_org_compliance(self, dataset: Dataset) -> float:
        """
        Calculate compliance with Schema.org standards.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Schema.org compliance score (0-100)
        """
        # Default score
        score = 40.0
        
        # Check if Schema.org metadata is defined
        if hasattr(dataset, 'schema_org') and dataset.schema_org:
            # Attempt to parse if it's a string
            schema_org_data = dataset.schema_org
            if isinstance(schema_org_data, str):
                try:
                    schema_org_data = json.loads(schema_org_data)
                except json.JSONDecodeError:
                    self.add_issue("Schema.org metadata is not valid JSON")
                    return 30.0
            
            # Check for required Schema.org Dataset fields
            required_fields = ['@type', 'name', 'description']
            recommended_fields = [
                'url', 'creator', 'publisher', 'datePublished', 
                'license', 'keywords', 'variableMeasured'
            ]
            
            if isinstance(schema_org_data, dict):
                # Check required fields
                required_present = all(field in schema_org_data for field in required_fields)
                
                # Count recommended fields
                recommended_count = sum(1 for field in recommended_fields if field in schema_org_data)
                
                # Calculate score
                if required_present:
                    score = 60.0 + (recommended_count / len(recommended_fields)) * 40.0
                else:
                    score = 40.0
                    missing = [field for field in required_fields if field not in schema_org_data]
                    self.add_issue(f"Missing required Schema.org fields: {', '.join(missing)}")
                    self.add_recommendation("Add required Schema.org fields to improve conformity")
            else:
                self.add_issue("Schema.org metadata is not properly structured")
                score = 30.0
        else:
            self.add_issue("Schema.org metadata not defined")
            self.add_recommendation("Define Schema.org metadata to improve standards conformity")
        
        return score
    
    def _calculate_fair_compliance(self, dataset: Dataset) -> Tuple[float, Dict[str, float]]:
        """
        Calculate compliance with FAIR principles.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Tuple of (fair_score, fair_details)
        """
        # Initialize FAIR component scores
        findable = self._assess_findability(dataset)
        accessible = self._assess_accessibility(dataset)
        interoperable = self._assess_interoperability(dataset)
        reusable = self._assess_reusability(dataset)
        
        # Calculate overall FAIR score
        fair_score = (findable + accessible + interoperable + reusable) / 4.0
        
        # Fair details
        fair_details = {
            'findable': findable,
            'accessible': accessible,
            'interoperable': interoperable,
            'reusable': reusable
        }
        
        return fair_score, fair_details
    
    def _assess_findability(self, dataset: Dataset) -> float:
        """
        Assess the Findability aspect of FAIR principles.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Findability score (0-100)
        """
        score = 0.0
        
        # F1: Globally unique, persistent identifier
        if hasattr(dataset, 'identifier') and dataset.identifier:
            score += 25.0
        else:
            self.add_recommendation("Assign a persistent identifier to improve findability")
        
        # F2: Rich metadata
        if hasattr(dataset, 'title') and dataset.title and hasattr(dataset, 'description') and dataset.description:
            score += 25.0
            
            # Additional metadata richness
            rich_metadata_count = 0
            rich_metadata_fields = ['keywords', 'tags', 'category', 'author', 'publisher']
            
            for field in rich_metadata_fields:
                if hasattr(dataset, field) and getattr(dataset, field):
                    rich_metadata_count += 1
            
            score += (rich_metadata_count / len(rich_metadata_fields)) * 25.0
        
        # F3: Metadata includes the identifier
        # This is typically handled internally by the system
        score += 20.0
        
        # F4: Indexed in a searchable resource
        if hasattr(dataset, 'indexed') and dataset.indexed:
            score += 25.0
        else:
            self.add_recommendation("Ensure the dataset is indexed in searchable resources")
        
        return score
    
    def _assess_accessibility(self, dataset: Dataset) -> float:
        """
        Assess the Accessibility aspect of FAIR principles.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Accessibility score (0-100)
        """
        score = 0.0
        
        # A1: Retrievable by identifier using standard protocol
        if hasattr(dataset, 'source_url') and dataset.source_url:
            score += 40.0
        else:
            self.add_recommendation("Provide a standard access URL to improve accessibility")
        
        # A1.1: Open, free protocol
        if hasattr(dataset, 'source_url') and dataset.source_url:
            if dataset.source_url.startswith(('http://', 'https://')):
                score += 20.0
        
        # A1.2: Authentication and authorization
        # This depends on the system implementation
        score += 20.0
        
        # A2: Metadata accessible even when data is not
        # This is typically handled by the metadata system
        score += 20.0
        
        return score
    
    def _assess_interoperability(self, dataset: Dataset) -> float:
        """
        Assess the Interoperability aspect of FAIR principles.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Interoperability score (0-100)
        """
        score = 0.0
        
        # I1: Knowledge representation language
        if hasattr(dataset, 'schema_org') and dataset.schema_org:
            score += 30.0
        elif hasattr(dataset, 'format') and dataset.format in ['json', 'xml', 'rdf', 'csv']:
            score += 20.0
        else:
            self.add_recommendation("Use a formal, accessible knowledge representation language")
        
        # I2: FAIR vocabularies
        if hasattr(dataset, 'vocabulary') and dataset.vocabulary:
            score += 30.0
        else:
            self.add_recommendation("Use FAIR vocabularies to improve interoperability")
        
        # I3: Qualified references to other data
        if hasattr(dataset, 'related_datasets') and dataset.related_datasets:
            score += 40.0
        else:
            self.add_recommendation("Include qualified references to related datasets")
        
        return score
    
    def _assess_reusability(self, dataset: Dataset) -> float:
        """
        Assess the Reusability aspect of FAIR principles.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Reusability score (0-100)
        """
        score = 0.0
        
        # R1: Rich metadata with relevant attributes
        rich_metadata_count = 0
        rich_metadata_fields = ['description', 'author', 'publisher', 'license', 'created_at', 'updated_at']
        
        for field in rich_metadata_fields:
            if hasattr(dataset, field) and getattr(dataset, field):
                rich_metadata_count += 1
        
        score += (rich_metadata_count / len(rich_metadata_fields)) * 40.0
        
        # R1.1: Clear and accessible data usage license
        if hasattr(dataset, 'license') and dataset.license:
            score += 30.0
        else:
            self.add_recommendation("Specify a clear license to improve reusability")
        
        # R1.2: Detailed provenance
        if hasattr(dataset, 'provenance') and dataset.provenance:
            score += 20.0
        else:
            if hasattr(dataset, 'source') and dataset.source:
                score += 10.0
            self.add_recommendation("Add detailed provenance information")
        
        # R1.3: Community standards
        if hasattr(dataset, 'standards') and dataset.standards:
            score += 10.0
        
        return score
    
    def _calculate_domain_standards_compliance(self, dataset: Dataset) -> float:
        """
        Calculate compliance with domain-specific standards.
        
        Args:
            dataset: The dataset being scored
            
        Returns:
            Domain standards compliance score (0-100)
        """
        # Default score
        score = 50.0
        
        # Check for domain-specific standards based on category
        if hasattr(dataset, 'category') and dataset.category:
            category = dataset.category.lower()
            
            # Different domains have different standards
            if category in ['health', 'healthcare', 'medical']:
                # Health data standards
                if hasattr(dataset, 'standards') and dataset.standards:
                    standards = dataset.standards.lower()
                    
                    if 'hipaa' in standards or 'fhir' in standards or 'hl7' in standards:
                        score = 90.0
                    elif 'omop' in standards or 'snomed' in standards or 'loinc' in standards:
                        score = 85.0
                    else:
                        score = 70.0
                        
                    self.add_recommendation("Consider adding HIPAA, FHIR, or HL7 compliance information")
                else:
                    score = 50.0
                    self.add_recommendation("Add healthcare data standards compliance information")
            
            elif category in ['education', 'academic']:
                # Education data standards
                if hasattr(dataset, 'standards') and dataset.standards:
                    standards = dataset.standards.lower()
                    
                    if 'ceds' in standards or 'lrmi' in standards:
                        score = 90.0
                    else:
                        score = 70.0
                        
                    self.add_recommendation("Consider adding CEDS or LRMI compliance information")
                else:
                    score = 50.0
                    self.add_recommendation("Add education data standards compliance information")
            
            elif category in ['environment', 'climate']:
                # Environmental data standards
                if hasattr(dataset, 'standards') and dataset.standards:
                    standards = dataset.standards.lower()
                    
                    if 'inspire' in standards or 'ogc' in standards:
                        score = 90.0
                    else:
                        score = 70.0
                        
                    self.add_recommendation("Consider adding INSPIRE or OGC compliance information")
                else:
                    score = 50.0
                    self.add_recommendation("Add environmental data standards compliance information")
        
        return score


class IntegrityScorer(BaseDimensionScorer):
    """
    Scorer for the integrity dimension.
    
    Assesses the data integrity based on:
    - Referential integrity
    - Constraint validation
    - Missing values assessment
    - Duplicate detection
    """
    
    def score(self, dataset: Dataset, processed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Score the dataset on data integrity.
        
        Args:
            dataset: The dataset being scored
            processed_data: The processed data from the dataset
            
        Returns:
            Tuple of (integrity_score, detailed_metrics)
        """
        # Default metrics
        metrics = {
            'missing_values_score': 0.0,
            'duplicates_score': 0.0,
            'constraints_score': 0.0,
            'referential_score': 0.0,
            'missing_percent': 0.0,
            'duplicate_percent': 0.0
        }
        
        # If no processed data, return minimal score
        if not processed_data:
            self.add_issue("No processed data available for integrity assessment")
            self.add_recommendation("Process the dataset to enable integrity scoring")
            return 40.0, metrics
        
        # Assess missing values
        missing_values_score, missing_percent = self._assess_missing_values(processed_data)
        metrics['missing_values_score'] = missing_values_score
        metrics['missing_percent'] = missing_percent
        
        # Detect duplicates
        duplicates_score, duplicate_percent = self._detect_duplicates(processed_data)
        metrics['duplicates_score'] = duplicates_score
        metrics['duplicate_percent'] = duplicate_percent
        
        # Validate constraints
        constraints_score = self._validate_constraints(processed_data)
        metrics['constraints_score'] = constraints_score
        
        # Check referential integrity
        referential_score = self._check_referential_integrity(processed_data)
        metrics['referential_score'] = referential_score
        
        # Calculate overall integrity score
        # Weightings: missing (30%), duplicates (30%), constraints (20%), referential (20%)
        integrity_score = (
            missing_values_score * 0.3 +
            duplicates_score * 0.3 +
            constraints_score * 0.2 +
            referential_score * 0.2
        )
        
        return self._normalize_score(integrity_score), metrics
    
    def _assess_missing_values(self, processed_data: Any) -> Tuple[float, float]:
        """
        Assess missing values in the dataset.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Tuple of (missing_values_score, missing_percent)
        """
        # Default values
        missing_values_score = 80.0
        missing_percent = 0.0
        
        # If processed data is a dictionary with samples
        if isinstance(processed_data, dict) and 'samples' in processed_data:
            samples = processed_data['samples']
            
            if samples and isinstance(samples, list) and all(isinstance(s, dict) for s in samples):
                # Get all field names
                all_fields = set()
                for sample in samples:
                    all_fields.update(sample.keys())
                
                # Count missing values
                total_cells = len(all_fields) * len(samples)
                missing_cells = 0
                
                for field in all_fields:
                    for sample in samples:
                        if field not in sample or sample[field] is None or (isinstance(sample[field], str) and sample[field].strip() == ''):
                            missing_cells += 1
                
                # Calculate missing percentage
                if total_cells > 0:
                    missing_percent = (missing_cells / total_cells) * 100
                    
                    # Adjust score based on missing percentage
                    if missing_percent == 0:
                        missing_values_score = 100.0
                    elif missing_percent < 5:
                        missing_values_score = 90.0
                    elif missing_percent < 10:
                        missing_values_score = 80.0
                    elif missing_percent < 20:
                        missing_values_score = 60.0
                    elif missing_percent < 30:
                        missing_values_score = 40.0
                    else:
                        missing_values_score = 20.0
                    
                    # Add issue for high missing values
                    if missing_percent > 10:
                        self.add_issue(f"High percentage of missing values: {missing_percent:.1f}%")
                        self.add_recommendation("Fill in missing values to improve data integrity")
        
        return missing_values_score, missing_percent
    
    def _detect_duplicates(self, processed_data: Any) -> Tuple[float, float]:
        """
        Detect duplicates in the dataset.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Tuple of (duplicates_score, duplicate_percent)
        """
        # Default values
        duplicates_score = 90.0
        duplicate_percent = 0.0
        
        # If processed data is a dictionary with samples
        if isinstance(processed_data, dict) and 'samples' in processed_data:
            samples = processed_data['samples']
            
            if samples and isinstance(samples, list) and len(samples) > 1:
                # Simplified duplicate detection for demo purposes
                # In a real implementation, this would be more sophisticated
                
                # Convert samples to hashable form (as tuples of sorted items)
                hashable_samples = []
                for sample in samples:
                    if isinstance(sample, dict):
                        # Convert dict to sorted tuple of items
                        items = sorted((k, str(v)) for k, v in sample.items())
                        hashable_samples.append(tuple(items))
                
                # Count unique samples
                unique_samples = set(hashable_samples)
                total_samples = len(hashable_samples)
                duplicate_count = total_samples - len(unique_samples)
                
                # Calculate duplicate percentage
                if total_samples > 0:
                    duplicate_percent = (duplicate_count / total_samples) * 100
                    
                    # Adjust score based on duplicate percentage
                    if duplicate_percent == 0:
                        duplicates_score = 100.0
                    elif duplicate_percent < 1:
                        duplicates_score = 95.0
                    elif duplicate_percent < 5:
                        duplicates_score = 85.0
                    elif duplicate_percent < 10:
                        duplicates_score = 70.0
                    else:
                        duplicates_score = 50.0
                    
                    # Add issue for high duplicate percentage
                    if duplicate_percent > 5:
                        self.add_issue(f"High percentage of duplicate entries: {duplicate_percent:.1f}%")
                        self.add_recommendation("Remove or consolidate duplicate entries")
        
        return duplicates_score, duplicate_percent
    
    def _validate_constraints(self, processed_data: Any) -> float:
        """
        Validate data constraints.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Constraints score (0-100)
        """
        # Default score
        constraints_score = 70.0
        
        # If processed data is a dictionary with samples and schema
        if isinstance(processed_data, dict):
            samples = processed_data.get('samples', [])
            schema = processed_data.get('schema', {})
            
            if samples and schema and isinstance(schema, dict):
                # Track constraint violations
                constraint_violations = 0
                total_constraints = 0
                
                for field, field_schema in schema.items():
                    if isinstance(field_schema, dict):
                        # Check for constraints
                        constraints = {}
                        
                        if 'required' in field_schema:
                            constraints['required'] = field_schema['required']
                            total_constraints += 1
                        
                        if 'min' in field_schema:
                            constraints['min'] = field_schema['min']
                            total_constraints += 1
                        
                        if 'max' in field_schema:
                            constraints['max'] = field_schema['max']
                            total_constraints += 1
                        
                        if 'pattern' in field_schema:
                            constraints['pattern'] = field_schema['pattern']
                            total_constraints += 1
                        
                        # Validate constraints for each sample
                        for sample in samples:
                            if isinstance(sample, dict):
                                # Required constraint
                                if 'required' in constraints and constraints['required']:
                                    if field not in sample or sample[field] is None:
                                        constraint_violations += 1
                                
                                # Min/max constraints for numeric fields
                                if field in sample and sample[field] is not None:
                                    value = sample[field]
                                    
                                    if isinstance(value, (int, float)):
                                        if 'min' in constraints and value < constraints['min']:
                                            constraint_violations += 1
                                        
                                        if 'max' in constraints and value > constraints['max']:
                                            constraint_violations += 1
                                    
                                    # Pattern constraint for string fields
                                    if isinstance(value, str) and 'pattern' in constraints:
                                        pattern = constraints['pattern']
                                        if not re.match(pattern, value):
                                            constraint_violations += 1
                
                # Calculate constraints score
                if total_constraints > 0:
                    violation_rate = constraint_violations / (total_constraints * len(samples))
                    constraints_score = 100.0 - (violation_rate * 100.0)
                    
                    # Add issue for high constraint violations
                    if violation_rate > 0.1:
                        self.add_issue(f"Data constraint violations detected ({violation_rate:.1%} violation rate)")
                        self.add_recommendation("Fix constraint violations to improve data integrity")
        
        return constraints_score
    
    def _check_referential_integrity(self, processed_data: Any) -> float:
        """
        Check referential integrity.
        
        Args:
            processed_data: The processed data
            
        Returns:
            Referential integrity score (0-100)
        """
        # Default score - in a basic dataset without relations, integrity is presumed good
        referential_score = 80.0
        
        # If processed data contains explicit relations
        if isinstance(processed_data, dict) and 'relations' in processed_data:
            relations = processed_data['relations']
            samples = processed_data.get('samples', [])
            
            if relations and samples:
                # Track referential integrity violations
                integrity_violations = 0
                total_relations = 0
                
                for relation in relations:
                    if isinstance(relation, dict) and 'source' in relation and 'target' in relation:
                        total_relations += 1
                        source_field = relation['source']
                        target_field = relation.get('target_field', 'id')
                        target_entity = relation['target']
                        
                        # Get all target values
                        target_values = set()
                        if target_entity in processed_data:
                            target_samples = processed_data[target_entity]
                            for target_sample in target_samples:
                                if isinstance(target_sample, dict) and target_field in target_sample:
                                    target_values.add(target_sample[target_field])
                        
                        # Check source references
                        for sample in samples:
                            if isinstance(sample, dict) and source_field in sample:
                                source_value = sample[source_field]
                                
                                if source_value and source_value not in target_values:
                                    integrity_violations += 1
                
                # Calculate referential integrity score
                if total_relations > 0:
                    integrity_score = 100.0 - ((integrity_violations / (total_relations * len(samples))) * 100.0)
                    referential_score = integrity_score
                    
                    # Add issue for integrity violations
                    if integrity_violations > 0:
                        self.add_issue(f"Referential integrity violations detected: {integrity_violations}")
                        self.add_recommendation("Fix referential integrity issues to improve data quality")
        
        return referential_score