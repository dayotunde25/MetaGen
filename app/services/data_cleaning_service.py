"""
Advanced Data Cleaning and Restructuring Service for AIMetaHarvest.

This service provides comprehensive data cleaning, transformation, and standardization
capabilities to ensure datasets meet Schema.org, FAIR principles, and AI standards.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataCleaningService:
    """
    Advanced data cleaning and restructuring service.
    
    Provides:
    - Data deduplication
    - Missing value handling
    - Data type standardization
    - Format normalization
    - Outlier detection
    - Schema standardization
    """
    
    def __init__(self):
        """Initialize the data cleaning service."""
        self.cleaning_stats = {}
        self.transformation_log = []
        
    def clean_dataset(self, dataset_data: Dict[str, Any], cleaning_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data cleaning on a dataset.
        
        Args:
            dataset_data: Raw dataset data with samples and schema
            cleaning_config: Configuration for cleaning operations
            
        Returns:
            Cleaned dataset with transformation log
        """
        if not cleaning_config:
            cleaning_config = self._get_default_cleaning_config()
            
        self.cleaning_stats = {}
        self.transformation_log = []
        
        try:
            # Extract samples for cleaning
            samples = dataset_data.get('sample_data', [])
            if not samples:
                return dataset_data
                
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(samples)
            original_shape = df.shape
            
            # Step 1: Remove duplicates
            if cleaning_config.get('remove_duplicates', True):
                df = self._remove_duplicates(df)
                
            # Step 2: Handle missing values
            if cleaning_config.get('handle_missing', True):
                df = self._handle_missing_values(df, cleaning_config.get('missing_strategy', 'auto'))
                
            # Step 3: Standardize data types
            if cleaning_config.get('standardize_types', True):
                df = self._standardize_data_types(df)
                
            # Step 4: Normalize formats
            if cleaning_config.get('normalize_formats', True):
                df = self._normalize_formats(df)
                
            # Step 5: Detect and handle outliers
            if cleaning_config.get('handle_outliers', True):
                df = self._handle_outliers(df, cleaning_config.get('outlier_method', 'iqr'))
                
            # Step 6: Validate and correct data
            if cleaning_config.get('validate_data', True):
                df = self._validate_and_correct(df)
                
            # Update dataset with cleaned data
            cleaned_data = dataset_data.copy()
            cleaned_data['sample_data'] = df.to_dict(orient='records')
            cleaned_data['record_count'] = len(df)
            cleaned_data['cleaning_stats'] = self.cleaning_stats
            cleaned_data['transformation_log'] = self.transformation_log
            
            # Update schema with cleaned types
            cleaned_data['schema'] = self._generate_cleaned_schema(df)
            
            self._log_transformation(f"Dataset cleaned: {original_shape} -> {df.shape}")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            self._log_transformation(f"Cleaning failed: {str(e)}")
            return dataset_data
    
    def _get_default_cleaning_config(self) -> Dict[str, Any]:
        """Get default cleaning configuration."""
        return {
            'remove_duplicates': True,
            'handle_missing': True,
            'missing_strategy': 'auto',  # auto, drop, fill, interpolate
            'standardize_types': True,
            'normalize_formats': True,
            'handle_outliers': True,
            'outlier_method': 'iqr',  # iqr, zscore, isolation
            'validate_data': True,
            'max_missing_percent': 50,  # Drop columns with >50% missing
            'outlier_threshold': 3.0    # Z-score threshold
        }
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_cleaned)
        
        if duplicates_removed > 0:
            self.cleaning_stats['duplicates_removed'] = duplicates_removed
            self._log_transformation(f"Removed {duplicates_removed} duplicate rows")
            
        return df_cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing values based on the specified strategy."""
        missing_stats = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_stats[column] = {
                'count': missing_count,
                'percent': missing_percent
            }
        
        self.cleaning_stats['missing_values'] = missing_stats
        
        # Apply strategy
        if strategy == 'auto':
            df_cleaned = self._auto_handle_missing(df)
        elif strategy == 'drop':
            df_cleaned = df.dropna()
        elif strategy == 'fill':
            df_cleaned = self._fill_missing_values(df)
        elif strategy == 'interpolate':
            df_cleaned = self._interpolate_missing_values(df)
        else:
            df_cleaned = df
            
        return df_cleaned
    
    def _auto_handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically handle missing values based on data characteristics."""
        df_cleaned = df.copy()
        
        for column in df.columns:
            missing_percent = (df[column].isnull().sum() / len(df)) * 100
            
            if missing_percent > 50:
                # Drop columns with >50% missing values
                df_cleaned = df_cleaned.drop(columns=[column])
                self._log_transformation(f"Dropped column '{column}' ({missing_percent:.1f}% missing)")
            elif missing_percent > 0:
                # Fill missing values based on data type
                if df[column].dtype in ['int64', 'float64']:
                    # Numeric: use median
                    df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
                    self._log_transformation(f"Filled missing values in '{column}' with median")
                else:
                    # Categorical: use mode or 'Unknown'
                    mode_value = df_cleaned[column].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    df_cleaned[column] = df_cleaned[column].fillna(fill_value)
                    self._log_transformation(f"Filled missing values in '{column}' with '{fill_value}'")
        
        return df_cleaned

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using appropriate strategies."""
        df_cleaned = df.copy()

        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype in ['int64', 'float64']:
                    # Numeric: use median
                    df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
                else:
                    # Categorical: use mode or 'Unknown'
                    mode_value = df_cleaned[column].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    df_cleaned[column] = df_cleaned[column].fillna(fill_value)

        return df_cleaned

    def _interpolate_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values for numeric columns."""
        df_cleaned = df.copy()

        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype in ['int64', 'float64']:
                    # Numeric: use interpolation
                    df_cleaned[column] = df_cleaned[column].interpolate()
                else:
                    # Categorical: use forward fill then backward fill
                    df_cleaned[column] = df_cleaned[column].fillna(method='ffill').fillna(method='bfill')

        return df_cleaned

    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types for consistency."""
        df_cleaned = df.copy()
        type_changes = {}
        
        for column in df.columns:
            original_type = str(df[column].dtype)
            
            # Try to infer and convert to appropriate types
            if self._is_numeric_column(df[column]):
                try:
                    df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
                    new_type = str(df_cleaned[column].dtype)
                    if new_type != original_type:
                        type_changes[column] = {'from': original_type, 'to': new_type}
                except:
                    pass
            elif self._is_datetime_column(df[column]):
                try:
                    df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
                    new_type = str(df_cleaned[column].dtype)
                    if new_type != original_type:
                        type_changes[column] = {'from': original_type, 'to': new_type}
                except:
                    pass
            elif self._is_boolean_column(df[column]):
                try:
                    df_cleaned[column] = df_cleaned[column].astype('bool')
                    new_type = str(df_cleaned[column].dtype)
                    if new_type != original_type:
                        type_changes[column] = {'from': original_type, 'to': new_type}
                except:
                    pass
        
        if type_changes:
            self.cleaning_stats['type_changes'] = type_changes
            self._log_transformation(f"Standardized data types for {len(type_changes)} columns")
        
        return df_cleaned
    
    def _normalize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data formats for consistency."""
        df_cleaned = df.copy()
        format_changes = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Normalize text data
                if self._is_text_column(df[column]):
                    # Trim whitespace and normalize case
                    df_cleaned[column] = df_cleaned[column].astype(str).str.strip()
                    format_changes.append(f"Normalized text format in '{column}'")
                
                # Normalize date formats
                elif self._is_date_string_column(df[column]):
                    try:
                        df_cleaned[column] = pd.to_datetime(df_cleaned[column]).dt.strftime('%Y-%m-%d')
                        format_changes.append(f"Normalized date format in '{column}' to YYYY-MM-DD")
                    except:
                        pass
        
        if format_changes:
            self.cleaning_stats['format_normalizations'] = format_changes
            for change in format_changes:
                self._log_transformation(change)
        
        return df_cleaned
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect and handle outliers in numeric columns."""
        df_cleaned = df.copy()
        outliers_detected = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'iqr':
                outliers = self._detect_outliers_iqr(df[column])
            elif method == 'zscore':
                outliers = self._detect_outliers_zscore(df[column])
            else:
                continue
                
            if len(outliers) > 0:
                outliers_detected[column] = len(outliers)
                # Cap outliers at 95th percentile
                upper_cap = df[column].quantile(0.95)
                lower_cap = df[column].quantile(0.05)
                df_cleaned[column] = df_cleaned[column].clip(lower=lower_cap, upper=upper_cap)
                self._log_transformation(f"Capped {len(outliers)} outliers in '{column}'")
        
        if outliers_detected:
            self.cleaning_stats['outliers_handled'] = outliers_detected
        
        return df_cleaned
    
    def _validate_and_correct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data and apply corrections."""
        df_cleaned = df.copy()
        corrections = []
        
        # Email validation and correction
        for column in df.columns:
            if 'email' in column.lower():
                invalid_emails = ~df[column].astype(str).str.contains(r'^[^@]+@[^@]+\.[^@]+$', na=False)
                if invalid_emails.any():
                    df_cleaned.loc[invalid_emails, column] = None
                    corrections.append(f"Removed {invalid_emails.sum()} invalid emails from '{column}'")
        
        # URL validation
        for column in df.columns:
            if 'url' in column.lower() or 'link' in column.lower():
                invalid_urls = ~df[column].astype(str).str.contains(r'^https?://', na=False)
                if invalid_urls.any():
                    df_cleaned.loc[invalid_urls, column] = None
                    corrections.append(f"Removed {invalid_urls.sum()} invalid URLs from '{column}'")
        
        if corrections:
            self.cleaning_stats['data_corrections'] = corrections
            for correction in corrections:
                self._log_transformation(correction)
        
        return df_cleaned
    
    # Helper methods for data type detection
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column should be numeric."""
        try:
            pd.to_numeric(series.dropna(), errors='raise')
            return True
        except:
            return False
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a column contains datetime data."""
        try:
            # Test with sample to avoid performance issues
            sample_data = series.dropna().head(10)
            if len(sample_data) == 0:
                return False

            # Try common date formats first
            common_formats = [
                '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
                '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%d-%m-%Y', '%d.%m.%Y', '%Y.%m.%d'
            ]

            for fmt in common_formats:
                try:
                    pd.to_datetime(sample_data, format=fmt, errors='raise')
                    return True
                except (ValueError, TypeError):
                    continue

            # Fallback to dateutil parsing with warning suppression
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample_data, errors='raise')
            return True
        except:
            return False
    
    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if a column should be boolean."""
        unique_values = set(series.dropna().astype(str).str.lower())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        return unique_values.issubset(boolean_values)
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if a column contains text data."""
        return series.dtype == 'object' and not self._is_datetime_column(series)
    
    def _is_date_string_column(self, series: pd.Series) -> bool:
        """Check if a column contains date strings."""
        sample = series.dropna().head(10)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}'   # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            if sample.astype(str).str.match(pattern).any():
                return True
        return False
    
    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)].index.tolist()
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return series[z_scores > threshold].index.tolist()
    
    def _generate_cleaned_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate schema for cleaned dataset."""
        schema = {}
        for column in df.columns:
            schema[column] = {
                'type': str(df[column].dtype),
                'sample_values': df[column].dropna().head(5).tolist(),
                'null_count': df[column].isnull().sum(),
                'unique_count': df[column].nunique()
            }
        return schema
    
    def _log_transformation(self, message: str):
        """Log a transformation step."""
        timestamp = datetime.now().isoformat()
        self.transformation_log.append({
            'timestamp': timestamp,
            'message': message
        })
        logger.info(f"Data cleaning: {message}")


# Global service instance
data_cleaning_service = DataCleaningService()
