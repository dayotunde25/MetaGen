"""
Validation utility functions for AIMetaHarvest.

This module provides validation functions for data integrity,
security, and input sanitization.
"""

import re
import os
import html
from typing import Dict, Any, List, Optional, Union, Tuple
from urllib.parse import urlparse
import pandas as pd

def validate_dataset_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate dataset data structure and content.
    
    Args:
        data: Dataset data dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ['title', 'format']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate title
    if 'title' in data:
        title = data['title']
        if len(title) < 3:
            errors.append("Title must be at least 3 characters long")
        elif len(title) > 200:
            errors.append("Title must be less than 200 characters")
        elif not re.match(r'^[a-zA-Z0-9\s\-_.,()]+$', title):
            errors.append("Title contains invalid characters")
    
    # Validate format
    if 'format' in data:
        valid_formats = ['csv', 'json', 'xlsx', 'xls', 'xml', 'txt', 'tsv', 'parquet']
        if data['format'].lower() not in valid_formats:
            errors.append(f"Invalid format. Must be one of: {', '.join(valid_formats)}")
    
    # Validate description if provided
    if 'description' in data and data['description']:
        if len(data['description']) > 5000:
            errors.append("Description must be less than 5000 characters")
    
    # Validate source if provided
    if 'source' in data and data['source']:
        if len(data['source']) > 500:
            errors.append("Source must be less than 500 characters")
    
    # Validate category if provided
    if 'category' in data and data['category']:
        if len(data['category']) > 100:
            errors.append("Category must be less than 100 characters")
    
    # Validate tags if provided
    if 'tags' in data and data['tags']:
        if isinstance(data['tags'], str):
            tags = data['tags'].split(',')
            if len(tags) > 20:
                errors.append("Maximum 20 tags allowed")
            for tag in tags:
                if len(tag.strip()) > 50:
                    errors.append("Each tag must be less than 50 characters")
    
    return len(errors) == 0, errors

def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate metadata structure and content.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate quality scores if present
    score_fields = ['quality_score', 'findable_score', 'accessible_score', 
                   'interoperable_score', 'reusable_score', 'fair_score']
    
    for field in score_fields:
        if field in metadata:
            score = metadata[field]
            if not isinstance(score, (int, float)):
                errors.append(f"{field} must be a number")
            elif score < 0 or score > 100:
                errors.append(f"{field} must be between 0 and 100")
    
    # Validate field count if present
    if 'field_count' in metadata:
        field_count = metadata['field_count']
        if not isinstance(field_count, int) or field_count < 0:
            errors.append("field_count must be a non-negative integer")
    
    # Validate record count if present
    if 'record_count' in metadata:
        record_count = metadata['record_count']
        if not isinstance(record_count, int) or record_count < 0:
            errors.append("record_count must be a non-negative integer")
    
    return len(errors) == 0, errors

def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent XSS and other attacks.
    
    Args:
        input_string: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        return ""
    
    # Truncate if too long
    if len(input_string) > max_length:
        input_string = input_string[:max_length]
    
    # HTML escape
    sanitized = html.escape(input_string)
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', sanitized)
    
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return sanitized

def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return re.match(pattern, email) is not None

def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def is_safe_path(path: str, base_path: str = None) -> bool:
    """
    Check if a file path is safe (no directory traversal).
    
    Args:
        path: Path to check
        base_path: Base path to restrict to
        
    Returns:
        True if path is safe, False otherwise
    """
    if not path:
        return False
    
    # Normalize the path
    normalized_path = os.path.normpath(path)
    
    # Check for directory traversal attempts
    if '..' in normalized_path or normalized_path.startswith('/'):
        return False
    
    # If base_path is provided, ensure path is within it
    if base_path:
        try:
            full_path = os.path.join(base_path, normalized_path)
            real_path = os.path.realpath(full_path)
            real_base = os.path.realpath(base_path)
            
            return real_path.startswith(real_base)
        except Exception:
            return False
    
    return True

def validate_json_structure(json_data: Union[str, Dict, List]) -> Tuple[bool, str]:
    """
    Validate JSON data structure.
    
    Args:
        json_data: JSON data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if isinstance(json_data, str):
            import json
            json.loads(json_data)
        elif isinstance(json_data, (dict, list)):
            # Already parsed JSON
            pass
        else:
            return False, "Invalid JSON data type"
        
        return True, "Valid JSON"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"JSON validation error: {str(e)}"

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate pandas DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if df is None:
        errors.append("DataFrame is None")
        return False, errors
    
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # Check for reasonable size limits
    if len(df) > 1000000:  # 1 million rows
        errors.append("DataFrame too large (>1M rows)")
    
    if len(df.columns) > 1000:  # 1000 columns
        errors.append("DataFrame has too many columns (>1000)")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        errors.append("DataFrame has duplicate column names")
    
    # Check for empty column names
    empty_cols = [col for col in df.columns if not str(col).strip()]
    if empty_cols:
        errors.append("DataFrame has empty column names")
    
    return len(errors) == 0, errors

def validate_file_upload(file_obj, allowed_extensions: set = None) -> Tuple[bool, str]:
    """
    Validate uploaded file object.
    
    Args:
        file_obj: File object from upload
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_obj:
        return False, "No file provided"
    
    if not file_obj.filename:
        return False, "No filename provided"
    
    # Check file extension
    if allowed_extensions:
        ext = file_obj.filename.rsplit('.', 1)[1].lower() if '.' in file_obj.filename else ''
        if ext not in allowed_extensions:
            return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    # Check file size (if available)
    if hasattr(file_obj, 'content_length') and file_obj.content_length:
        max_size = 5 * 1024 * 1024 * 1024  # 5GB
        if file_obj.content_length > max_size:
            return False, f"File too large. Maximum size: {max_size // (1024*1024*1024)}GB"
    
    return True, "File is valid"

def clean_text(text: str) -> str:
    """
    Clean and normalize text input.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize quotes
    text = re.sub(r'[""''`]', '"', text)
    
    return text
