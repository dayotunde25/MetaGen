"""
File utility functions for AIMetaHarvest.

This module provides utility functions for file handling, validation,
and processing operations.
"""

import os
import mimetypes
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from werkzeug.utils import secure_filename
import pandas as pd
import json

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'csv', 'json', 'xlsx', 'xls', 'xml', 'txt', 'tsv', 'parquet', 'zip'
}

# Maximum file size (5GB)
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024

def allowed_file(filename: str) -> bool:
    """
    Check if a file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension in lowercase
    """
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def secure_filename_custom(filename: str) -> str:
    """
    Create a secure filename with additional customizations.
    
    Args:
        filename: Original filename
        
    Returns:
        Secure filename
    """
    # Use werkzeug's secure_filename as base
    secure_name = secure_filename(filename)
    
    # Additional customizations
    if not secure_name:
        secure_name = 'unnamed_file'
    
    # Ensure we have an extension
    if '.' not in secure_name:
        secure_name += '.txt'
    
    return secure_name

def validate_file_content(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate file content and extract basic information.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, error_message, file_info)
    """
    try:
        file_info = {
            'size': get_file_size(file_path),
            'extension': get_file_extension(file_path),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'records': 0,
            'columns': 0
        }
        
        # Check file size
        if file_info['size'] > MAX_FILE_SIZE:
            return False, f"File too large. Maximum size is {format_file_size(MAX_FILE_SIZE)}", file_info
        
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            return False, "File does not exist", file_info
        
        if not os.access(file_path, os.R_OK):
            return False, "File is not readable", file_info
        
        # Try to read and validate content based on extension
        extension = file_info['extension']
        
        if extension == 'csv':
            df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows for validation
            file_info['records'] = len(pd.read_csv(file_path))
            file_info['columns'] = len(df.columns)
            
        elif extension in ['xlsx', 'xls']:
            df = pd.read_excel(file_path, nrows=5)
            file_info['records'] = len(pd.read_excel(file_path))
            file_info['columns'] = len(df.columns)
            
        elif extension == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                file_info['records'] = len(data)
                if data and isinstance(data[0], dict):
                    file_info['columns'] = len(data[0].keys())
            elif isinstance(data, dict):
                file_info['records'] = 1
                file_info['columns'] = len(data.keys())
                
        elif extension == 'xml':
            # Basic XML validation
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            file_info['records'] = len(list(root))
            
        elif extension in ['txt', 'tsv']:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            file_info['records'] = len(lines)
            if lines:
                # Assume first line contains headers
                separator = '\t' if extension == 'tsv' else ','
                file_info['columns'] = len(lines[0].split(separator))
        
        return True, "File is valid", file_info
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}", file_info

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    try:
        stat = os.stat(file_path)
        
        info = {
            'name': os.path.basename(file_path),
            'path': file_path,
            'size': stat.st_size,
            'size_human': format_file_size(stat.st_size),
            'extension': get_file_extension(file_path),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK)
        }
        
        return info
        
    except OSError as e:
        return {
            'name': os.path.basename(file_path),
            'path': file_path,
            'error': str(e)
        }

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def cleanup_temp_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
        
    Returns:
        Number of files cleaned up
    """
    import time
    
    if not os.path.exists(directory):
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    cleaned_count += 1
                    
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")
    
    return cleaned_count

def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except OSError:
        return False

def extract_zip_file(zip_path: str, extract_to: str = None) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Extract a zip file and return information about extracted files.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to (if None, creates temp directory)

    Returns:
        Tuple of (success, extract_directory, list_of_extracted_files)
    """
    try:
        if not zipfile.is_zipfile(zip_path):
            return False, "", []

        # Create extraction directory
        if extract_to is None:
            extract_to = tempfile.mkdtemp(prefix="dataset_extract_")
        else:
            ensure_directory_exists(extract_to)

        extracted_files = []

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in zip
            file_list = zip_ref.namelist()

            # Filter out directories and hidden files
            data_files = [f for f in file_list if not f.endswith('/') and not f.startswith('.')]

            # Extract files
            for file_name in data_files:
                try:
                    # Extract file
                    zip_ref.extract(file_name, extract_to)
                    extracted_path = os.path.join(extract_to, file_name)

                    # Get file info
                    if os.path.exists(extracted_path):
                        file_info = get_file_info(extracted_path)
                        file_info['original_name'] = file_name
                        file_info['is_dataset'] = get_file_extension(file_name) in ALLOWED_EXTENSIONS
                        extracted_files.append(file_info)

                except Exception as e:
                    print(f"Error extracting {file_name}: {e}")
                    continue

        return True, extract_to, extracted_files

    except Exception as e:
        return False, str(e), []

def validate_zip_contents(zip_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate zip file contents and provide summary.

    Args:
        zip_path: Path to the zip file

    Returns:
        Tuple of (is_valid, error_message, zip_info)
    """
    try:
        if not zipfile.is_zipfile(zip_path):
            return False, "File is not a valid zip archive", {}

        zip_info = {
            'total_files': 0,
            'dataset_files': 0,
            'total_size': 0,
            'file_types': {},
            'dataset_files_list': []
        }

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            for file_name in file_list:
                if not file_name.endswith('/'):  # Skip directories
                    zip_info['total_files'] += 1

                    # Get file extension
                    ext = get_file_extension(file_name)
                    zip_info['file_types'][ext] = zip_info['file_types'].get(ext, 0) + 1

                    # Check if it's a dataset file
                    if ext in ALLOWED_EXTENSIONS and ext != 'zip':
                        zip_info['dataset_files'] += 1
                        zip_info['dataset_files_list'].append({
                            'name': file_name,
                            'extension': ext,
                            'size': zip_ref.getinfo(file_name).file_size
                        })

                    # Add to total size
                    zip_info['total_size'] += zip_ref.getinfo(file_name).file_size

        # Validate that we have at least one dataset file
        if zip_info['dataset_files'] == 0:
            return False, "Zip file contains no valid dataset files", zip_info

        # Check total uncompressed size
        if zip_info['total_size'] > MAX_FILE_SIZE:
            return False, f"Total uncompressed size exceeds limit ({format_file_size(MAX_FILE_SIZE)})", zip_info

        return True, "Zip file is valid", zip_info

    except Exception as e:
        return False, f"Error validating zip file: {str(e)}", {}

def generate_collection_title(file_names: List[str]) -> str:
    """
    Generate a title for a collection of datasets based on file names.

    Args:
        file_names: List of file names in the collection

    Returns:
        Generated collection title
    """
    if not file_names:
        return "Dataset Collection"

    # Extract base names without extensions
    base_names = [os.path.splitext(name)[0] for name in file_names]

    # Find common prefixes or patterns
    if len(base_names) == 1:
        return base_names[0].replace('_', ' ').replace('-', ' ').title()

    # Look for common words
    common_words = set()
    for name in base_names:
        words = name.replace('_', ' ').replace('-', ' ').split()
        if not common_words:
            common_words = set(words)
        else:
            common_words = common_words.intersection(set(words))

    if common_words:
        title = ' '.join(sorted(common_words)).title()
        if len(title) > 5:  # Only use if meaningful
            return f"{title} Dataset Collection"

    # Fallback: use first file name as base
    base_title = base_names[0].replace('_', ' ').replace('-', ' ').title()
    return f"{base_title} and Related Datasets"
