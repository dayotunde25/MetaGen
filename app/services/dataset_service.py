import os
import json
import urllib.request
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from app.models.dataset import Dataset
from app.models.metadata import ProcessingQueue
from app.utils.file_utils import extract_zip_file, validate_zip_contents, generate_collection_title

logger = logging.getLogger(__name__)

class DatasetService:
    """Service for dataset operations including fetching, processing, and analysis"""

    def __init__(self, upload_folder):
        """Initialize dataset service"""
        self.upload_folder = upload_folder
        self.tempdir = os.path.join(upload_folder, 'temp')

        # Create directories if they don't exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(self.tempdir, exist_ok=True)

    def fetch_dataset(self, url):
        """Fetch a dataset from a URL and save it locally"""
        if not url:
            return None

        try:
            import requests
            from urllib.parse import urlparse, unquote

            # Parse URL to get filename
            parsed_url = urlparse(url)
            filename = os.path.basename(unquote(parsed_url.path))

            # If no filename in URL, generate one
            if not filename or '.' not in filename:
                filename = f"dataset_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

            # Ensure filename is secure
            filename = secure_filename(filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            local_path = os.path.join(self.tempdir, f"{timestamp}_{filename}")

            # Download file with better error handling
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # Write file in chunks
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Determine format from extension or content type
            format = os.path.splitext(filename)[1][1:].lower()
            if not format:
                content_type = response.headers.get('content-type', '').lower()
                if 'csv' in content_type:
                    format = 'csv'
                elif 'json' in content_type:
                    format = 'json'
                elif 'xml' in content_type:
                    format = 'xml'
                else:
                    format = 'unknown'

            # Get file size
            size = os.path.getsize(local_path)

            print(f"✅ Successfully fetched dataset from URL: {url}")
            print(f"   Local path: {local_path}")
            print(f"   Format: {format}")
            print(f"   Size: {size} bytes")

            return {
                'path': local_path,
                'format': format,
                'size': size,
                'original_url': url,
                'filename': filename
            }
        except Exception as e:
            print(f"❌ Error fetching dataset from URL {url}: {str(e)}")
            return None

    def process_dataset(self, file_info, format_hint=None, process_full_dataset=True):
        """Process a dataset file to extract information and structure"""
        if not file_info or not os.path.exists(file_info['path']):
            return None

        try:
            # Determine format
            format = format_hint or file_info['format']

            # Process according to format with full dataset option
            if format in ['csv', 'tsv', 'txt']:
                return self._process_csv(file_info['path'], process_full_dataset)
            elif format in ['xlsx', 'xls']:
                return self._process_excel(file_info['path'], process_full_dataset)
            elif format in ['json']:
                return self._process_json(file_info['path'], process_full_dataset)
            elif format in ['xml']:
                return self._process_xml(file_info['path'])
            elif format == 'zip':
                return self._process_zip(file_info['path'])
            else:
                return self._process_generic_file(file_info['path'])
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            return None

    def _process_csv(self, file_path, process_full_dataset=True):
        """Process a CSV file to extract schema and full dataset or sample data"""
        try:
            # Try to detect delimiter with multiple approaches
            delimiter = ','  # Default

            with open(file_path, 'r', encoding='utf-8') as file:
                sample = file.read(1024)

            # Try common delimiters
            delimiters = [',', ';', '\t', '|']
            for test_delimiter in delimiters:
                if test_delimiter in sample:
                    delimiter = test_delimiter
                    break

            # Try sniffer as backup
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    dialect = csv.Sniffer().sniff(file.read(1024), delimiters=',;\t|')
                    delimiter = dialect.delimiter
            except Exception:
                pass  # Keep the delimiter we found above

            # Count total rows efficiently
            total_rows = self._count_csv_rows(file_path, delimiter)

            # Decide whether to process full dataset or sample based on size
            if process_full_dataset and total_rows <= 10000:  # Process full dataset for reasonable sizes
                print(f"Processing full dataset with {total_rows} records")
                df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8')
                df_sample = df.head(100)  # Keep sample for quick schema analysis
            elif process_full_dataset and total_rows > 10000:  # For very large datasets, use chunked processing
                print(f"Processing large dataset with {total_rows} records using chunked approach")
                df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8', nrows=5000)  # Process first 5000 rows
                df_sample = df.head(100)
            else:  # Sample-only processing (legacy mode)
                print(f"Processing sample of dataset with {total_rows} total records")
                df = pd.read_csv(file_path, sep=delimiter, nrows=100, encoding='utf-8')
                df_sample = df

            # Extract comprehensive schema from sample
            schema = {}
            for col in df_sample.columns:
                dtype = str(df_sample[col].dtype)

                # Enhanced schema with statistics
                col_data = df_sample[col].dropna()
                schema[col] = {
                    'type': dtype,
                    'sample_values': col_data.head(5).tolist(),
                    'null_count': df_sample[col].isnull().sum(),
                    'unique_count': df_sample[col].nunique(),
                    'data_type_inferred': self._infer_semantic_type(col, col_data)
                }

            # Return full dataset information
            result = {
                'format': 'csv',
                'columns': list(df.columns),
                'schema': schema,
                'record_count': total_rows,
                'processed_records': len(df),
                'full_data': df.to_dict(orient='records') if len(df) <= 1000 else None,  # Include full data for small datasets
                'sample_data': df_sample.head(10).to_dict(orient='records'),
                'processing_mode': 'full' if process_full_dataset else 'sample',
                'data_summary': {
                    'total_rows': total_rows,
                    'total_columns': len(df.columns),
                    'processed_rows': len(df),
                    'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            }

            # Add full dataset DataFrame for processing if reasonable size
            if len(df) <= 5000:
                result['dataframe'] = df

            return result
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            # Try fallback processing with different encodings and delimiters
            try:
                return self._fallback_csv_processing(file_path)
            except Exception as fallback_error:
                print(f"Fallback CSV processing also failed: {fallback_error}")
                return {
                    'format': 'csv',
                    'error': f"CSV processing failed: {str(e)}. Fallback also failed: {str(fallback_error)}",
                    'summary': 'Failed to process CSV file',
                    'record_count': 0,
                    'schema': {}
                }

    def _fallback_csv_processing(self, file_path: str):
        """Fallback CSV processing with multiple attempts."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        delimiters = [',', ';', '\t', '|']

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    # Try reading with this combination
                    df = pd.read_csv(file_path, sep=delimiter, encoding=encoding, nrows=50, on_bad_lines='skip')

                    if len(df.columns) > 1 and len(df) > 0:  # Valid data found
                        # Count total rows with this encoding/delimiter combination
                        total_rows = self._count_csv_rows_with_encoding(file_path, delimiter, encoding)

                        # Extract basic schema
                        schema = {}
                        for col in df.columns:
                            dtype = str(df[col].dtype)
                            schema[col] = {
                                'type': dtype,
                                'sample_values': df[col].dropna().head(3).tolist()
                            }

                        return {
                            'format': 'csv',
                            'columns': list(df.columns),
                            'schema': schema,
                            'record_count': total_rows,
                            'sample_data': df.head(10).to_dict(orient='records'),
                            'encoding_used': encoding,
                            'delimiter_used': delimiter,
                            'fallback_processing': True
                        }

                except Exception:
                    continue  # Try next combination

        # If all combinations fail, return minimal structure
        raise Exception("Could not process CSV with any encoding/delimiter combination")

    def _count_csv_rows(self, file_path, delimiter=','):
        """Efficiently count total rows in CSV file without loading all data"""
        try:
            row_count = 0
            with open(file_path, 'r', encoding='utf-8') as file:
                # Skip header
                next(file, None)
                # Count remaining rows
                for line in file:
                    if line.strip():  # Skip empty lines
                        row_count += 1
            return row_count
        except Exception as e:
            print(f"Error counting CSV rows: {e}")
            # Fallback: try with different encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    row_count = 0
                    with open(file_path, 'r', encoding=encoding) as file:
                        next(file, None)  # Skip header
                        for line in file:
                            if line.strip():
                                row_count += 1
                    return row_count
                except Exception:
                    continue
            # If all fail, return 0
            return 0

    def _count_csv_rows_with_encoding(self, file_path, delimiter, encoding):
        """Count CSV rows with specific encoding and delimiter"""
        try:
            row_count = 0
            with open(file_path, 'r', encoding=encoding) as file:
                # Skip header
                next(file, None)
                # Count remaining rows
                for line in file:
                    if line.strip():  # Skip empty lines
                        row_count += 1
            return row_count
        except Exception as e:
            print(f"Error counting CSV rows with encoding {encoding}: {e}")
            return 0

    def _infer_semantic_type(self, column_name, data):
        """Infer semantic type of column based on name and data patterns."""
        column_lower = column_name.lower()

        # Check for common semantic types
        if any(word in column_lower for word in ['id', 'identifier']):
            return 'identifier'
        elif any(word in column_lower for word in ['name', 'title', 'description']):
            return 'text'
        elif any(word in column_lower for word in ['email', 'mail']):
            return 'email'
        elif any(word in column_lower for word in ['url', 'link', 'website']):
            return 'url'
        elif any(word in column_lower for word in ['date', 'time', 'timestamp']):
            return 'datetime'
        elif any(word in column_lower for word in ['price', 'cost', 'amount', 'value']):
            return 'currency'
        elif any(word in column_lower for word in ['count', 'number', 'quantity']):
            return 'numeric'
        elif any(word in column_lower for word in ['category', 'type', 'class']):
            return 'categorical'
        else:
            # Infer from data patterns
            if data.dtype in ['int64', 'float64']:
                return 'numeric'
            elif data.dtype == 'object':
                return 'text'
            else:
                return 'unknown'

    def _process_excel(self, file_path: str, process_full_dataset: bool = False):
        """Process Excel files (XLSX, XLS) with comprehensive analysis"""
        try:
            print(f"🔄 Processing Excel file: {file_path}")

            # Read Excel file - try to get all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            print(f"📊 Found {len(sheet_names)} sheet(s): {sheet_names}")

            # Process the first sheet or largest sheet
            main_sheet = sheet_names[0]
            if len(sheet_names) > 1:
                # Find the sheet with most data
                sheet_sizes = {}
                for sheet in sheet_names:
                    try:
                        temp_df = pd.read_excel(file_path, sheet_name=sheet, nrows=1)
                        full_df = pd.read_excel(file_path, sheet_name=sheet)
                        sheet_sizes[sheet] = len(full_df)
                    except:
                        sheet_sizes[sheet] = 0

                main_sheet = max(sheet_sizes, key=sheet_sizes.get)
                print(f"📈 Using main sheet: {main_sheet} with {sheet_sizes[main_sheet]} rows")

            # Read the main sheet
            if process_full_dataset:
                df = pd.read_excel(file_path, sheet_name=main_sheet)
                df_sample = df.head(100)
                print(f"📊 Processing full Excel dataset with {len(df)} records")
            else:
                df_sample = pd.read_excel(file_path, sheet_name=main_sheet, nrows=100)
                df = pd.read_excel(file_path, sheet_name=main_sheet)  # Get full for count
                print(f"📊 Processing Excel sample with {len(df_sample)} records from {len(df)} total")

            # Extract schema information
            schema = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null_values = df[col].dropna()
                schema[col] = {
                    'type': dtype,
                    'sample_values': non_null_values.head(3).tolist(),
                    'null_count': df[col].isnull().sum(),
                    'unique_count': df[col].nunique()
                }

            # Prepare result
            result = {
                'format': 'excel',
                'file_type': file_path.split('.')[-1].upper(),
                'sheet_names': sheet_names,
                'main_sheet': main_sheet,
                'columns': list(df.columns),
                'schema': schema,
                'record_count': len(df),
                'processed_records': len(df) if process_full_dataset else len(df_sample),
                'sample_data': df_sample.head(10).to_dict(orient='records'),
                'processing_mode': 'full' if process_full_dataset else 'sample',
                'data_summary': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'total_sheets': len(sheet_names),
                    'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            }

            # Add full dataset for processing if reasonable size
            if process_full_dataset and len(df) <= 5000:
                result['dataframe'] = df
                result['full_data'] = df.to_dict(orient='records') if len(df) <= 1000 else None

            # Add text content for NLP analysis
            text_content = self._extract_excel_text_content(df_sample)
            if text_content:
                result['text_content'] = text_content

            print(f"✅ Excel processing completed: {len(df)} records, {len(df.columns)} columns")
            return result

        except Exception as e:
            print(f"❌ Error processing Excel file: {str(e)}")
            return {
                'format': 'excel',
                'error': f"Excel processing failed: {str(e)}",
                'summary': 'Failed to process Excel file',
                'record_count': 0,
                'schema': {}
            }

    def _extract_excel_text_content(self, df: pd.DataFrame) -> str:
        """Extract text content from Excel DataFrame for NLP analysis"""
        try:
            text_parts = []

            # Extract text from string columns
            for col in df.columns:
                if df[col].dtype == 'object':  # Likely text column
                    text_values = df[col].dropna().astype(str).tolist()
                    # Filter out numeric-looking strings
                    text_values = [val for val in text_values if not val.replace('.', '').replace('-', '').isdigit()]
                    if text_values:
                        text_parts.extend(text_values[:20])  # Limit to first 20 values per column

            return ' '.join(text_parts) if text_parts else ""
        except Exception as e:
            print(f"⚠️ Error extracting Excel text content: {e}")
            return ""

    def _process_json(self, file_path: str, process_full_dataset: bool = False):
        """Process a JSON file to extract schema and sample data with enhanced NLP support"""
        try:
            print(f"🔄 Processing JSON file: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # If it's a list of records
            if isinstance(data, list) and len(data) > 0:
                print(f"📊 Processing JSON array with {len(data)} records")

                # Determine processing scope
                if process_full_dataset and len(data) <= 10000:
                    processed_data = data
                    sample_data = data[:10]
                    print(f"📊 Processing full JSON dataset with {len(data)} records")
                elif process_full_dataset and len(data) > 10000:
                    processed_data = data[:5000]  # Process first 5000 for large datasets
                    sample_data = data[:10]
                    print(f"📊 Processing large JSON dataset: first 5000 of {len(data)} records")
                else:
                    processed_data = data[:100]  # Sample processing
                    sample_data = data[:10]
                    print(f"📊 Processing JSON sample: {len(processed_data)} records from {len(data)} total")

                # Extract schema from multiple items for better accuracy
                schema = {}
                first_item = data[0]

                if isinstance(first_item, dict):
                    # Analyze schema from multiple records
                    for key in first_item.keys():
                        values = [item.get(key) for item in processed_data[:20] if item.get(key) is not None]
                        if values:
                            value_types = list(set([type(v).__name__ for v in values]))
                            schema[key] = {
                                'type': value_types[0] if len(value_types) == 1 else 'mixed',
                                'sample_values': values[:5],
                                'null_count': sum(1 for item in processed_data if item.get(key) is None),
                                'unique_count': len(set(str(v) for v in values))
                            }

                # Extract text content for NLP analysis
                text_content = self._extract_json_text_content(processed_data)

                result = {
                    'format': 'json',
                    'record_count': len(data),
                    'processed_records': len(processed_data),
                    'is_array': True,
                    'schema': schema,
                    'sample_data': sample_data,
                    'processing_mode': 'full' if process_full_dataset else 'sample',
                    'data_summary': {
                        'total_records': len(data),
                        'processed_records': len(processed_data),
                        'total_fields': len(schema),
                        'structure_type': 'array_of_objects'
                    }
                }

                # Add text content for NLP
                if text_content:
                    result['text_content'] = text_content

                # Add full data for small datasets
                if process_full_dataset and len(data) <= 1000:
                    result['full_data'] = data

                return result
            else:
                # Treat as a single object
                print("📊 Processing JSON as single object")
                text_content = self._extract_json_text_content([data] if isinstance(data, dict) else data)

                result = {
                    'format': 'json',
                    'is_array': False,
                    'structure': self._get_json_structure(data),
                    'sample_data': data,
                    'record_count': 1,
                    'data_summary': {
                        'structure_type': 'single_object',
                        'complexity': self._calculate_json_complexity(data)
                    }
                }

                if text_content:
                    result['text_content'] = text_content

                return result
        except Exception as e:
            print(f"❌ Error processing JSON: {str(e)}")
            return {
                'format': 'json',
                'error': str(e),
                'summary': 'Failed to process JSON file',
                'record_count': 0,
                'schema': {}
            }

    def _extract_json_text_content(self, data) -> str:
        """Extract text content from JSON data for NLP analysis"""
        try:
            text_parts = []

            if isinstance(data, list):
                # Process list of objects
                for item in data[:50]:  # Limit to first 50 items
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > 3:
                                # Filter out IDs, codes, and numeric strings
                                if not value.replace('-', '').replace('_', '').isalnum() or len(value) > 10:
                                    text_parts.append(value)
                    elif isinstance(item, str) and len(item) > 3:
                        text_parts.append(item)
            elif isinstance(data, dict):
                # Process single object
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 3:
                        text_parts.append(value)
                    elif isinstance(value, list):
                        for sub_item in value[:10]:  # Limit sub-items
                            if isinstance(sub_item, str) and len(sub_item) > 3:
                                text_parts.append(sub_item)

            return ' '.join(text_parts[:100]) if text_parts else ""  # Limit total text
        except Exception as e:
            print(f"⚠️ Error extracting JSON text content: {e}")
            return ""

    def _calculate_json_complexity(self, data, max_depth=5, current_depth=0) -> int:
        """Calculate complexity score for JSON structure"""
        try:
            if current_depth >= max_depth:
                return 1

            complexity = 0
            if isinstance(data, dict):
                complexity += len(data)
                for value in data.values():
                    complexity += self._calculate_json_complexity(value, max_depth, current_depth + 1)
            elif isinstance(data, list):
                complexity += len(data)
                if data:
                    complexity += self._calculate_json_complexity(data[0], max_depth, current_depth + 1)
            else:
                complexity = 1

            return min(complexity, 1000)  # Cap complexity score
        except:
            return 1

    def _get_json_structure(self, data, max_depth=3, current_depth=0):
        """Recursively analyze JSON structure"""
        if current_depth >= max_depth:
            return {'type': type(data).__name__}

        if isinstance(data, dict):
            structure = {'type': 'object', 'properties': {}}
            for key, value in data.items():
                structure['properties'][key] = self._get_json_structure(
                    value, max_depth, current_depth + 1)
            return structure
        elif isinstance(data, list):
            if len(data) == 0:
                return {'type': 'array', 'items': {'type': 'unknown'}}
            elif len(data) > 0:
                return {
                    'type': 'array',
                    'items': self._get_json_structure(data[0], max_depth, current_depth + 1),
                    'count': len(data)
                }
        else:
            return {
                'type': type(data).__name__,
                'sample': str(data)[:100] if isinstance(data, str) else data
            }

    def _process_xml(self, file_path):
        """Process an XML file (simplified)"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract basic structure
            structure = {
                'root_tag': root.tag,
                'attributes': root.attrib,
                'children': []
            }

            # Extract first level children
            for child in root[:10]:  # Limit to first 10 children
                child_info = {
                    'tag': child.tag,
                    'attributes': child.attrib,
                    'text': child.text.strip() if child.text else ''
                }
                structure['children'].append(child_info)

            return {
                'format': 'xml',
                'structure': structure
            }
        except Exception as e:
            print(f"Error processing XML: {str(e)}")
            return {
                'format': 'xml',
                'error': str(e),
                'summary': 'Failed to process XML file'
            }

    def _process_generic_file(self, file_path):
        """Process an unknown file format"""
        try:
            # Get file size
            size = os.path.getsize(file_path)

            # Read first few lines
            preview = ''
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    preview = ''.join(file.readline() for _ in range(10))
            except:
                # If we can't read as text, treat as binary
                with open(file_path, 'rb') as file:
                    binary_preview = file.read(100)
                    preview = f"Binary data: {binary_preview}"

            return {
                'format': 'unknown',
                'size': size,
                'preview': preview[:500]  # Limit preview size
            }
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return {
                'format': 'unknown',
                'error': str(e),
                'summary': 'Failed to process file'
            }

    def assess_quality(self, processed_data, dataset):
        """Assess the quality of the dataset and its metadata"""
        # Simple quality assessment
        quality_score = 0
        issues = []

        # Check for basic metadata
        if not dataset.description:
            issues.append("Missing dataset description")
        else:
            quality_score += 20

        if not dataset.tags:
            issues.append("Missing dataset tags")
        else:
            quality_score += 10

        # Check for source information
        if not dataset.source:
            issues.append("Missing dataset source")
        else:
            quality_score += 10

        if not dataset.source_url:
            issues.append("Missing dataset source URL")
        else:
            quality_score += 10

        # Check for categorical information
        if not dataset.category:
            issues.append("Missing dataset category")
        else:
            quality_score += 10

        if not dataset.data_type:
            issues.append("Missing dataset data type")
        else:
            quality_score += 10

        # Check for processed data quality
        if processed_data:
            if 'error' in processed_data:
                issues.append(f"Processing error: {processed_data['error']}")
            else:
                quality_score += 30
        else:
            issues.append("No processed data available")

        return {
            'score': quality_score,
            'issues': issues
        }

    def add_to_processing_queue(self, dataset_id, priority=1):
        """Add a dataset to the processing queue (simplified for MongoDB)"""
        try:
            # Check if already in queue
            existing = ProcessingQueue.objects(dataset=str(dataset_id)).first()
            if existing:
                # If not already completed or failed, just return it
                if existing.status not in ['completed', 'failed']:
                    return existing

                # Otherwise update it to re-process
                existing.update(
                    status='pending',
                    progress=0,
                    started_at=None,
                    completed_at=None,
                    error=None
                )
                return existing

            # Create new queue item
            queue_item = ProcessingQueue.create(
                dataset=str(dataset_id),
                status='pending',
                priority=priority
            )

            return queue_item
        except Exception as e:
            print(f"Error adding to processing queue: {e}")
            return None

    # Note: Complex processing methods removed for simplification
    # The manual processing in the routes handles basic quality assessment

    def index_dataset_for_search(self, dataset_id: str) -> bool:
        """
        Index a single dataset for semantic search.

        Args:
            dataset_id: ID of the dataset to index

        Returns:
            True if indexing successful, False otherwise
        """
        try:
            from app.services.semantic_search_service import get_semantic_search_service
            from app.models.dataset import Dataset

            # Get the dataset
            dataset = Dataset.objects(id=dataset_id).first()
            if not dataset:
                logger.warning(f"Dataset {dataset_id} not found for indexing")
                return False

            # Get semantic search service
            semantic_search_service = get_semantic_search_service()

            # Index the dataset
            success = semantic_search_service.index_datasets([dataset])

            if success:
                logger.info(f"Successfully indexed dataset {dataset_id} for semantic search")
            else:
                logger.warning(f"Failed to index dataset {dataset_id} for semantic search")

            return success

        except Exception as e:
            logger.error(f"Error indexing dataset {dataset_id} for search: {e}")
            return False

    def reindex_all_datasets(self) -> bool:
        """
        Reindex all datasets for semantic search.

        Returns:
            True if indexing successful, False otherwise
        """
        try:
            from app.services.semantic_search_service import get_semantic_search_service
            from app.models.dataset import Dataset

            # Get all datasets
            datasets = list(Dataset.objects())

            if not datasets:
                logger.info("No datasets found to index")
                return True

            # Get semantic search service
            semantic_search_service = get_semantic_search_service()

            # Index all datasets
            success = semantic_search_service.index_datasets(datasets)

            if success:
                logger.info(f"Successfully indexed {len(datasets)} datasets for semantic search")
            else:
                logger.warning("Failed to index datasets for semantic search")

            return success

        except Exception as e:
            logger.error(f"Error reindexing all datasets for search: {e}")
            return False

    # Note: Search functionality removed - uses SQLAlchemy which is not compatible with MongoDB

    def _process_zip(self, file_path):
        """Process a ZIP file containing multiple datasets"""
        try:
            # First validate the zip file
            is_valid, error_msg, zip_info = validate_zip_contents(file_path)
            if not is_valid:
                return {
                    'format': 'zip',
                    'error': error_msg,
                    'summary': 'Invalid ZIP file'
                }

            # Extract the zip file
            success, extract_dir, extracted_files = extract_zip_file(file_path)
            if not success:
                return {
                    'format': 'zip',
                    'error': f"Failed to extract ZIP file: {extract_dir}",
                    'summary': 'ZIP extraction failed'
                }

            # Process each extracted dataset file
            processed_files = []
            for file_info in extracted_files:
                if file_info.get('is_dataset', False):
                    try:
                        # Create file info structure for processing
                        dataset_file_info = {
                            'path': file_info['path'],
                            'format': file_info['extension'],
                            'size': file_info['size']
                        }

                        # Process the individual file
                        processed_data = self.process_dataset(dataset_file_info, process_full_dataset=False)
                        if processed_data:
                            processed_data['original_filename'] = file_info['original_name']
                            processed_data['extracted_path'] = file_info['path']
                            processed_files.append(processed_data)
                    except Exception as e:
                        print(f"Error processing file {file_info['original_name']}: {e}")
                        continue

            # Generate collection title
            file_names = [f['original_name'] for f in extracted_files if f.get('is_dataset', False)]
            collection_title = generate_collection_title(file_names)

            return {
                'format': 'zip',
                'is_collection': True,
                'collection_title': collection_title,
                'total_files': zip_info['total_files'],
                'dataset_files': zip_info['dataset_files'],
                'total_size': zip_info['total_size'],
                'file_types': zip_info['file_types'],
                'extracted_directory': extract_dir,
                'processed_files': processed_files,
                'file_list': zip_info['dataset_files_list'],
                'summary': f"ZIP collection with {len(processed_files)} processed datasets"
            }

        except Exception as e:
            print(f"Error processing ZIP file: {str(e)}")
            return {
                'format': 'zip',
                'error': str(e),
                'summary': 'Failed to process ZIP file'
            }


# Global service instance factory
_dataset_service = None

def get_dataset_service(upload_folder):
    """Get or create the dataset service instance"""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService(upload_folder)
    return _dataset_service