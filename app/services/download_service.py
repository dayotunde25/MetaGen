"""
Download Service for creating zip files of datasets and collections.
"""

import os
import json
import zipfile
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import current_app
from app.models.dataset import Dataset


class DownloadService:
    """Service for creating downloadable zip files of datasets and collections"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def create_dataset_zip(self, dataset_id: str) -> Optional[str]:
        """
        Create a zip file containing dataset and its metadata.
        
        Args:
            dataset_id: ID of the dataset to download
            
        Returns:
            Path to the created zip file or None if failed
        """
        try:
            dataset = Dataset.find_by_id(dataset_id)
            if not dataset:
                return None
            
            # Create temporary directory for zip contents
            temp_zip_dir = os.path.join(self.temp_dir, f"dataset_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(temp_zip_dir, exist_ok=True)
            
            # Create zip file path
            zip_filename = f"{dataset.title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(self.temp_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add original dataset file if it exists
                if dataset.file_path and os.path.exists(dataset.file_path):
                    original_filename = os.path.basename(dataset.file_path)
                    zipf.write(dataset.file_path, f"data/{original_filename}")
                
                # Add metadata as JSON
                metadata = self._extract_dataset_metadata(dataset)
                metadata_json = json.dumps(metadata, indent=2, default=str)
                zipf.writestr("metadata/dataset_metadata.json", metadata_json)
                
                # Add FAIR metadata if available
                if dataset.dublin_core:
                    zipf.writestr("metadata/dublin_core.json", dataset.dublin_core)
                
                if dataset.dcat_metadata:
                    zipf.writestr("metadata/dcat_metadata.json", dataset.dcat_metadata)
                
                if dataset.json_ld:
                    zipf.writestr("metadata/schema_org_jsonld.json", dataset.json_ld)
                
                # Add quality report if available
                quality_report = self._generate_quality_report(dataset)
                if quality_report:
                    zipf.writestr("reports/quality_report.json", json.dumps(quality_report, indent=2, default=str))
                
                # Add visualizations if available
                if dataset.visualizations:
                    zipf.writestr("visualizations/charts_data.json", dataset.visualizations)
                
                # Add README file
                readme_content = self._generate_readme(dataset)
                zipf.writestr("README.md", readme_content)
            
            # Clean up temp directory
            if os.path.exists(temp_zip_dir):
                shutil.rmtree(temp_zip_dir)
            
            return zip_path
            
        except Exception as e:
            print(f"Error creating dataset zip: {e}")
            return None
    
    def create_collection_zip(self, collection_id: str) -> Optional[str]:
        """
        Create a zip file containing all datasets in a collection.
        
        Args:
            collection_id: ID of the collection to download
            
        Returns:
            Path to the created zip file or None if failed
        """
        try:
            # Find the parent collection dataset
            parent_dataset = Dataset.find_by_collection_id(collection_id)
            if not parent_dataset:
                return None
            
            # Find all datasets in the collection
            collection_datasets = Dataset.get_collection_datasets(collection_id)
            
            # Create zip file path
            zip_filename = f"{parent_dataset.title.replace(' ', '_')}_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(self.temp_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add collection metadata
                collection_metadata = self._extract_collection_metadata(parent_dataset, collection_datasets)
                zipf.writestr("collection_metadata.json", json.dumps(collection_metadata, indent=2, default=str))
                
                # Add individual datasets
                for i, dataset in enumerate(collection_datasets, 1):
                    dataset_folder = f"datasets/{i:02d}_{dataset.title.replace(' ', '_')}"
                    
                    # Add dataset file if it exists
                    if dataset.file_path and os.path.exists(dataset.file_path):
                        original_filename = os.path.basename(dataset.file_path)
                        zipf.write(dataset.file_path, f"{dataset_folder}/data/{original_filename}")
                    
                    # Add dataset metadata
                    dataset_metadata = self._extract_dataset_metadata(dataset)
                    zipf.writestr(f"{dataset_folder}/metadata.json", json.dumps(dataset_metadata, indent=2, default=str))
                    
                    # Add visualizations if available
                    if dataset.visualizations:
                        zipf.writestr(f"{dataset_folder}/visualizations.json", dataset.visualizations)
                
                # Add collection README
                readme_content = self._generate_collection_readme(parent_dataset, collection_datasets)
                zipf.writestr("README.md", readme_content)
            
            return zip_path
            
        except Exception as e:
            print(f"Error creating collection zip: {e}")
            return None
    
    def _extract_dataset_metadata(self, dataset: Dataset) -> Dict[str, Any]:
        """Extract comprehensive metadata from a dataset"""
        return {
            "basic_info": {
                "id": str(dataset.id),
                "title": dataset.title,
                "description": dataset.description,
                "source": dataset.source,
                "source_url": dataset.source_url,
                "category": dataset.category,
                "data_type": dataset.data_type,
                "format": dataset.format,
                "size": dataset.size,
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None
            },
            "content_info": {
                "record_count": dataset.record_count,
                "field_count": dataset.field_count,
                "field_names": dataset.field_names.split(',') if dataset.field_names else [],
                "data_types": dataset.data_types.split(',') if dataset.data_types else [],
                "keywords": dataset.keywords.split(',') if dataset.keywords else [],
                "tags": dataset.tags.split(',') if dataset.tags else [],
                "use_cases": dataset.use_cases.split(',') if dataset.use_cases else []
            },
            "quality_metrics": {
                "fair_score": dataset.fair_score,
                "fair_compliant": dataset.fair_compliant,
                "quality_score": getattr(dataset, 'quality_score', None),
                "status": dataset.status
            },
            "compliance": {
                "schema_org_compliant": getattr(dataset, 'schema_org_compliant', False),
                "persistent_identifier": dataset.persistent_identifier,
                "license": dataset.license,
                "encoding_format": dataset.encoding_format
            }
        }
    
    def _extract_collection_metadata(self, parent_dataset: Dataset, collection_datasets: List[Dataset]) -> Dict[str, Any]:
        """Extract metadata for a collection"""
        return {
            "collection_info": {
                "id": str(parent_dataset.id),
                "collection_id": parent_dataset.collection_id,
                "title": parent_dataset.title,
                "description": parent_dataset.description,
                "total_datasets": len(collection_datasets),
                "created_at": parent_dataset.created_at.isoformat() if parent_dataset.created_at else None
            },
            "datasets": [
                {
                    "id": str(dataset.id),
                    "title": dataset.title,
                    "format": dataset.format,
                    "record_count": dataset.record_count,
                    "field_count": dataset.field_count,
                    "status": dataset.status
                }
                for dataset in collection_datasets
            ],
            "aggregated_stats": {
                "total_records": sum(d.record_count or 0 for d in collection_datasets),
                "total_fields": sum(d.field_count or 0 for d in collection_datasets),
                "formats": list(set(d.format for d in collection_datasets if d.format)),
                "average_fair_score": sum(d.fair_score or 0 for d in collection_datasets) / len(collection_datasets) if collection_datasets else 0
            }
        }
    
    def _generate_quality_report(self, dataset: Dataset) -> Optional[Dict[str, Any]]:
        """Generate a quality report for the dataset"""
        try:
            return {
                "dataset_id": str(dataset.id),
                "title": dataset.title,
                "quality_assessment": {
                    "fair_score": dataset.fair_score,
                    "fair_compliant": dataset.fair_compliant,
                    "completeness": getattr(dataset, 'completeness', None),
                    "consistency": getattr(dataset, 'consistency', None),
                    "accuracy": getattr(dataset, 'accuracy', None)
                },
                "data_health": {
                    "record_count": dataset.record_count,
                    "field_count": dataset.field_count,
                    "missing_values": getattr(dataset, 'missing_values', None),
                    "duplicate_records": getattr(dataset, 'duplicate_records', None)
                },
                "generated_at": datetime.now().isoformat()
            }
        except Exception:
            return None
    
    def _generate_readme(self, dataset: Dataset) -> str:
        """Generate README content for dataset download"""
        return f"""# {dataset.title}

## Dataset Information
- **Source**: {dataset.source or 'Not specified'}
- **Category**: {dataset.category or 'Not specified'}
- **Format**: {dataset.format or 'Unknown'}
- **Records**: {dataset.record_count or 'Unknown'}
- **Fields**: {dataset.field_count or 'Unknown'}

## Description
{dataset.description or 'No description available.'}

## Files Included
- `data/`: Original dataset file(s)
- `metadata/`: Comprehensive metadata in various formats
- `reports/`: Quality and health reports
- `visualizations/`: Chart data and visualizations (if available)

## Metadata Standards
This dataset package includes metadata compliant with:
- Schema.org JSON-LD
- Dublin Core
- DCAT (Data Catalog Vocabulary)
- FAIR principles

## Generated
Downloaded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
From:  Metadata Harvesting System
"""
    
    def _generate_collection_readme(self, parent_dataset: Dataset, collection_datasets: List[Dataset]) -> str:
        """Generate README content for collection download"""
        return f"""# {parent_dataset.title}

## Collection Information
- **Total Datasets**: {len(collection_datasets)}
- **Collection Type**: Multi-dataset archive
- **Total Records**: {sum(d.record_count or 0 for d in collection_datasets)}
- **Total Fields**: {sum(d.field_count or 0 for d in collection_datasets)}

## Description
{parent_dataset.description or 'No description available.'}

## Included Datasets
{chr(10).join(f"- {i+1:02d}. {dataset.title} ({dataset.format}, {dataset.record_count or 0} records)" for i, dataset in enumerate(collection_datasets))}

## Structure
- `collection_metadata.json`: Overall collection metadata
- `datasets/`: Individual dataset folders
  - Each dataset folder contains:
    - `data/`: Original dataset file
    - `metadata.json`: Dataset-specific metadata
    - `visualizations.json`: Chart data (if available)

## Generated
Downloaded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
From:  Metadata Harvesting System
"""


def get_download_service():
    """Get download service instance"""
    return DownloadService()
