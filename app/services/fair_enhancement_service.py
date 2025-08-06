"""
FAIR Enhancement Service

This service automatically improves dataset FAIR compliance and Schema.org metadata.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from app.models.dataset import Dataset
from app.services.quality_scoring.dimension_scorers import ConformityScorer

logger = logging.getLogger(__name__)


class FAIREnhancementService:
    """Service for automatically enhancing FAIR compliance and Schema.org metadata."""
    
    def __init__(self):
        self.conformity_scorer = ConformityScorer()
    
    def enhance_dataset_fair_compliance(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Automatically enhance a dataset's FAIR compliance.
        
        Args:
            dataset: Dataset to enhance
            
        Returns:
            Dictionary with enhancement results
        """
        try:
            logger.info(f"Starting FAIR enhancement for dataset {dataset.id}")
            
            # Get current FAIR assessment
            current_fair = self._assess_current_fair_compliance(dataset)
            
            # Apply enhancements
            enhancements = {
                'findability': self._enhance_findability(dataset),
                'accessibility': self._enhance_accessibility(dataset),
                'interoperability': self._enhance_interoperability(dataset),
                'reusability': self._enhance_reusability(dataset)
            }
            
            # Generate Schema.org metadata
            schema_org_metadata = self._generate_schema_org_metadata(dataset)
            if schema_org_metadata:
                dataset.schema_org = json.dumps(schema_org_metadata)
                dataset.schema_org_json = json.dumps(schema_org_metadata)
                enhancements['schema_org'] = True
            
            # Generate persistent identifier if missing
            if not hasattr(dataset, 'persistent_id') or not dataset.persistent_id:
                persistent_id = self._generate_persistent_identifier(dataset)
                dataset.persistent_id = persistent_id
                enhancements['persistent_id'] = persistent_id
            
            # Save changes
            dataset.save()
            
            # Get new FAIR assessment
            new_fair = self._assess_current_fair_compliance(dataset)
            
            result = {
                'success': True,
                'dataset_id': str(dataset.id),
                'before_fair_score': current_fair['fair_score'],
                'after_fair_score': new_fair['fair_score'],
                'improvement': new_fair['fair_score'] - current_fair['fair_score'],
                'enhancements_applied': enhancements,
                'recommendations': new_fair['recommendations'],
                'enhanced_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"FAIR enhancement completed for dataset {dataset.id}. "
                       f"Score improved from {current_fair['fair_score']:.1f} to {new_fair['fair_score']:.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing FAIR compliance for dataset {dataset.id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'dataset_id': str(dataset.id)
            }
    
    def _assess_current_fair_compliance(self, dataset: Dataset) -> Dict[str, Any]:
        """Assess current FAIR compliance of a dataset."""
        try:
            fair_score, fair_details = self.conformity_scorer._calculate_fair_compliance(dataset)
            return {
                'fair_score': fair_score,
                'fair_details': fair_details,
                'recommendations': self.conformity_scorer.recommendations.copy()
            }
        except Exception as e:
            logger.error(f"Error assessing FAIR compliance: {e}")
            return {
                'fair_score': 0.0,
                'fair_details': {'findable': 0, 'accessible': 0, 'interoperable': 0, 'reusable': 0},
                'recommendations': []
            }
    
    def _enhance_findability(self, dataset: Dataset) -> Dict[str, Any]:
        """Enhance dataset findability."""
        enhancements = {}
        
        # Ensure title is descriptive
        if not dataset.title or len(dataset.title.strip()) < 10:
            if dataset.file_path:
                import os
                filename = os.path.basename(dataset.file_path)
                dataset.title = f"Dataset: {filename}"
                enhancements['title'] = 'Generated descriptive title'
        
        # Ensure description exists and is detailed
        if not dataset.description or len(dataset.description.strip()) < 50:
            generated_desc = self._generate_basic_description(dataset)
            if generated_desc:
                dataset.description = generated_desc
                enhancements['description'] = 'Generated basic description'
        
        # Ensure tags exist
        if not dataset.tags or (isinstance(dataset.tags, list) and len(dataset.tags) == 0):
            generated_tags = self._generate_basic_tags(dataset)
            if generated_tags:
                dataset.tags = generated_tags
                enhancements['tags'] = f'Generated {len(generated_tags)} tags'
        
        # Ensure category is set
        if not dataset.category:
            dataset.category = self._infer_category(dataset)
            enhancements['category'] = 'Inferred category'
        
        return enhancements
    
    def _enhance_accessibility(self, dataset: Dataset) -> Dict[str, Any]:
        """Enhance dataset accessibility."""
        enhancements = {}
        
        # Ensure access URL is available
        if not hasattr(dataset, 'access_url') or not dataset.access_url:
            dataset.access_url = f"/datasets/{dataset.id}"
            enhancements['access_url'] = 'Generated access URL'
        
        # Ensure download URL is available
        if not hasattr(dataset, 'download_url') or not dataset.download_url:
            dataset.download_url = f"/datasets/{dataset.id}/download"
            enhancements['download_url'] = 'Generated download URL'
        
        # Set access rights
        if not hasattr(dataset, 'access_rights') or not dataset.access_rights:
            dataset.access_rights = 'Open access with proper attribution'
            enhancements['access_rights'] = 'Set default access rights'
        
        return enhancements
    
    def _enhance_interoperability(self, dataset: Dataset) -> Dict[str, Any]:
        """Enhance dataset interoperability."""
        enhancements = {}
        
        # Ensure format is specified
        if not dataset.format and dataset.file_path:
            import os
            ext = os.path.splitext(dataset.file_path)[1][1:].lower()
            if ext:
                dataset.format = ext
                enhancements['format'] = f'Detected format: {ext}'
        
        # Ensure data type is specified
        if not dataset.data_type:
            dataset.data_type = self._infer_data_type(dataset)
            enhancements['data_type'] = 'Inferred data type'
        
        return enhancements
    
    def _enhance_reusability(self, dataset: Dataset) -> Dict[str, Any]:
        """Enhance dataset reusability."""
        enhancements = {}
        
        # Set default license if missing
        if not hasattr(dataset, 'license') or not dataset.license:
            dataset.license = 'Open Data Commons Open Database License (ODbL)'
            enhancements['license'] = 'Set default open license'
        
        # Ensure source is documented
        if not dataset.source:
            if dataset.user:
                dataset.source = f"Uploaded by {dataset.user.username}"
                enhancements['source'] = 'Documented data source'
        
        return enhancements
    
    def _generate_schema_org_metadata(self, dataset: Dataset) -> Dict[str, Any]:
        """Generate Schema.org compliant metadata."""
        try:
            schema_org = {
                "@context": "https://schema.org/",
                "@type": "Dataset",
                "@id": f"/datasets/{dataset.id}",
                "name": dataset.title or "Untitled Dataset",
                "description": dataset.description or "No description available",
                "url": f"/datasets/{dataset.id}",
                "dateCreated": dataset.created_at.isoformat() if dataset.created_at else datetime.utcnow().isoformat(),
                "dateModified": dataset.updated_at.isoformat() if dataset.updated_at else datetime.utcnow().isoformat(),
                "creator": {
                    "@type": "Person",
                    "name": dataset.user.username if dataset.user else "Unknown"
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "AI Meta Harvest"
                },
                "license": getattr(dataset, 'license', 'Open Data Commons Open Database License (ODbL)'),
                "keywords": dataset.tags if isinstance(dataset.tags, list) else [dataset.tags] if dataset.tags else [],
                "distribution": {
                    "@type": "DataDownload",
                    "encodingFormat": dataset.format or "unknown",
                    "contentUrl": f"/datasets/{dataset.id}/download"
                }
            }
            
            # Add optional fields if available
            if dataset.category:
                schema_org["about"] = dataset.category
            
            if hasattr(dataset, 'record_count') and dataset.record_count:
                schema_org["size"] = f"{dataset.record_count} records"
            
            return schema_org
            
        except Exception as e:
            logger.error(f"Error generating Schema.org metadata: {e}")
            return None
    
    def _generate_persistent_identifier(self, dataset: Dataset) -> str:
        """Generate a persistent identifier for the dataset."""
        # For now, use a DOI-like format with the dataset ID
        return f"doi:10.5555/aimetaharvest.{dataset.id}"
    
    def _generate_basic_description(self, dataset: Dataset) -> str:
        """Generate a basic description for the dataset."""
        parts = []
        
        if dataset.format:
            parts.append(f"This is a {dataset.format.upper()} dataset")
        else:
            parts.append("This dataset")
        
        if dataset.category:
            parts.append(f"in the {dataset.category} category")
        
        if hasattr(dataset, 'record_count') and dataset.record_count:
            parts.append(f"containing {dataset.record_count} records")
        
        if hasattr(dataset, 'field_count') and dataset.field_count:
            parts.append(f"with {dataset.field_count} fields")
        
        if dataset.source:
            parts.append(f"sourced from {dataset.source}")
        
        description = " ".join(parts) + "."
        
        # Add usage information
        description += " This dataset is available for research and analysis purposes."
        
        return description
    
    def _generate_basic_tags(self, dataset: Dataset) -> List[str]:
        """Generate basic tags for the dataset."""
        tags = set()
        
        # Add format-based tags
        if dataset.format:
            tags.add(dataset.format.lower())
            if dataset.format.lower() in ['csv', 'tsv']:
                tags.add('tabular')
            elif dataset.format.lower() in ['json', 'xml']:
                tags.add('structured')
        
        # Add category-based tags
        if dataset.category:
            tags.add(dataset.category.lower())
        
        # Add data type tags
        if dataset.data_type:
            tags.add(dataset.data_type.lower())
        
        # Add generic tags
        tags.update(['dataset', 'data', 'research'])
        
        return list(tags)
    
    def _infer_category(self, dataset: Dataset) -> str:
        """Infer category from dataset characteristics."""
        if dataset.format in ['csv', 'xlsx', 'xls']:
            return 'tabular'
        elif dataset.format in ['json', 'xml']:
            return 'structured'
        elif dataset.format in ['txt', 'text']:
            return 'text'
        else:
            return 'general'
    
    def _infer_data_type(self, dataset: Dataset) -> str:
        """Infer data type from dataset characteristics."""
        if dataset.format in ['csv', 'xlsx', 'xls', 'tsv']:
            return 'tabular'
        elif dataset.format in ['json']:
            return 'json'
        elif dataset.format in ['xml']:
            return 'xml'
        elif dataset.format in ['txt', 'text']:
            return 'text'
        else:
            return 'mixed'


# Global service instance
fair_enhancement_service = FAIREnhancementService()


def get_fair_enhancement_service() -> FAIREnhancementService:
    """Get the global FAIR enhancement service instance."""
    return fair_enhancement_service
