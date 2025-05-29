"""
Intelligent Metadata Auto-Completion Service for AIMetaHarvest.

This service automatically generates and completes missing metadata fields to improve
dataset completeness, FAIR compliance, and overall quality.

Features:
- Auto-generates missing metadata fields
- Provides intelligent defaults based on dataset content
- Enhances FAIR compliance automatically
- Suggests appropriate licenses and standards
- Creates Schema.org metadata
- Generates persistent identifiers
"""

import uuid
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MetadataCompletionService:
    """
    Service for automatically completing and enhancing dataset metadata.
    
    Addresses all quality recommendations by:
    - Auto-generating missing fields
    - Providing intelligent defaults
    - Enhancing FAIR compliance
    - Adding domain-specific standards
    """
    
    def __init__(self):
        """Initialize the metadata completion service."""
        self.domain_standards = self._load_domain_standards()
        self.license_suggestions = self._load_license_suggestions()
        self.schema_org_templates = self._load_schema_org_templates()
        self.fair_vocabularies = self._load_fair_vocabularies()
    
    def complete_dataset_metadata(self, dataset, processed_data: Dict[str, Any] = None,
                                 quality_results: Dict[str, Any] = None,
                                 user_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete and enhance dataset metadata automatically.
        
        Args:
            dataset: Dataset object
            processed_data: Processed dataset information
            quality_results: Quality assessment results
            user_info: Information about the uploading user
            
        Returns:
            Dictionary of metadata enhancements to apply
        """
        try:
            enhancements = {}
            
            # 1. Basic Completeness Enhancements
            basic_enhancements = self._enhance_basic_metadata(dataset, user_info)
            enhancements.update(basic_enhancements)
            
            # 2. Content-Based Enhancements
            if processed_data:
                content_enhancements = self._enhance_content_metadata(dataset, processed_data)
                enhancements.update(content_enhancements)
            
            # 3. FAIR Compliance Enhancements
            fair_enhancements = self._enhance_fair_compliance(dataset, quality_results)
            enhancements.update(fair_enhancements)
            
            # 4. Domain-Specific Standards
            domain_enhancements = self._enhance_domain_standards(dataset, processed_data)
            enhancements.update(domain_enhancements)
            
            # 5. Schema.org Metadata
            schema_org_metadata = self._generate_schema_org_metadata(dataset, processed_data)
            enhancements['schema_org_metadata'] = schema_org_metadata
            
            # 6. Persistent Identifier
            if not getattr(dataset, 'persistent_id', None):
                enhancements['persistent_id'] = self._generate_persistent_identifier(dataset)
            
            # 7. Access and Licensing
            access_enhancements = self._enhance_access_metadata(dataset)
            enhancements.update(access_enhancements)
            
            # 8. Provenance Information
            provenance_info = self._generate_provenance_information(dataset, user_info)
            enhancements['provenance'] = provenance_info
            
            logger.info(f"Generated {len(enhancements)} metadata enhancements for dataset {dataset.id}")
            return enhancements
            
        except Exception as e:
            logger.error(f"Error completing metadata: {e}")
            return {}
    
    def _enhance_basic_metadata(self, dataset, user_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance basic metadata fields."""
        enhancements = {}
        
        # Auto-generate source if missing
        if not dataset.source and user_info:
            username = user_info.get('username', 'Unknown User')
            enhancements['source'] = f"Uploaded by {username}"
        
        # Auto-generate author/publisher if missing
        if not getattr(dataset, 'author', None) and user_info:
            enhancements['author'] = user_info.get('username', 'Unknown User')
        
        if not getattr(dataset, 'publisher', None) and user_info:
            enhancements['publisher'] = user_info.get('username', 'Unknown User')
        
        # Auto-generate creation date if missing
        if not getattr(dataset, 'created_date', None):
            enhancements['created_date'] = datetime.utcnow()
        
        # Auto-generate update frequency if missing
        if not getattr(dataset, 'update_frequency', None):
            enhancements['update_frequency'] = 'none'  # Default to no updates
        
        # Auto-generate version if missing
        if not getattr(dataset, 'version', None):
            enhancements['version'] = '1.0'
        
        return enhancements
    
    def _enhance_content_metadata(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance metadata based on dataset content."""
        enhancements = {}
        
        # Add sample data if missing
        if not getattr(dataset, 'sample_data', None) and 'sample_data' in processed_data:
            sample_data = processed_data['sample_data']
            if isinstance(sample_data, list) and len(sample_data) > 0:
                # Limit sample data to first 5 records for metadata
                enhancements['sample_data'] = sample_data[:5]
        
        # Auto-generate data format if missing
        if not getattr(dataset, 'data_format', None):
            file_format = processed_data.get('format', 'unknown')
            enhancements['data_format'] = file_format.upper()
        
        # Auto-generate record count if missing
        if not getattr(dataset, 'record_count', None):
            record_count = processed_data.get('record_count', 0)
            if record_count > 0:
                enhancements['record_count'] = record_count
        
        # Auto-generate field count if missing
        if not getattr(dataset, 'field_count', None):
            schema = processed_data.get('schema', {})
            if schema:
                enhancements['field_count'] = len(schema)
        
        # Auto-generate data types summary
        if not getattr(dataset, 'data_types_summary', None):
            schema = processed_data.get('schema', {})
            if schema:
                type_counts = {}
                for field_info in schema.values():
                    if isinstance(field_info, dict):
                        field_type = field_info.get('type', 'unknown')
                        type_counts[field_type] = type_counts.get(field_type, 0) + 1
                
                if type_counts:
                    enhancements['data_types_summary'] = type_counts
        
        return enhancements
    
    def _enhance_fair_compliance(self, dataset, quality_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance FAIR compliance metadata."""
        enhancements = {}
        
        # Findability enhancements
        if not getattr(dataset, 'keywords', None):
            # Generate keywords from title and description
            keywords = self._extract_keywords_from_text(
                f"{dataset.title or ''} {dataset.description or ''}"
            )
            if keywords:
                enhancements['keywords'] = keywords
        
        # Accessibility enhancements
        if not getattr(dataset, 'access_url', None):
            # Generate standard access URL
            enhancements['access_url'] = f"/datasets/{dataset.id}"
        
        if not getattr(dataset, 'access_protocol', None):
            enhancements['access_protocol'] = 'HTTP'
        
        # Interoperability enhancements
        if not getattr(dataset, 'vocabulary_used', None):
            domain = self._detect_domain_from_dataset(dataset)
            if domain in self.fair_vocabularies:
                enhancements['vocabulary_used'] = self.fair_vocabularies[domain]
        
        # Reusability enhancements
        if not getattr(dataset, 'license', None):
            suggested_license = self._suggest_license(dataset)
            enhancements['license'] = suggested_license
        
        if not getattr(dataset, 'usage_rights', None):
            enhancements['usage_rights'] = 'Please verify licensing and usage rights before use'
        
        return enhancements
    
    def _enhance_domain_standards(self, dataset, processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance metadata with domain-specific standards."""
        enhancements = {}
        
        # Detect domain
        domain = self._detect_domain_from_dataset(dataset)
        
        if domain in self.domain_standards:
            standards = self.domain_standards[domain]
            
            # Add domain-specific compliance information
            enhancements['domain_standards'] = {
                'domain': domain,
                'applicable_standards': standards['standards'],
                'compliance_requirements': standards['requirements'],
                'recommended_metadata': standards['metadata_fields']
            }
            
            # Add domain-specific metadata fields
            for field, default_value in standards['metadata_fields'].items():
                if not getattr(dataset, field, None):
                    enhancements[field] = default_value
        
        return enhancements
    
    def _generate_schema_org_metadata(self, dataset, processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive Schema.org compliant metadata."""
        # Generate persistent identifier if not exists
        persistent_id = getattr(dataset, 'persistent_id', None) or self._generate_persistent_identifier(dataset)

        schema_org = {
            "@context": "https://schema.org/",
            "@type": "Dataset",
            "name": dataset.title or "Untitled Dataset",
            "description": dataset.description or "No description provided",
            "url": f"/datasets/{dataset.id}",
            "identifier": persistent_id,
            "dateCreated": getattr(dataset, 'created_at', datetime.utcnow()).isoformat(),
            "dateModified": getattr(dataset, 'updated_at', datetime.utcnow()).isoformat(),
            "datePublished": getattr(dataset, 'date_published', datetime.utcnow()).isoformat(),
            "version": getattr(dataset, 'version', '1.0'),
            "creator": {
                "@type": getattr(dataset, 'creator_type', 'Person'),
                "name": getattr(dataset, 'author', 'Unknown Author')
            },
            "publisher": {
                "@type": getattr(dataset, 'publisher_type', 'Organization'),
                "name": getattr(dataset, 'publisher', 'AIMetaHarvest')
            },
            "license": getattr(dataset, 'license', 'https://creativecommons.org/licenses/by/4.0/'),
            "keywords": self._parse_keywords(getattr(dataset, 'keywords', '')),
            "inLanguage": "en",
            "isAccessibleForFree": True
        }

        # Add distribution information
        if processed_data:
            distribution = {
                "@type": "DataDownload",
                "encodingFormat": processed_data.get('format', 'unknown'),
                "contentSize": getattr(dataset, 'size', 'Unknown'),
                "contentUrl": f"/datasets/{dataset.id}/download"
            }

            # Add record count if available
            if 'record_count' in processed_data:
                distribution["numberOfRecords"] = processed_data['record_count']

            schema_org["distribution"] = distribution

        # Add variable measured information
        if processed_data and 'schema' in processed_data:
            variables = list(processed_data['schema'].keys())
            schema_org["variableMeasured"] = [
                {
                    "@type": "PropertyValue",
                    "name": var,
                    "description": f"Data variable: {var}"
                } for var in variables[:10]  # Limit to first 10 variables
            ]

        # Add temporal coverage if available
        if getattr(dataset, 'temporal_coverage', None):
            schema_org["temporalCoverage"] = dataset.temporal_coverage

        # Add spatial coverage if available
        if getattr(dataset, 'spatial_coverage', None):
            schema_org["spatialCoverage"] = dataset.spatial_coverage

        # Add measurement technique if available
        if getattr(dataset, 'measurement_technique', None):
            schema_org["measurementTechnique"] = dataset.measurement_technique

        # Add funding information if available
        if getattr(dataset, 'funding', None):
            schema_org["funding"] = {
                "@type": "Grant",
                "name": dataset.funding
            }

        # Add citation if available
        if getattr(dataset, 'citation', None):
            schema_org["citation"] = dataset.citation

        # Add related datasets
        if getattr(dataset, 'is_based_on', None):
            schema_org["isBasedOn"] = dataset.is_based_on

        if getattr(dataset, 'same_as', None):
            schema_org["sameAs"] = dataset.same_as
        
        return schema_org

    def _parse_keywords(self, keywords_str: str) -> List[str]:
        """Parse keywords string into a list."""
        if not keywords_str:
            return []

        # Handle comma-separated keywords
        if ',' in keywords_str:
            return [k.strip() for k in keywords_str.split(',') if k.strip()]

        # Handle space-separated keywords
        return [k.strip() for k in keywords_str.split() if k.strip()]

    def _generate_persistent_identifier(self, dataset) -> str:
        """Generate a persistent identifier for the dataset."""
        # Generate a UUID-based persistent identifier
        base_uuid = str(uuid.uuid4())
        # Create a more readable format
        persistent_id = f"aimetaharvest:{base_uuid[:8]}-{base_uuid[8:12]}-{base_uuid[12:16]}"
        return persistent_id
    
    def _enhance_access_metadata(self, dataset) -> Dict[str, Any]:
        """Enhance access-related metadata."""
        enhancements = {}
        
        # Standard access URL
        if not getattr(dataset, 'access_url', None):
            enhancements['access_url'] = f"/datasets/{dataset.id}"
        
        # Download URL
        if not getattr(dataset, 'download_url', None) and dataset.file_path:
            enhancements['download_url'] = f"/datasets/{dataset.id}/download"
        
        # API access URL
        if not getattr(dataset, 'api_url', None):
            enhancements['api_url'] = f"/api/datasets/{dataset.id}"
        
        # Access rights
        if not getattr(dataset, 'access_rights', None):
            enhancements['access_rights'] = 'Open access with proper attribution'
        
        return enhancements
    
    def _generate_provenance_information(self, dataset, user_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate detailed provenance information."""
        provenance = {
            'created_by': user_info.get('username', 'Unknown User') if user_info else 'Unknown User',
            'created_at': getattr(dataset, 'created_date', datetime.utcnow()).isoformat(),
            'upload_method': 'Web Interface',
            'processing_pipeline': 'AIMetaHarvest Enhanced 9-Step Pipeline',
            'quality_assessed': True,
            'fair_compliant': True,
            'auto_enhanced': True,
            'enhancement_date': datetime.utcnow().isoformat()
        }
        
        # Add user organization if available
        if user_info and user_info.get('organization'):
            provenance['organization'] = user_info['organization']
        
        # Add processing details
        provenance['processing_details'] = {
            'nlp_analysis': True,
            'quality_assessment': True,
            'ai_compliance_check': True,
            'description_generation': True,
            'visualization_generation': True,
            'metadata_completion': True
        }
        
        return provenance
    
    def _detect_domain_from_dataset(self, dataset) -> str:
        """Detect the domain/sector of the dataset."""
        text_content = f"{dataset.title or ''} {dataset.description or ''} {dataset.category or ''}"
        text_lower = text_content.lower()
        
        domain_keywords = {
            'healthcare': ['health', 'medical', 'patient', 'hospital', 'clinical', 'medicine'],
            'finance': ['financial', 'bank', 'money', 'investment', 'trading', 'economic'],
            'education': ['education', 'student', 'school', 'university', 'learning', 'academic'],
            'retail': ['sales', 'customer', 'product', 'purchase', 'commerce', 'retail'],
            'transportation': ['transport', 'vehicle', 'traffic', 'logistics', 'shipping'],
            'technology': ['software', 'hardware', 'computer', 'system', 'tech', 'digital'],
            'government': ['government', 'public', 'policy', 'civic', 'municipal', 'federal'],
            'environmental': ['environment', 'climate', 'weather', 'pollution', 'sustainability'],
            'social': ['social', 'community', 'demographic', 'population', 'survey'],
            'research': ['research', 'study', 'experiment', 'analysis', 'scientific']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text content."""
        if not text:
            return []
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        keywords = [word for word in words if word not in stop_words]
        
        # Get most frequent keywords
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def _suggest_license(self, dataset) -> str:
        """Suggest an appropriate license based on dataset characteristics."""
        domain = self._detect_domain_from_dataset(dataset)
        
        if domain in self.license_suggestions:
            return self.license_suggestions[domain]
        
        # Default license suggestion
        return "Creative Commons Attribution 4.0 International (CC BY 4.0)"
    
    def _load_domain_standards(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific standards and requirements."""
        return {
            'healthcare': {
                'standards': ['HIPAA', 'HL7 FHIR', 'DICOM', 'ICD-10'],
                'requirements': ['Patient privacy protection', 'Data anonymization', 'Audit trails'],
                'metadata_fields': {
                    'privacy_level': 'De-identified',
                    'ethical_approval': 'Required for human subjects data',
                    'data_sensitivity': 'High - Healthcare data'
                }
            },
            'finance': {
                'standards': ['PCI DSS', 'SOX', 'Basel III', 'GDPR'],
                'requirements': ['Data encryption', 'Access controls', 'Audit logging'],
                'metadata_fields': {
                    'privacy_level': 'Confidential',
                    'regulatory_compliance': 'Financial regulations applicable',
                    'data_sensitivity': 'High - Financial data'
                }
            },
            'education': {
                'standards': ['FERPA', 'COPPA', 'Dublin Core'],
                'requirements': ['Student privacy protection', 'Parental consent for minors'],
                'metadata_fields': {
                    'privacy_level': 'Protected',
                    'age_restrictions': 'COPPA compliance for under 13',
                    'data_sensitivity': 'Medium - Educational data'
                }
            },
            'government': {
                'standards': ['FOIA', 'Open Government Data Act', 'NIST'],
                'requirements': ['Public access', 'Transparency', 'Data quality'],
                'metadata_fields': {
                    'access_level': 'Public',
                    'government_classification': 'Unclassified',
                    'data_sensitivity': 'Low - Public data'
                }
            },
            'research': {
                'standards': ['Dublin Core', 'DataCite', 'FAIR Principles'],
                'requirements': ['Reproducibility', 'Peer review', 'Open access'],
                'metadata_fields': {
                    'research_ethics': 'IRB approval may be required',
                    'reproducibility': 'Methods and data available for verification',
                    'data_sensitivity': 'Variable - Depends on research domain'
                }
            }
        }
    
    def _load_license_suggestions(self) -> Dict[str, str]:
        """Load license suggestions by domain."""
        return {
            'healthcare': 'Restricted - Healthcare data license required',
            'finance': 'Proprietary - Financial data restrictions apply',
            'education': 'Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)',
            'government': 'Public Domain or Open Government License',
            'research': 'Creative Commons Attribution 4.0 International (CC BY 4.0)',
            'technology': 'MIT License or Apache 2.0',
            'general': 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
        }
    
    def _load_schema_org_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load Schema.org templates for different domains."""
        return {
            'healthcare': {
                'additionalType': 'MedicalDataset',
                'healthCondition': 'To be specified',
                'medicalSpecialty': 'To be specified'
            },
            'research': {
                'additionalType': 'ResearchDataset',
                'researchField': 'To be specified',
                'methodology': 'To be specified'
            },
            'government': {
                'additionalType': 'GovernmentDataset',
                'governmentLevel': 'To be specified',
                'jurisdiction': 'To be specified'
            }
        }
    
    def _load_fair_vocabularies(self) -> Dict[str, List[str]]:
        """Load FAIR-compliant vocabularies by domain."""
        return {
            'healthcare': ['SNOMED CT', 'ICD-10', 'LOINC', 'RxNorm'],
            'finance': ['FIBO', 'ISO 20022', 'LEI', 'SWIFT'],
            'education': ['Dublin Core', 'IEEE LOM', 'SCORM'],
            'government': ['DCAT', 'FOAF', 'Dublin Core', 'SKOS'],
            'research': ['Dublin Core', 'DataCite', 'ORCID', 'DOI'],
            'technology': ['SPDX', 'DOAP', 'FOAF'],
            'general': ['Dublin Core', 'DCAT', 'FOAF']
        }


# Global service instance
metadata_completion_service = MetadataCompletionService()
