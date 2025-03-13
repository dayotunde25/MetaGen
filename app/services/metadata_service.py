import json
from datetime import datetime

from app.services.nlp_service import nlp_service

class MetadataService:
    """Service for generating, validating, and enhancing dataset metadata"""
    
    def __init__(self):
        """Initialize metadata service"""
        pass
    
    def generate_metadata(self, dataset, processed_data=None):
        """Generate metadata for a dataset following Schema.org and FAIR principles"""
        # Generate Schema.org metadata
        schema_org = self._generate_schema_org(dataset, processed_data)
        
        # Assess FAIR compliance
        fair_assessment = self._assess_fair_compliance(schema_org, dataset)
        
        # Calculate overall scores
        fair_scores = {
            'findable': fair_assessment['findable']['score'],
            'accessible': fair_assessment['accessible']['score'],
            'interoperable': fair_assessment['interoperable']['score'],
            'reusable': fair_assessment['reusable']['score']
        }
        
        # Determine if dataset is FAIR compliant
        fair_compliant = all(score >= 70 for score in fair_scores.values())
        
        # Calculate completeness
        completeness = self._calculate_completeness(dataset, schema_org)
        
        # Calculate consistency
        consistency = self._calculate_consistency(processed_data) if processed_data else 50
        
        # Generate recommendations
        recommendations = self._generate_recommendations(fair_assessment, dataset)
        
        return {
            'schema_org': schema_org,
            'schema_org_compliant': self._is_schema_org_compliant(schema_org),
            'fair_scores': fair_scores,
            'fair_compliant': fair_compliant,
            'completeness': completeness,
            'consistency': consistency,
            'recommendations': recommendations
        }
    
    def _generate_schema_org(self, dataset, processed_data=None):
        """Generate Schema.org compatible metadata"""
        # Basic dataset metadata
        metadata = {
            "@context": "https://schema.org/",
            "@type": "Dataset",
            "name": dataset.title,
            "description": dataset.description or "",
            "url": dataset.source_url or "",
            "identifier": str(dataset.id),
            "keywords": dataset.tags_list,
            "dateCreated": dataset.created_at.isoformat() if dataset.created_at else "",
            "dateModified": dataset.updated_at.isoformat() if dataset.updated_at else "",
            "creator": {
                "@type": "Person",
                "name": dataset.owner.username if hasattr(dataset, 'owner') and dataset.owner else "Unknown"
            },
            "publisher": {
                "@type": "Organization",
                "name": dataset.source or "Unknown Publisher"
            },
            "license": self._infer_license(dataset)
        }
        
        # Add category
        if dataset.category:
            metadata["about"] = {
                "@type": "Thing",
                "name": dataset.category
            }
        
        # Add data format
        if dataset.format:
            metadata["encodingFormat"] = dataset.format
        
        # Add size information
        if dataset.size:
            metadata["contentSize"] = dataset.size
        
        # Add variable measured from processed data
        if processed_data:
            variables = self._extract_variable_measured(processed_data)
            if variables:
                metadata["variableMeasured"] = variables
        
        # Add temporal coverage
        temporal_coverage = self._infer_temporal_coverage(dataset)
        if temporal_coverage:
            metadata["temporalCoverage"] = temporal_coverage
        
        return metadata
    
    def _extract_variable_measured(self, processed_data):
        """Extract variable measurements from processed data"""
        variables = []
        
        # Check for CSV/tabular data
        if processed_data.get('format') == 'csv' and 'schema' in processed_data:
            for col_name, col_info in processed_data['schema'].items():
                variable = {
                    "@type": "PropertyValue",
                    "name": col_name,
                    "valueType": col_info.get('type', 'Unknown'),
                }
                
                # Add sample values if available
                if 'sample_values' in col_info and col_info['sample_values']:
                    if isinstance(col_info['sample_values'], list):
                        sample_values = [str(val) for val in col_info['sample_values'] if val is not None]
                        if sample_values:
                            variable["value"] = sample_values[0]
                    
                variables.append(variable)
        
        # Check for JSON data
        elif processed_data.get('format') == 'json' and 'schema' in processed_data:
            for prop_name, prop_info in processed_data['schema'].items():
                variable = {
                    "@type": "PropertyValue",
                    "name": prop_name,
                    "valueType": prop_info.get('type', 'Unknown')
                }
                
                # Add sample values if available
                if 'sample_values' in prop_info and prop_info['sample_values']:
                    if isinstance(prop_info['sample_values'], list):
                        sample_values = [str(val) for val in prop_info['sample_values'] if val is not None]
                        if sample_values:
                            variable["value"] = sample_values[0]
                
                variables.append(variable)
        
        return variables
    
    def _infer_temporal_coverage(self, dataset):
        """Infer temporal coverage from dataset information"""
        # Extract dates from description using NLP
        if dataset.description:
            # This is a simplified approach - in a real system, we would use
            # NLP to extract date references from the description
            # For now, we'll use the dataset creation date
            if dataset.created_at:
                return dataset.created_at.strftime("%Y")
        
        # Default to current year
        return datetime.now().strftime("%Y")
    
    def _infer_license(self, dataset):
        """Infer or assign a license to the dataset"""
        # If the description contains license information, try to extract it
        common_licenses = {
            "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
            "cc by": "https://creativecommons.org/licenses/by/4.0/",
            "cc by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
            "cc by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
            "cc by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
            "cc by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "cc by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
            "mit": "https://opensource.org/licenses/MIT",
            "apache": "https://www.apache.org/licenses/LICENSE-2.0",
            "gpl": "https://www.gnu.org/licenses/gpl-3.0.en.html"
        }
        
        if dataset.description:
            for name, url in common_licenses.items():
                if name.lower() in dataset.description.lower():
                    return url
        
        # Default to a general non-commercial research license
        return "https://creativecommons.org/licenses/by-nc/4.0/"
    
    def _assess_fair_compliance(self, schema_org, dataset):
        """Assess FAIR (Findable, Accessible, Interoperable, Reusable) compliance"""
        # Perform assessment for each FAIR principle
        findable = self._assess_findability(schema_org, dataset)
        accessible = self._assess_accessibility(schema_org, dataset)
        interoperable = self._assess_interoperability(schema_org, dataset)
        reusable = self._assess_reusability(schema_org, dataset)
        
        return {
            'findable': findable,
            'accessible': accessible,
            'interoperable': interoperable,
            'reusable': reusable
        }
    
    def _assess_findability(self, schema_org, dataset):
        """Assess findability criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Assess metadata richness for discovery
        if schema_org.get('name'):
            score += 10
        else:
            issues.append("Missing dataset title")
        
        if schema_org.get('description') and len(schema_org['description']) > 50:
            score += 20
        else:
            issues.append("Missing or insufficient dataset description")
        
        if schema_org.get('keywords') and len(schema_org['keywords']) >= 3:
            score += 20
        else:
            issues.append("Insufficient keywords (minimum 3 recommended)")
        
        if schema_org.get('identifier'):
            score += 10
        else:
            issues.append("Missing dataset identifier")
        
        if schema_org.get('url'):
            score += 20
        else:
            issues.append("Missing dataset URL")
        
        if schema_org.get('creator') and schema_org['creator'].get('name'):
            score += 10
        else:
            issues.append("Missing dataset creator information")
        
        if schema_org.get('publisher') and schema_org['publisher'].get('name'):
            score += 10
        else:
            issues.append("Missing dataset publisher information")
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _assess_accessibility(self, schema_org, dataset):
        """Assess accessibility criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Check if dataset has a URL
        if schema_org.get('url'):
            score += 40
        else:
            issues.append("Dataset lacks a persistent URL")
        
        # Check if dataset format is specified
        if schema_org.get('encodingFormat'):
            score += 20
        else:
            issues.append("Dataset format not specified")
        
        # Check for API or download information
        if dataset.source_url:
            score += 40
        else:
            issues.append("No download link provided")
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _assess_interoperability(self, schema_org, dataset):
        """Assess interoperability criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Check for use of standard vocabularies
        if schema_org.get('@context') == 'https://schema.org/':
            score += 20
        else:
            issues.append("Metadata doesn't use standard vocabulary (Schema.org)")
        
        # Check for structured format
        if dataset.format in ['csv', 'json', 'xml']:
            score += 30
        else:
            issues.append("Dataset not in a standard structured format (CSV, JSON, XML)")
        
        # Check for variable/field descriptions
        if schema_org.get('variableMeasured') and len(schema_org['variableMeasured']) > 0:
            score += 30
        else:
            issues.append("Missing variable/field descriptions")
        
        # Check for controlled vocabularies in category
        if dataset.category:
            score += 20
        else:
            issues.append("Dataset missing category classification")
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _assess_reusability(self, schema_org, dataset):
        """Assess reusability criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Check for license information
        if schema_org.get('license'):
            score += 30
        else:
            issues.append("Missing license information")
        
        # Check for provenance information
        if schema_org.get('creator') or schema_org.get('publisher'):
            score += 20
        else:
            issues.append("Missing provenance information (creator/publisher)")
        
        # Check for temporal information
        if schema_org.get('dateCreated') or schema_org.get('dateModified'):
            score += 20
        else:
            issues.append("Missing temporal information (creation date)")
        
        # Check for detailed description
        if schema_org.get('description') and len(schema_org['description']) > 200:
            score += 30
        else:
            issues.append("Insufficient detailed description for reuse")
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _calculate_completeness(self, dataset, schema_org):
        """Calculate metadata completeness score"""
        # Count metadata fields that are populated
        total_fields = 12  # Number of important metadata fields
        populated_fields = 0
        
        # Check required fields
        if dataset.title:
            populated_fields += 1
        if dataset.description:
            populated_fields += 1
        if dataset.source:
            populated_fields += 1
        if dataset.source_url:
            populated_fields += 1
        if dataset.format:
            populated_fields += 1
        if dataset.data_type:
            populated_fields += 1
        if dataset.category:
            populated_fields += 1
        if dataset.tags:
            populated_fields += 1
        if dataset.size:
            populated_fields += 1
        
        # Check Schema.org specific fields
        if schema_org.get('license'):
            populated_fields += 1
        if schema_org.get('variableMeasured'):
            populated_fields += 1
        if schema_org.get('temporalCoverage'):
            populated_fields += 1
        
        # Calculate percentage
        return (populated_fields / total_fields) * 100
    
    def _calculate_consistency(self, processed_data):
        """Calculate data consistency score based on processed data"""
        # This is a simplified implementation
        # In a real application, we would do more detailed analysis
        
        if not processed_data:
            return 50  # Default score
        
        consistency_score = 70  # Start with a decent base score
        
        # Check for errors in processing
        if 'error' in processed_data:
            consistency_score -= 30
        
        # Check if we could determine the schema
        if processed_data.get('format') == 'csv' and 'schema' in processed_data:
            consistency_score += 15
        elif processed_data.get('format') == 'json' and 'schema' in processed_data:
            consistency_score += 15
        
        # Cap the score at 100
        return min(consistency_score, 100)
    
    def _is_schema_org_compliant(self, schema_org):
        """Check if metadata is Schema.org compliant"""
        # Check required fields
        required_fields = ['@context', '@type', 'name', 'description']
        for field in required_fields:
            if field not in schema_org:
                return False
        
        # Check that context is Schema.org
        if schema_org['@context'] != 'https://schema.org/':
            return False
        
        # Check that type is Dataset
        if schema_org['@type'] != 'Dataset':
            return False
        
        return True
    
    def _generate_recommendations(self, fair_assessment, dataset):
        """Generate recommendations to improve metadata quality"""
        recommendations = []
        
        # Add recommendations based on FAIR assessment
        for category in ['findable', 'accessible', 'interoperable', 'reusable']:
            for issue in fair_assessment[category]['issues']:
                if category == 'findable':
                    recommendations.append(f"Improve findability: {issue}")
                elif category == 'accessible':
                    recommendations.append(f"Improve accessibility: {issue}")
                elif category == 'interoperable':
                    recommendations.append(f"Improve interoperability: {issue}")
                elif category == 'reusable':
                    recommendations.append(f"Improve reusability: {issue}")
        
        # If we have too many recommendations, prioritize the most important ones
        if len(recommendations) > 5:
            # Sort by priority: findable, accessible, reusable, interoperable
            priority_order = {
                'Improve findability': 0,
                'Improve accessibility': 1,
                'Improve reusability': 2,
                'Improve interoperability': 3
            }
            
            recommendations.sort(key=lambda x: 
                                 next((priority_order[prefix] for prefix in priority_order 
                                      if x.startswith(prefix)), 99))
            
            recommendations = recommendations[:5]
        
        return recommendations


# Create a singleton instance
metadata_service = MetadataService()