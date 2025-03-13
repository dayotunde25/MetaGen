import json
from datetime import datetime
import re
from app.services.nlp_service import nlp_service

class MetadataService:
    """Service for generating, validating, and enhancing dataset metadata"""
    
    def __init__(self):
        pass
    
    def generate_metadata(self, dataset, processed_data=None):
        """Generate metadata for a dataset following Schema.org and FAIR principles"""
        
        # Create basic Schema.org metadata
        schema_org = self._generate_schema_org(dataset, processed_data)
        
        # Assess FAIR compliance
        fair_assessment = self._assess_fair_compliance(schema_org, dataset)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(fair_assessment, dataset)
        
        return {
            'schema_org': schema_org,
            'fair_assessment': fair_assessment,
            'recommendations': recommendations
        }
    
    def _generate_schema_org(self, dataset, processed_data=None):
        """Generate Schema.org compatible metadata"""
        # Base Schema.org Dataset structure
        schema_org = {
            "@context": "https://schema.org/",
            "@type": "Dataset",
            "name": dataset.title,
            "description": dataset.description or "",
            "url": dataset.source_url,
            "sameAs": dataset.source_url,
            "identifier": str(dataset.id),
            "keywords": dataset.tags_list,
            "datePublished": dataset.created_at.isoformat() if dataset.created_at else datetime.utcnow().isoformat(),
            "dateModified": dataset.updated_at.isoformat() if dataset.updated_at else datetime.utcnow().isoformat(),
            "creator": {
                "@type": "Organization",
                "name": dataset.source or "Unknown"
            },
            "includedInDataCatalog": {
                "@type": "DataCatalog",
                "name": "Metadata Generator - AI Research Dataset Catalog"
            },
            "distribution": {
                "@type": "DataDownload",
                "contentUrl": dataset.source_url,
                "encodingFormat": dataset.format,
                "contentSize": dataset.size
            }
        }
        
        # Add variable measurements if processed data is available
        if processed_data and 'schema' in processed_data:
            schema_org["variableMeasured"] = self._extract_variable_measured(processed_data)
        
        # Add spatial coverage if category suggests a geographic dataset
        if dataset.category in ['geography', 'environmental', 'demographics', 'agriculture']:
            schema_org["spatialCoverage"] = {
                "@type": "Place",
                "name": "Global" # Default, would be refined with actual data
            }
        
        # Add temporal coverage if available or can be inferred
        temporal_coverage = self._infer_temporal_coverage(dataset)
        if temporal_coverage:
            schema_org["temporalCoverage"] = temporal_coverage
        
        # Add license information
        schema_org["license"] = self._infer_license(dataset)
        
        return schema_org
    
    def _extract_variable_measured(self, processed_data):
        """Extract variable measurements from processed data"""
        variables = []
        
        if 'schema' in processed_data:
            for field in processed_data['schema']:
                variable = {
                    "@type": "PropertyValue",
                    "name": field.get('name', ''),
                    "description": f"Field of type {field.get('type', 'unknown')}"
                }
                
                # Add example if available
                if 'example' in field and field['example']:
                    variable["value"] = str(field['example'])
                    
                variables.append(variable)
        
        return variables
    
    def _infer_temporal_coverage(self, dataset):
        """Infer temporal coverage from dataset information"""
        # Try to find date patterns in title or description
        text_to_search = f"{dataset.title} {dataset.description or ''}"
        
        # Look for year patterns
        year_pattern = r'\b(19|20)\d{2}\b'  # Years from 1900-2099
        years = re.findall(year_pattern, text_to_search)
        
        if years:
            # If we find multiple years, use the min and max as a range
            if len(years) > 1:
                min_year = min(years)
                max_year = max(years)
                return f"{min_year}/{max_year}"
            else:
                # Single year
                return years[0]
        
        # Default to current year if nothing found
        return str(datetime.utcnow().year)
    
    def _infer_license(self, dataset):
        """Infer or assign a license to the dataset"""
        # Default license
        default_license = "https://creativecommons.org/licenses/by/4.0/"
        
        # Look for license information in description
        if dataset.description:
            license_patterns = {
                "CC0": r'\bCC0\b|\bCC 0\b|\bCreative\s+Commons\s+Zero\b',
                "CC-BY": r'\bCC\s*-?\s*BY\b|\bCreative\s+Commons\s+Attribution\b',
                "CC-BY-SA": r'\bCC\s*-?\s*BY\s*-?\s*SA\b|\bShare-Alike\b',
                "CC-BY-NC": r'\bCC\s*-?\s*BY\s*-?\s*NC\b|\bNon-Commercial\b',
                "MIT": r'\bMIT\s+License\b',
                "Apache": r'\bApache\s+License\b',
                "GPL": r'\bGPL\b|\bGNU\s+General\s+Public\s+License\b'
            }
            
            for license_name, pattern in license_patterns.items():
                if re.search(pattern, dataset.description, re.IGNORECASE):
                    if license_name == "CC-BY":
                        return "https://creativecommons.org/licenses/by/4.0/"
                    elif license_name == "CC-BY-SA":
                        return "https://creativecommons.org/licenses/by-sa/4.0/"
                    elif license_name == "CC-BY-NC":
                        return "https://creativecommons.org/licenses/by-nc/4.0/"
                    elif license_name == "CC0":
                        return "https://creativecommons.org/publicdomain/zero/1.0/"
                    elif license_name == "MIT":
                        return "https://opensource.org/licenses/MIT"
                    elif license_name == "Apache":
                        return "https://www.apache.org/licenses/LICENSE-2.0"
                    elif license_name == "GPL":
                        return "https://www.gnu.org/licenses/gpl-3.0.en.html"
        
        return default_license
    
    def _assess_fair_compliance(self, schema_org, dataset):
        """Assess FAIR (Findable, Accessible, Interoperable, Reusable) compliance"""
        assessment = {
            "findable": self._assess_findability(schema_org, dataset),
            "accessible": self._assess_accessibility(schema_org, dataset),
            "interoperable": self._assess_interoperability(schema_org, dataset),
            "reusable": self._assess_reusability(schema_org, dataset)
        }
        
        # Calculate overall FAIR score
        fair_scores = [
            assessment["findable"]["score"],
            assessment["accessible"]["score"],
            assessment["interoperable"]["score"],
            assessment["reusable"]["score"]
        ]
        
        assessment["overall"] = {
            "score": sum(fair_scores) / len(fair_scores),
            "is_compliant": all(score >= 70 for score in fair_scores)
        }
        
        return assessment
    
    def _assess_findability(self, schema_org, dataset):
        """Assess findability criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Check for persistent identifier
        if "identifier" in schema_org and schema_org["identifier"]:
            score += 25
        else:
            issues.append("Missing persistent identifier")
        
        # Check for rich metadata
        if "description" in schema_org and len(schema_org["description"]) > 50:
            score += 25
        else:
            issues.append("Insufficient description")
        
        # Check for keywords/tags
        if "keywords" in schema_org and schema_org["keywords"] and len(schema_org["keywords"]) >= 3:
            score += 25
        else:
            issues.append("Insufficient keywords/tags")
        
        # Check for metadata registration/discoverability
        if "includedInDataCatalog" in schema_org:
            score += 25
        else:
            issues.append("Not registered in a searchable resource")
        
        return {
            "score": score,
            "issues": issues,
            "is_compliant": score >= 70
        }
    
    def _assess_accessibility(self, schema_org, dataset):
        """Assess accessibility criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Check for open, free protocol
        if dataset.source_url and (dataset.source_url.startswith("http://") or dataset.source_url.startswith("https://")):
            score += 25
        else:
            issues.append("Not accessible via standard protocol")
        
        # Check for actual accessibility
        if dataset.status == "completed":
            score += 25
        else:
            issues.append("Dataset may not be accessible")
        
        # Check for metadata persistence
        score += 25  # Assume metadata will persist even if data becomes unavailable
        
        # Check for authentication if needed
        score += 25  # Simplified - assume no authentication needed
        
        return {
            "score": score,
            "issues": issues,
            "is_compliant": score >= 70
        }
    
    def _assess_interoperability(self, schema_org, dataset):
        """Assess interoperability criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Check for formal, accessible, shared language
        if dataset.format in ["csv", "json", "xml"]:
            score += 25
        else:
            issues.append("Not using a widely-used format")
        
        # Check for vocabularies
        if "variableMeasured" in schema_org and schema_org["variableMeasured"]:
            score += 25
        else:
            issues.append("No standardized vocabularies")
        
        # Check for qualified references
        if "url" in schema_org and "sameAs" in schema_org:
            score += 25
        else:
            issues.append("Missing references to other metadata")
        
        # Check for standard metadata format
        if schema_org.get("@context") == "https://schema.org/":
            score += 25
        else:
            issues.append("Not using a standard metadata format")
        
        return {
            "score": score,
            "issues": issues,
            "is_compliant": score >= 70
        }
    
    def _assess_reusability(self, schema_org, dataset):
        """Assess reusability criteria of FAIR principles"""
        score = 0
        issues = []
        
        # Check for rich metadata attributes
        if all(key in schema_org for key in ["description", "datePublished", "creator"]):
            score += 20
        else:
            issues.append("Missing essential metadata attributes")
        
        # Check for clear license
        if "license" in schema_org and schema_org["license"]:
            score += 20
        else:
            issues.append("No clear license")
        
        # Check for provenance/source
        if "creator" in schema_org and schema_org["creator"]:
            score += 20
        else:
            issues.append("Missing provenance information")
        
        # Check for community standards
        if dataset.format in ["csv", "json", "xml"]:
            score += 20
        else:
            issues.append("Not following community data standards")
        
        # Check for detailed attribution
        if "creator" in schema_org and isinstance(schema_org["creator"], dict) and "name" in schema_org["creator"]:
            score += 20
        else:
            issues.append("Incomplete attribution")
        
        return {
            "score": score,
            "issues": issues,
            "is_compliant": score >= 70
        }
    
    def _generate_recommendations(self, fair_assessment, dataset):
        """Generate recommendations to improve metadata quality"""
        recommendations = []
        
        # Add recommendations based on FAIR assessment issues
        for category in ["findable", "accessible", "interoperable", "reusable"]:
            for issue in fair_assessment[category]["issues"]:
                if "description" in issue.lower():
                    recommendations.append(f"Improve dataset description with more details")
                elif "keyword" in issue.lower() or "tag" in issue.lower():
                    recommendations.append(f"Add more descriptive keywords/tags")
                elif "identifier" in issue.lower():
                    recommendations.append(f"Add a persistent identifier (DOI, URI, etc.)")
                elif "protocol" in issue.lower():
                    recommendations.append(f"Ensure dataset is accessible via standard protocols")
                elif "format" in issue.lower() or "vocabulary" in issue.lower():
                    recommendations.append(f"Convert dataset to a standard format (CSV, JSON, XML)")
                elif "license" in issue.lower():
                    recommendations.append(f"Add a clear license (preferably Creative Commons)")
                elif "provenance" in issue.lower() or "attribution" in issue.lower():
                    recommendations.append(f"Add detailed information about the creator/source")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations

# Initialize singleton service
metadata_service = MetadataService()