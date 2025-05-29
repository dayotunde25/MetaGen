"""
Metadata Service for generating comprehensive dataset metadata.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from app.services.nlp_service import nlp_service


class MetadataService:
    """Service for generating and managing dataset metadata"""
    
    def __init__(self):
        self.schema_org_template = {
            "@context": "https://schema.org",
            "@type": "Dataset",
            "name": "",
            "description": "",
            "url": "",
            "creator": {
                "@type": "Person",
                "name": ""
            },
            "dateCreated": "",
            "dateModified": "",
            "keywords": [],
            "license": "",
            "distribution": {
                "@type": "DataDownload",
                "encodingFormat": "",
                "contentSize": ""
            }
        }
    
    def generate_metadata(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata for a dataset"""
        metadata = {
            'basic_info': self._generate_basic_info(dataset),
            'content_analysis': self._analyze_content(dataset, processed_data),
            'quality_metrics': self._calculate_quality_metrics(dataset, processed_data),
            'fair_assessment': self._assess_fair_compliance(dataset),
            'schema_org': self._generate_schema_org(dataset),
            'recommendations': self._generate_recommendations(dataset, processed_data),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return metadata
    
    def _generate_basic_info(self, dataset) -> Dict[str, Any]:
        """Generate basic dataset information"""
        return {
            'title': dataset.title,
            'description': dataset.description,
            'source': dataset.source,
            'source_url': dataset.source_url,
            'data_type': dataset.data_type,
            'category': dataset.category,
            'format': dataset.format,
            'size': dataset.size,
            'record_count': getattr(dataset, 'record_count', None),
            'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
            'updated_at': dataset.updated_at.isoformat() if dataset.updated_at else None
        }
    
    def _analyze_content(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset content using NLP"""
        content_text = self._extract_content_text(dataset, processed_data)
        
        analysis = {
            'keywords': [],
            'suggested_tags': [],
            'entities': [],
            'sentiment': {},
            'summary': '',
            'language': 'en'  # Default to English
        }
        
        if content_text:
            analysis.update({
                'keywords': nlp_service.extract_keywords(content_text, 15),
                'suggested_tags': nlp_service.suggest_tags(content_text, 8),
                'entities': nlp_service.extract_entities(content_text),
                'sentiment': nlp_service.analyze_sentiment(content_text),
                'summary': nlp_service.generate_summary(content_text, 2)
            })
        
        return analysis
    
    def _extract_content_text(self, dataset, processed_data: Dict[str, Any]) -> str:
        """Extract text content for NLP analysis"""
        text_parts = []
        
        # Add dataset metadata
        if dataset.title:
            text_parts.append(dataset.title)
        if dataset.description:
            text_parts.append(dataset.description)
        if dataset.tags:
            text_parts.extend(dataset.tags if isinstance(dataset.tags, list) else [dataset.tags])
        
        # Add processed data content
        if processed_data and 'sample_data' in processed_data:
            sample_data = processed_data['sample_data']
            if isinstance(sample_data, list):
                for row in sample_data[:5]:  # First 5 rows
                    if isinstance(row, dict):
                        text_parts.extend([str(v) for v in row.values() if isinstance(v, str)])
                    elif isinstance(row, list):
                        text_parts.extend([str(v) for v in row if isinstance(v, str)])
        
        return ' '.join(text_parts)
    
    def _calculate_quality_metrics(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset"""
        metrics = {
            'completeness': 0,
            'consistency': 0,
            'accuracy': 0,
            'timeliness': 0,
            'validity': 0,
            'overall_score': 0
        }
        
        # Completeness (based on metadata fields)
        completeness_score = 0
        total_fields = 8
        
        if dataset.title: completeness_score += 1
        if dataset.description: completeness_score += 1
        if dataset.source: completeness_score += 1
        if dataset.data_type: completeness_score += 1
        if dataset.category: completeness_score += 1
        if dataset.tags: completeness_score += 1
        if dataset.format: completeness_score += 1
        if getattr(dataset, 'record_count', None): completeness_score += 1
        
        metrics['completeness'] = round((completeness_score / total_fields) * 100, 2)
        
        # Consistency (based on data structure if available)
        if processed_data and 'columns' in processed_data:
            consistency_score = 85  # Base score
            columns = processed_data['columns']
            
            # Check for consistent column types
            if len(columns) > 0:
                consistency_score += 10
            
            metrics['consistency'] = min(consistency_score, 100)
        else:
            metrics['consistency'] = 70  # Default for non-tabular data
        
        # Accuracy (basic checks)
        accuracy_score = 80  # Base score
        if dataset.source_url and dataset.source_url.startswith('http'):
            accuracy_score += 10
        if dataset.description and len(dataset.description) > 50:
            accuracy_score += 10
        
        metrics['accuracy'] = min(accuracy_score, 100)
        
        # Timeliness (based on creation date)
        if dataset.created_at:
            days_old = (datetime.utcnow() - dataset.created_at).days
            if days_old < 30:
                metrics['timeliness'] = 100
            elif days_old < 90:
                metrics['timeliness'] = 80
            elif days_old < 365:
                metrics['timeliness'] = 60
            else:
                metrics['timeliness'] = 40
        else:
            metrics['timeliness'] = 50
        
        # Validity (based on format and structure)
        validity_score = 70  # Base score
        if dataset.format in ['csv', 'json', 'xml']:
            validity_score += 20
        if processed_data and 'record_count' in processed_data:
            validity_score += 10
        
        metrics['validity'] = min(validity_score, 100)
        
        # Overall score (weighted average)
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.20,
            'timeliness': 0.15,
            'validity': 0.20
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
        metrics['overall_score'] = round(overall_score, 2)
        
        return metrics
    
    def _assess_fair_compliance(self, dataset) -> Dict[str, Any]:
        """Assess FAIR (Findable, Accessible, Interoperable, Reusable) compliance"""
        fair_scores = {
            'findable': 0,
            'accessible': 0,
            'interoperable': 0,
            'reusable': 0,
            'overall': 0
        }
        
        # Findable
        findable_score = 0
        if dataset.title: findable_score += 25
        if dataset.description: findable_score += 25
        if dataset.tags: findable_score += 25
        if dataset.category: findable_score += 25
        fair_scores['findable'] = findable_score
        
        # Accessible
        accessible_score = 50  # Base score for being in the system
        if dataset.source_url: accessible_score += 30
        if dataset.format: accessible_score += 20
        fair_scores['accessible'] = min(accessible_score, 100)
        
        # Interoperable
        interoperable_score = 40  # Base score
        if dataset.format in ['csv', 'json', 'xml']: interoperable_score += 40
        if dataset.data_type: interoperable_score += 20
        fair_scores['interoperable'] = min(interoperable_score, 100)
        
        # Reusable
        reusable_score = 30  # Base score
        if dataset.description and len(dataset.description) > 100: reusable_score += 30
        if dataset.source: reusable_score += 20
        if dataset.tags: reusable_score += 20
        fair_scores['reusable'] = min(reusable_score, 100)
        
        # Overall FAIR score
        fair_scores['overall'] = round(sum(fair_scores.values()) / 4, 2)
        
        return {
            'scores': fair_scores,
            'compliant': fair_scores['overall'] >= 70,
            'recommendations': self._generate_fair_recommendations(fair_scores)
        }
    
    def _generate_fair_recommendations(self, fair_scores: Dict[str, int]) -> List[str]:
        """Generate recommendations for improving FAIR compliance"""
        recommendations = []
        
        if fair_scores['findable'] < 80:
            recommendations.append("Improve findability by adding more descriptive tags and keywords")
        
        if fair_scores['accessible'] < 80:
            recommendations.append("Enhance accessibility by providing direct download links or API access")
        
        if fair_scores['interoperable'] < 80:
            recommendations.append("Use standard data formats (CSV, JSON) for better interoperability")
        
        if fair_scores['reusable'] < 80:
            recommendations.append("Add detailed documentation and usage examples for better reusability")
        
        return recommendations
    
    def _generate_schema_org(self, dataset) -> Dict[str, Any]:
        """Generate Schema.org compliant metadata"""
        schema_org = self.schema_org_template.copy()
        
        schema_org.update({
            "name": dataset.title or "",
            "description": dataset.description or "",
            "url": dataset.source_url or "",
            "dateCreated": dataset.created_at.isoformat() if dataset.created_at else "",
            "dateModified": dataset.updated_at.isoformat() if dataset.updated_at else "",
            "keywords": dataset.tags if isinstance(dataset.tags, list) else [dataset.tags] if dataset.tags else []
        })
        
        if hasattr(dataset, 'user') and dataset.user:
            schema_org["creator"]["name"] = getattr(dataset.user, 'username', 'Unknown')
        
        if dataset.format:
            schema_org["distribution"]["encodingFormat"] = dataset.format
        
        if dataset.size:
            schema_org["distribution"]["contentSize"] = dataset.size
        
        return schema_org
    
    def _generate_recommendations(self, dataset, processed_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving dataset quality"""
        recommendations = []
        
        if not dataset.description or len(dataset.description) < 50:
            recommendations.append("Add a more detailed description (at least 50 characters)")
        
        if not dataset.tags:
            recommendations.append("Add relevant tags to improve discoverability")
        
        if not dataset.category:
            recommendations.append("Specify a category for better organization")
        
        if not dataset.data_type:
            recommendations.append("Specify the data type for better classification")
        
        if not dataset.source:
            recommendations.append("Provide information about the data source")
        
        if processed_data and processed_data.get('record_count', 0) < 10:
            recommendations.append("Consider providing more data samples for better analysis")
        
        return recommendations


# Global service instance
metadata_service = MetadataService()
