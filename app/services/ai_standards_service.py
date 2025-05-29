"""
AI Standards Compliance Service for AIMetaHarvest.

This service ensures datasets and models comply with worldwide AI standards including:
- Model Cards (Google's standard)
- Dataset Cards (Hugging Face standard)
- AI Ethics guidelines
- Bias detection and fairness assessment
- Explainability requirements
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AIStandardsService:
    """
    Service for assessing and ensuring AI standards compliance.
    
    Implements:
    - Model Cards standard
    - Dataset Cards standard
    - AI Ethics compliance
    - Bias detection
    - Fairness assessment
    - Explainability requirements
    """
    
    def __init__(self):
        """Initialize the AI standards service."""
        self.model_card_template = self._get_model_card_template()
        self.dataset_card_template = self._get_dataset_card_template()
        self.ethics_checklist = self._get_ethics_checklist()
    
    def assess_ai_compliance(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess dataset compliance with AI standards.
        
        Args:
            dataset: Dataset object
            processed_data: Processed dataset information
            
        Returns:
            Comprehensive AI standards compliance assessment
        """
        assessment = {
            'dataset_card': self._generate_dataset_card(dataset, processed_data),
            'ethics_compliance': self._assess_ethics_compliance(dataset, processed_data),
            'bias_assessment': self._assess_bias_risks(dataset, processed_data),
            'fairness_metrics': self._calculate_fairness_metrics(dataset, processed_data),
            'explainability_score': self._assess_explainability(dataset, processed_data),
            'ai_readiness_score': 0.0,
            'recommendations': [],
            'compliance_status': 'pending'
        }
        
        # Calculate overall AI readiness score
        assessment['ai_readiness_score'] = self._calculate_ai_readiness_score(assessment)
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_ai_recommendations(assessment)
        
        # Determine compliance status
        assessment['compliance_status'] = self._determine_compliance_status(assessment)
        
        return assessment
    
    def _generate_dataset_card(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Dataset Card following Hugging Face standard."""
        card = self.dataset_card_template.copy()
        
        # Basic information
        card['dataset_name'] = dataset.title or 'Unnamed Dataset'
        card['dataset_description'] = dataset.description or ''
        card['dataset_summary'] = self._generate_dataset_summary(dataset, processed_data)
        
        # Dataset details
        card['languages'] = ['en']  # Default to English
        card['size_categories'] = self._categorize_dataset_size(processed_data)
        card['task_categories'] = self._infer_task_categories(dataset)
        
        # Data fields
        if processed_data and 'schema' in processed_data:
            card['data_fields'] = self._describe_data_fields(processed_data['schema'])
        
        # Data splits
        card['data_splits'] = self._describe_data_splits(processed_data)
        
        # Dataset creation
        card['dataset_creation'] = {
            'curation_rationale': dataset.description or 'Not specified',
            'source_data': dataset.source or 'Not specified',
            'annotations': 'Not specified',
            'personal_information': 'Unknown'
        }
        
        # Considerations
        card['considerations'] = {
            'social_impact': 'To be assessed',
            'biases': 'To be evaluated',
            'limitations': 'To be documented',
            'recommendations': 'To be provided'
        }
        
        # Additional information
        card['additional_information'] = {
            'dataset_curators': getattr(dataset.user, 'username', 'Unknown') if hasattr(dataset, 'user') else 'Unknown',
            'licensing_information': 'Not specified',
            'citation_information': self._generate_citation(dataset),
            'contributions': 'Community contributed dataset'
        }
        
        return card
    
    def _assess_ethics_compliance(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with AI ethics guidelines."""
        compliance = {
            'transparency_score': 0.0,
            'accountability_score': 0.0,
            'fairness_score': 0.0,
            'privacy_score': 0.0,
            'human_oversight_score': 0.0,
            'overall_ethics_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Transparency assessment
        transparency_score = 0.0
        if dataset.description and len(dataset.description) > 100:
            transparency_score += 30.0
        if dataset.source:
            transparency_score += 25.0
        if hasattr(dataset, 'methodology') and dataset.methodology:
            transparency_score += 25.0
        else:
            compliance['issues'].append('Missing methodology documentation')
            compliance['recommendations'].append('Add detailed methodology documentation')
        if processed_data and 'schema' in processed_data:
            transparency_score += 20.0
        
        compliance['transparency_score'] = transparency_score
        
        # Accountability assessment
        accountability_score = 0.0
        if hasattr(dataset, 'user') and dataset.user:
            accountability_score += 40.0
        if dataset.source:
            accountability_score += 30.0
        if hasattr(dataset, 'license') and dataset.license:
            accountability_score += 30.0
        else:
            compliance['issues'].append('Missing license information')
            compliance['recommendations'].append('Specify dataset license and usage terms')
        
        compliance['accountability_score'] = accountability_score
        
        # Privacy assessment
        privacy_score = 50.0  # Default neutral score
        if self._contains_personal_data(dataset, processed_data):
            privacy_score = 20.0
            compliance['issues'].append('Dataset may contain personal information')
            compliance['recommendations'].append('Review and anonymize personal information')
        elif self._has_privacy_measures(dataset):
            privacy_score = 90.0
        
        compliance['privacy_score'] = privacy_score
        
        # Calculate overall ethics score
        compliance['overall_ethics_score'] = (
            compliance['transparency_score'] * 0.3 +
            compliance['accountability_score'] * 0.3 +
            compliance['fairness_score'] * 0.2 +
            compliance['privacy_score'] * 0.2
        )
        
        return compliance
    
    def _assess_bias_risks(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential bias risks in the dataset."""
        bias_assessment = {
            'representation_bias': 0.0,
            'measurement_bias': 0.0,
            'aggregation_bias': 0.0,
            'historical_bias': 0.0,
            'overall_bias_risk': 0.0,
            'bias_indicators': [],
            'mitigation_suggestions': []
        }
        
        # Representation bias assessment
        if processed_data and 'sample_data' in processed_data:
            representation_score = self._assess_representation_bias(processed_data['sample_data'])
            bias_assessment['representation_bias'] = representation_score
        
        # Measurement bias assessment
        measurement_score = self._assess_measurement_bias(dataset, processed_data)
        bias_assessment['measurement_bias'] = measurement_score
        
        # Historical bias assessment
        historical_score = self._assess_historical_bias(dataset)
        bias_assessment['historical_bias'] = historical_score
        
        # Calculate overall bias risk (lower is better)
        bias_assessment['overall_bias_risk'] = (
            (100 - bias_assessment['representation_bias']) * 0.4 +
            (100 - bias_assessment['measurement_bias']) * 0.3 +
            (100 - bias_assessment['historical_bias']) * 0.3
        )
        
        return bias_assessment
    
    def _calculate_fairness_metrics(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fairness metrics for the dataset."""
        fairness = {
            'demographic_parity': 'unknown',
            'equalized_odds': 'unknown',
            'individual_fairness': 'unknown',
            'fairness_score': 50.0,  # Neutral default
            'protected_attributes': [],
            'fairness_recommendations': []
        }
        
        # Identify potential protected attributes
        if processed_data and 'schema' in processed_data:
            protected_attrs = self._identify_protected_attributes(processed_data['schema'])
            fairness['protected_attributes'] = protected_attrs
            
            if protected_attrs:
                fairness['fairness_recommendations'].append(
                    f"Review fairness implications of protected attributes: {', '.join(protected_attrs)}"
                )
        
        return fairness
    
    def _assess_explainability(self, dataset, processed_data: Dict[str, Any]) -> float:
        """Assess the explainability of the dataset."""
        explainability_score = 0.0
        
        # Documentation quality
        if dataset.description and len(dataset.description) > 200:
            explainability_score += 30.0
        elif dataset.description:
            explainability_score += 15.0
        
        # Schema documentation
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            if isinstance(schema, dict):
                documented_fields = sum(1 for field_info in schema.values() 
                                      if isinstance(field_info, dict) and 'description' in field_info)
                if documented_fields > 0:
                    explainability_score += 25.0
        
        # Methodology documentation
        if hasattr(dataset, 'methodology') and dataset.methodology:
            explainability_score += 25.0
        
        # Data lineage
        if dataset.source and dataset.source_url:
            explainability_score += 20.0
        
        return explainability_score
    
    def _calculate_ai_readiness_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall AI readiness score."""
        weights = {
            'ethics_compliance': 0.3,
            'bias_assessment': 0.25,
            'fairness_metrics': 0.2,
            'explainability_score': 0.25
        }
        
        ethics_score = assessment['ethics_compliance']['overall_ethics_score']
        bias_score = 100 - assessment['bias_assessment']['overall_bias_risk']  # Invert bias risk
        fairness_score = assessment['fairness_metrics']['fairness_score']
        explainability_score = assessment['explainability_score']
        
        ai_readiness = (
            ethics_score * weights['ethics_compliance'] +
            bias_score * weights['bias_assessment'] +
            fairness_score * weights['fairness_metrics'] +
            explainability_score * weights['explainability_score']
        )
        
        return round(ai_readiness, 2)
    
    def _get_model_card_template(self) -> Dict[str, Any]:
        """Get the Model Card template following Google's standard."""
        return {
            'model_details': {
                'name': '',
                'version': '',
                'date': '',
                'type': '',
                'paper': '',
                'citation': '',
                'license': '',
                'contact': ''
            },
            'intended_use': {
                'primary_uses': [],
                'primary_users': [],
                'out_of_scope': []
            },
            'factors': {
                'relevant_factors': [],
                'evaluation_factors': []
            },
            'metrics': {
                'model_performance': {},
                'decision_thresholds': {},
                'variation_approaches': []
            },
            'evaluation_data': {
                'datasets': [],
                'motivation': '',
                'preprocessing': ''
            },
            'training_data': {
                'datasets': [],
                'motivation': '',
                'preprocessing': ''
            },
            'quantitative_analyses': {
                'unitary_results': {},
                'intersectional_results': {}
            },
            'ethical_considerations': {
                'sensitive_data': '',
                'human_life': '',
                'mitigations': '',
                'risks_and_harms': '',
                'use_cases': ''
            },
            'caveats_and_recommendations': {
                'caveats': [],
                'recommendations': []
            }
        }

    def _get_dataset_card_template(self) -> Dict[str, Any]:
        """Get the Dataset Card template following Hugging Face standard."""
        return {
            'dataset_name': '',
            'dataset_description': '',
            'dataset_summary': '',
            'languages': [],
            'size_categories': '',
            'task_categories': [],
            'data_fields': {},
            'data_splits': {},
            'dataset_creation': {
                'curation_rationale': '',
                'source_data': '',
                'annotations': '',
                'personal_information': ''
            },
            'considerations': {
                'social_impact': '',
                'biases': '',
                'limitations': '',
                'recommendations': ''
            },
            'additional_information': {
                'dataset_curators': '',
                'licensing_information': '',
                'citation_information': '',
                'contributions': ''
            }
        }
    
    def _get_ethics_checklist(self) -> List[Dict[str, str]]:
        """Get AI ethics compliance checklist."""
        return [
            {'category': 'Transparency', 'requirement': 'Clear documentation of data sources'},
            {'category': 'Transparency', 'requirement': 'Methodology documentation'},
            {'category': 'Accountability', 'requirement': 'Clear ownership and responsibility'},
            {'category': 'Fairness', 'requirement': 'Bias assessment and mitigation'},
            {'category': 'Privacy', 'requirement': 'Personal data protection'},
            {'category': 'Human Oversight', 'requirement': 'Human review processes'}
        ]
    
    # Helper methods
    def _categorize_dataset_size(self, processed_data: Dict[str, Any]) -> str:
        """Categorize dataset size."""
        record_count = processed_data.get('record_count', 0)
        if record_count < 1000:
            return 'n<1K'
        elif record_count < 10000:
            return '1K<n<10K'
        elif record_count < 100000:
            return '10K<n<100K'
        elif record_count < 1000000:
            return '100K<n<1M'
        else:
            return 'n>1M'
    
    def _infer_task_categories(self, dataset) -> List[str]:
        """Infer AI task categories from dataset."""
        categories = []
        
        if dataset.category:
            category_lower = dataset.category.lower()
            if 'text' in category_lower or 'nlp' in category_lower:
                categories.append('text-classification')
            elif 'image' in category_lower or 'vision' in category_lower:
                categories.append('image-classification')
            elif 'audio' in category_lower or 'speech' in category_lower:
                categories.append('audio-classification')
            elif 'tabular' in category_lower or 'structured' in category_lower:
                categories.append('tabular-classification')
        
        return categories if categories else ['other']
    
    def _contains_personal_data(self, dataset, processed_data: Dict[str, Any]) -> bool:
        """Check if dataset contains personal information."""
        personal_indicators = ['name', 'email', 'phone', 'address', 'ssn', 'id']
        
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            if isinstance(schema, dict):
                for field_name in schema.keys():
                    if any(indicator in field_name.lower() for indicator in personal_indicators):
                        return True
        
        return False
    
    def _identify_protected_attributes(self, schema: Dict[str, Any]) -> List[str]:
        """Identify potential protected attributes in the schema."""
        protected_indicators = ['gender', 'race', 'age', 'religion', 'ethnicity', 'nationality']
        protected_attrs = []

        for field_name in schema.keys():
            if any(indicator in field_name.lower() for indicator in protected_indicators):
                protected_attrs.append(field_name)

        return protected_attrs

    def _generate_dataset_summary(self, dataset, processed_data: Dict[str, Any]) -> str:
        """Generate a summary for the dataset."""
        summary_parts = []

        if dataset.description:
            summary_parts.append(dataset.description[:200])

        if processed_data:
            record_count = processed_data.get('record_count', 0)
            if record_count > 0:
                summary_parts.append(f"Contains {record_count} records.")

            if 'schema' in processed_data:
                schema = processed_data['schema']
                if isinstance(schema, dict):
                    field_count = len(schema)
                    summary_parts.append(f"Has {field_count} fields.")

        return ' '.join(summary_parts) if summary_parts else 'Dataset summary not available.'

    def _describe_data_fields(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the data fields in the schema."""
        field_descriptions = {}

        for field_name, field_info in schema.items():
            if isinstance(field_info, dict):
                field_descriptions[field_name] = {
                    'type': field_info.get('type', 'unknown'),
                    'description': f"Field containing {field_info.get('type', 'data')} values",
                    'sample_values': field_info.get('sample_values', [])
                }
            else:
                field_descriptions[field_name] = {
                    'type': 'unknown',
                    'description': 'Data field',
                    'sample_values': []
                }

        return field_descriptions

    def _describe_data_splits(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Describe data splits (train/test/validation)."""
        # For now, return a basic structure
        # This could be enhanced based on actual data splitting logic
        return {
            'train': 'Training data split not specified',
            'test': 'Test data split not specified',
            'validation': 'Validation data split not specified'
        }

    def _generate_citation(self, dataset) -> str:
        """Generate a citation for the dataset."""
        author = getattr(dataset.user, 'username', 'Unknown') if hasattr(dataset, 'user') else 'Unknown'
        title = dataset.title or 'Untitled Dataset'
        year = dataset.created_at.year if dataset.created_at else 'n.d.'

        return f"{author}. ({year}). {title}. Retrieved from AIMetaHarvest."

    def _assess_representation_bias(self, sample_data: List[Dict]) -> float:
        """Assess representation bias in the sample data."""
        # Basic representation bias assessment
        # This is a simplified implementation
        if not sample_data or len(sample_data) < 10:
            return 50.0  # Neutral score for insufficient data

        # Check for diversity in categorical fields
        categorical_diversity = 0
        categorical_fields = 0

        for field_name in sample_data[0].keys():
            values = [row.get(field_name) for row in sample_data if row.get(field_name) is not None]
            if values and isinstance(values[0], str):
                categorical_fields += 1
                unique_values = len(set(values))
                total_values = len(values)
                diversity_ratio = unique_values / total_values if total_values > 0 else 0
                categorical_diversity += diversity_ratio

        if categorical_fields > 0:
            avg_diversity = (categorical_diversity / categorical_fields) * 100
            return min(avg_diversity * 1.5, 100.0)  # Scale up diversity score

        return 75.0  # Default score if no categorical fields

    def _assess_measurement_bias(self, dataset, processed_data: Dict[str, Any]) -> float:
        """Assess measurement bias in the dataset."""
        score = 70.0  # Default score

        # Check if measurement methodology is documented
        if hasattr(dataset, 'methodology') and dataset.methodology:
            score += 20.0

        # Check for consistent measurement units
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            if isinstance(schema, dict):
                # Look for potential measurement inconsistencies
                numeric_fields = [field for field, info in schema.items()
                                if isinstance(info, dict) and 'type' in info and
                                'int' in str(info['type']).lower() or 'float' in str(info['type']).lower()]

                if len(numeric_fields) > 0:
                    score += 10.0  # Bonus for having numeric data with potential for measurement

        return min(score, 100.0)

    def _assess_historical_bias(self, dataset) -> float:
        """Assess historical bias in the dataset."""
        score = 60.0  # Default score

        # Check if dataset acknowledges historical context
        if dataset.description:
            historical_keywords = ['historical', 'bias', 'context', 'period', 'era', 'time']
            description_lower = dataset.description.lower()

            if any(keyword in description_lower for keyword in historical_keywords):
                score += 20.0

        # Check if data source is documented
        if dataset.source:
            score += 20.0

        return min(score, 100.0)

    def _has_privacy_measures(self, dataset) -> bool:
        """Check if the dataset has privacy protection measures."""
        if dataset.description:
            privacy_keywords = ['anonymized', 'privacy', 'gdpr', 'hipaa', 'protected', 'confidential']
            description_lower = dataset.description.lower()
            return any(keyword in description_lower for keyword in privacy_keywords)

        return False

    def _generate_ai_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate AI-specific recommendations based on assessment."""
        recommendations = []

        # Ethics recommendations
        ethics_score = assessment['ethics_compliance']['overall_ethics_score']
        if ethics_score < 70:
            recommendations.append("Improve ethics compliance by adding detailed documentation and methodology")

        # Bias recommendations
        bias_risk = assessment['bias_assessment']['overall_bias_risk']
        if bias_risk > 30:
            recommendations.append("Consider bias mitigation strategies and diverse data collection")

        # Fairness recommendations
        fairness_score = assessment['fairness_metrics']['fairness_score']
        if fairness_score < 70:
            recommendations.append("Review dataset for potential fairness issues and protected attributes")

        # Explainability recommendations
        explainability_score = assessment['explainability_score']
        if explainability_score < 70:
            recommendations.append("Add more detailed documentation and methodology explanations")

        return recommendations

    def _determine_compliance_status(self, assessment: Dict[str, Any]) -> str:
        """Determine overall compliance status."""
        ai_readiness = assessment['ai_readiness_score']

        if ai_readiness >= 80:
            return 'compliant'
        elif ai_readiness >= 60:
            return 'partially_compliant'
        else:
            return 'non_compliant'


# Global service instance
ai_standards_service = AIStandardsService()
