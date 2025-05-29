"""
Intelligent Dataset Description Generator for AIMetaHarvest.

This service automatically generates comprehensive dataset descriptions by analyzing:
- Dataset content and structure
- Data types and patterns
- Statistical properties
- Potential use cases
- Domain-specific insights
"""

import re
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DatasetDescriptionGenerator:
    """
    Service for automatically generating comprehensive dataset descriptions.
    
    Features:
    - Content analysis and summarization
    - Use case identification
    - Domain detection
    - Statistical insights
    - Quality assessment integration
    """
    
    def __init__(self):
        """Initialize the description generator."""
        self.domain_keywords = self._load_domain_keywords()
        self.use_case_patterns = self._load_use_case_patterns()
        self.data_type_descriptions = self._load_data_type_descriptions()
    
    def generate_description(self, dataset, processed_data: Dict[str, Any],
                           nlp_results: Dict[str, Any] = None,
                           quality_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive structured dataset description.

        Args:
            dataset: Dataset object
            processed_data: Processed dataset information
            nlp_results: NLP analysis results
            quality_results: Quality assessment results

        Returns:
            Dictionary with structured description components
        """
        try:
            # Check if existing description is sufficient
            existing_description = dataset.description or ""
            if self._is_description_sufficient(existing_description):
                # Return structured version of existing description
                return self._structure_existing_description(
                    existing_description, dataset, processed_data, nlp_results, quality_results
                )

            # Generate new structured description
            structured_description = {
                'overview': self._generate_overview(dataset, processed_data),
                'content_analysis': self._analyze_content(dataset, processed_data, nlp_results),
                'data_structure': self._describe_data_structure(processed_data),
                'statistical_insights': self._generate_statistical_insights(processed_data),
                'domain_and_use_cases': self._identify_domain_and_use_cases(dataset, processed_data, nlp_results),
                'quality_aspects': self._describe_quality_aspects(quality_results),
                'usage_recommendations': self._generate_usage_recommendations(dataset, processed_data, nlp_results),
                'metadata': {
                    'auto_generated': True,
                    'generated_date': datetime.now().strftime('%Y-%m-%d'),
                    'version': '2.0'
                }
            }

            # Also generate a plain text version for backward compatibility
            description_parts = []
            for section_key, section_content in structured_description.items():
                if section_key != 'metadata' and section_content:
                    if isinstance(section_content, dict):
                        # Handle structured content
                        if 'text' in section_content:
                            description_parts.append(section_content['text'])
                        elif 'summary' in section_content:
                            description_parts.append(section_content['summary'])
                    elif isinstance(section_content, str):
                        description_parts.append(section_content)

            # Create plain text version
            plain_text = "\n\n".join(filter(None, description_parts))
            auto_gen_note = f"\n\n[Auto-generated description based on dataset analysis - {datetime.now().strftime('%Y-%m-%d')}]"

            structured_description['plain_text'] = plain_text + auto_gen_note

            return structured_description

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return self._generate_fallback_description(dataset, processed_data)
    
    def _is_description_sufficient(self, description: str) -> bool:
        """Check if existing description is comprehensive enough."""
        if not description or len(description.strip()) < 50:
            return False
        
        # Check for key elements
        has_content_info = any(keyword in description.lower() for keyword in 
                              ['contains', 'includes', 'data', 'information', 'records'])
        has_purpose_info = any(keyword in description.lower() for keyword in 
                              ['used for', 'purpose', 'analysis', 'research', 'study'])
        has_structure_info = any(keyword in description.lower() for keyword in 
                                ['columns', 'fields', 'variables', 'attributes'])
        
        # Consider sufficient if it has at least 2 of these elements and is reasonably long
        return (has_content_info + has_purpose_info + has_structure_info) >= 2 and len(description) > 100
    
    def _enhance_existing_description(self, existing_description: str, dataset, 
                                    processed_data: Dict[str, Any], 
                                    nlp_results: Dict[str, Any] = None,
                                    quality_results: Dict[str, Any] = None) -> str:
        """Enhance existing description with additional insights."""
        enhancements = []
        
        # Add statistical insights if missing
        if 'records' not in existing_description.lower() and processed_data:
            record_count = processed_data.get('record_count', 0)
            if record_count > 0:
                enhancements.append(f"The dataset contains {record_count:,} records.")
        
        # Add structure information if missing
        if 'columns' not in existing_description.lower() and 'fields' not in existing_description.lower():
            structure_info = self._describe_data_structure(processed_data)
            if structure_info and len(structure_info) < 200:  # Only add if concise
                enhancements.append(structure_info)
        
        # Add use case suggestions if missing
        if 'used for' not in existing_description.lower() and 'analysis' not in existing_description.lower():
            use_cases = self._generate_use_case_suggestions(dataset, processed_data, nlp_results)
            if use_cases:
                enhancements.append(f"Potential applications: {use_cases}")
        
        if enhancements:
            enhanced = existing_description + "\n\n" + " ".join(enhancements)
            enhanced += f"\n\n[Enhanced with auto-generated insights - {datetime.now().strftime('%Y-%m-%d')}]"
            return enhanced
        
        return existing_description

    def _structure_existing_description(self, existing_description: str, dataset,
                                      processed_data: Dict[str, Any],
                                      nlp_results: Dict[str, Any] = None,
                                      quality_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert existing description to structured format."""
        # Create structured version with existing description as overview
        structured = {
            'overview': {
                'text': existing_description,
                'type': 'user_provided',
                'enhanced': False
            },
            'data_structure': self._describe_data_structure(processed_data),
            'statistical_insights': self._generate_statistical_insights(processed_data),
            'metadata': {
                'auto_generated': False,
                'enhanced_date': datetime.now().strftime('%Y-%m-%d'),
                'version': '2.0'
            },
            'plain_text': existing_description
        }

        return structured

    def _generate_overview(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured dataset overview section."""
        title = dataset.title or "Untitled Dataset"

        # Basic statistics
        stats = {}
        size_category = "unknown"

        if processed_data:
            record_count = processed_data.get('record_count', 0)
            stats['record_count'] = record_count

            if record_count > 0:
                if record_count < 1000:
                    size_category = "small"
                elif record_count < 10000:
                    size_category = "medium"
                elif record_count < 100000:
                    size_category = "large"
                else:
                    size_category = "very_large"

                # Add schema information
                schema = processed_data.get('schema', {})
                if schema:
                    stats['field_count'] = len(schema)
                    stats['data_types'] = list(set(
                        field_info.get('type', 'unknown')
                        for field_info in schema.values()
                        if isinstance(field_info, dict)
                    ))

        # Generate text summary
        text_parts = []
        text_parts.append(f"This dataset, '{title}', ")

        if stats.get('record_count', 0) > 0:
            size_desc = {
                'small': 'a small dataset',
                'medium': 'a medium-sized dataset',
                'large': 'a large dataset',
                'very_large': 'a very large dataset'
            }.get(size_category, 'a dataset')

            text_parts.append(f"is {size_desc} containing {stats['record_count']:,} records")

            if stats.get('field_count'):
                text_parts.append(f" with {stats['field_count']} data fields")

        if dataset.source:
            text_parts.append(f". The data originates from {dataset.source}")

        return {
            'text': "".join(text_parts) + ".",
            'title': title,
            'size_category': size_category,
            'statistics': stats,
            'source': dataset.source or 'Unknown',
            'type': 'auto_generated'
        }
    
    def _analyze_content(self, dataset, processed_data: Dict[str, Any], 
                        nlp_results: Dict[str, Any] = None) -> str:
        """Analyze and describe dataset content."""
        content_parts = []
        
        # Use NLP results if available
        if nlp_results:
            keywords = nlp_results.get('keywords', [])
            if keywords:
                top_keywords = keywords[:5]
                content_parts.append(f"Key topics include: {', '.join(top_keywords)}")
            
            entities = nlp_results.get('entities', [])
            if entities:
                entity_types = list(set([entity.get('label', 'UNKNOWN') for entity in entities[:10]]))
                if entity_types:
                    content_parts.append(f"The dataset contains entities such as {', '.join(entity_types)}")
        
        # Analyze field names for content insights
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            field_insights = self._analyze_field_names(schema)
            if field_insights:
                content_parts.append(field_insights)
        
        if content_parts:
            return "Content Analysis: " + ". ".join(content_parts) + "."
        
        return ""
    
    def _describe_data_structure(self, processed_data: Dict[str, Any]) -> str:
        """Describe the structure and types of data."""
        if not processed_data or 'schema' not in processed_data:
            return ""
        
        schema = processed_data['schema']
        if not schema:
            return ""
        
        # Analyze data types
        type_counts = {}
        field_examples = []
        
        for field_name, field_info in schema.items():
            if isinstance(field_info, dict):
                field_type = field_info.get('type', 'unknown')
                type_counts[field_type] = type_counts.get(field_type, 0) + 1
                
                # Add interesting field examples
                if len(field_examples) < 5:
                    sample_values = field_info.get('sample_values', [])
                    if sample_values:
                        field_examples.append(f"{field_name} ({field_type})")
        
        structure_parts = []
        
        # Describe data types
        if type_counts:
            type_descriptions = []
            for data_type, count in type_counts.items():
                type_desc = self.data_type_descriptions.get(data_type, data_type)
                type_descriptions.append(f"{count} {type_desc} field{'s' if count > 1 else ''}")
            
            structure_parts.append(f"Data Structure: The dataset includes {', '.join(type_descriptions)}")
        
        # Add field examples
        if field_examples:
            structure_parts.append(f"Key fields include: {', '.join(field_examples)}")
        
        return ". ".join(structure_parts) + "." if structure_parts else ""
    
    def _generate_statistical_insights(self, processed_data: Dict[str, Any]) -> str:
        """Generate statistical insights about the dataset."""
        if not processed_data:
            return ""
        
        insights = []
        
        # Record count insights
        record_count = processed_data.get('record_count', 0)
        if record_count > 0:
            if record_count >= 1000000:
                insights.append("This is a large-scale dataset suitable for big data analysis")
            elif record_count >= 10000:
                insights.append("The dataset size is appropriate for statistical analysis and machine learning")
            elif record_count >= 1000:
                insights.append("The dataset provides a good sample size for analysis")
        
        # Data quality insights
        if 'cleaning_stats' in processed_data:
            cleaning_stats = processed_data['cleaning_stats']
            
            duplicates = cleaning_stats.get('duplicates_removed', 0)
            if duplicates > 0:
                insights.append(f"Data cleaning removed {duplicates} duplicate records")
            
            missing_values = cleaning_stats.get('missing_values', {})
            if missing_values:
                total_missing = sum(info.get('count', 0) for info in missing_values.values())
                if total_missing > 0:
                    insights.append(f"Missing value handling was applied to improve data completeness")
        
        # Schema insights
        schema = processed_data.get('schema', {})
        if schema:
            numeric_fields = sum(1 for field_info in schema.values() 
                               if isinstance(field_info, dict) and 
                               any(num_type in str(field_info.get('type', '')).lower() 
                                   for num_type in ['int', 'float', 'number']))
            
            if numeric_fields > 0:
                insights.append(f"Contains {numeric_fields} numeric fields suitable for quantitative analysis")
        
        if insights:
            return "Statistical Overview: " + ". ".join(insights) + "."
        
        return ""
    
    def _identify_domain_and_use_cases(self, dataset, processed_data: Dict[str, Any], 
                                     nlp_results: Dict[str, Any] = None) -> str:
        """Identify domain and potential use cases."""
        # Detect domain
        domain = self._detect_domain(dataset, processed_data, nlp_results)
        
        # Generate use cases
        use_cases = self._generate_use_case_suggestions(dataset, processed_data, nlp_results)
        
        parts = []
        if domain:
            parts.append(f"Domain: This appears to be a {domain} dataset")
        
        if use_cases:
            parts.append(f"Potential Use Cases: {use_cases}")
        
        return ". ".join(parts) + "." if parts else ""
    
    def _detect_domain(self, dataset, processed_data: Dict[str, Any], 
                      nlp_results: Dict[str, Any] = None) -> str:
        """Detect the domain/field of the dataset."""
        text_content = []
        
        # Collect text for analysis
        if dataset.title:
            text_content.append(dataset.title.lower())
        if dataset.description:
            text_content.append(dataset.description.lower())
        if dataset.category:
            text_content.append(dataset.category.lower())
        
        # Add field names
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            field_names = [name.lower() for name in schema.keys()]
            text_content.extend(field_names)
        
        # Add NLP keywords
        if nlp_results and 'keywords' in nlp_results:
            keywords = [kw.lower() for kw in nlp_results['keywords'][:10]]
            text_content.extend(keywords)
        
        full_text = " ".join(text_content)
        
        # Check against domain keywords
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] >= 2:  # Require at least 2 keyword matches
                return best_domain
        
        return ""
    
    def _generate_use_case_suggestions(self, dataset, processed_data: Dict[str, Any], 
                                     nlp_results: Dict[str, Any] = None) -> str:
        """Generate potential use case suggestions."""
        use_cases = []
        
        # Analyze data structure for use cases
        if processed_data and 'schema' in processed_data:
            schema = processed_data['schema']
            
            # Check for time series data
            time_fields = [name for name in schema.keys() 
                          if any(time_word in name.lower() for time_word in ['date', 'time', 'timestamp', 'year', 'month'])]
            if time_fields:
                use_cases.append("time series analysis")
                use_cases.append("trend analysis")
            
            # Check for geographic data
            geo_fields = [name for name in schema.keys() 
                         if any(geo_word in name.lower() for geo_word in ['lat', 'lon', 'location', 'address', 'city', 'country', 'zip'])]
            if geo_fields:
                use_cases.append("geographic analysis")
                use_cases.append("spatial data visualization")
            
            # Check for categorical data
            categorical_fields = [name for name in schema.keys() 
                                if any(cat_word in name.lower() for cat_word in ['category', 'type', 'class', 'group', 'status'])]
            if categorical_fields:
                use_cases.append("classification analysis")
                use_cases.append("categorical data mining")
            
            # Check for numeric data
            numeric_fields = [name for name, info in schema.items() 
                            if isinstance(info, dict) and 
                            any(num_type in str(info.get('type', '')).lower() for num_type in ['int', 'float', 'number'])]
            if len(numeric_fields) >= 3:
                use_cases.append("statistical modeling")
                use_cases.append("machine learning")
                use_cases.append("predictive analytics")
        
        # Add domain-specific use cases
        domain = self._detect_domain(dataset, processed_data, nlp_results)
        if domain in self.use_case_patterns:
            use_cases.extend(self.use_case_patterns[domain])
        
        # Remove duplicates and limit
        unique_use_cases = list(dict.fromkeys(use_cases))[:5]
        
        return ", ".join(unique_use_cases) if unique_use_cases else ""
    
    def _describe_quality_aspects(self, quality_results: Dict[str, Any] = None) -> str:
        """Describe quality and compliance aspects."""
        if not quality_results:
            return ""
        
        quality_parts = []
        
        # Overall quality score
        quality_score = quality_results.get('quality_score', 0)
        if quality_score > 0:
            if quality_score >= 80:
                quality_desc = "high-quality"
            elif quality_score >= 60:
                quality_desc = "good-quality"
            else:
                quality_desc = "moderate-quality"
            
            quality_parts.append(f"This is a {quality_desc} dataset (quality score: {quality_score:.1f}/100)")
        
        # FAIR compliance
        fair_compliant = quality_results.get('fair_compliant', False)
        if fair_compliant:
            quality_parts.append("The dataset meets FAIR (Findable, Accessible, Interoperable, Reusable) principles")
        
        # Schema.org compliance
        schema_compliant = quality_results.get('schema_org_compliant', False)
        if schema_compliant:
            quality_parts.append("Metadata follows Schema.org standards for improved discoverability")
        
        if quality_parts:
            return "Quality Assessment: " + ". ".join(quality_parts) + "."
        
        return ""
    
    def _generate_usage_recommendations(self, dataset, processed_data: Dict[str, Any], 
                                      nlp_results: Dict[str, Any] = None) -> str:
        """Generate usage recommendations and best practices."""
        recommendations = []
        
        # Size-based recommendations
        if processed_data:
            record_count = processed_data.get('record_count', 0)
            
            if record_count > 100000:
                recommendations.append("suitable for big data analytics and distributed computing frameworks")
            elif record_count > 10000:
                recommendations.append("appropriate for machine learning model training and validation")
            elif record_count > 1000:
                recommendations.append("good for statistical analysis and exploratory data analysis")
            
            # Data type recommendations
            schema = processed_data.get('schema', {})
            if schema:
                numeric_ratio = sum(1 for info in schema.values() 
                                  if isinstance(info, dict) and 
                                  any(num_type in str(info.get('type', '')).lower() for num_type in ['int', 'float', 'number'])) / len(schema)
                
                if numeric_ratio > 0.7:
                    recommendations.append("well-suited for quantitative analysis and mathematical modeling")
                elif numeric_ratio < 0.3:
                    recommendations.append("ideal for text analysis and categorical data exploration")
        
        # Add licensing and usage notes
        if dataset.source:
            recommendations.append("please verify licensing and usage rights before commercial use")
        
        if recommendations:
            return "Usage Recommendations: " + ", ".join(recommendations) + "."
        
        return ""
    
    def _analyze_field_names(self, schema: Dict[str, Any]) -> str:
        """Analyze field names to understand content."""
        if not schema:
            return ""
        
        field_categories = {
            'personal': ['name', 'email', 'phone', 'address', 'age', 'gender'],
            'financial': ['price', 'cost', 'revenue', 'profit', 'salary', 'income', 'amount'],
            'temporal': ['date', 'time', 'timestamp', 'year', 'month', 'day'],
            'geographic': ['location', 'city', 'country', 'state', 'zip', 'latitude', 'longitude'],
            'performance': ['score', 'rating', 'rank', 'performance', 'metric', 'kpi'],
            'categorical': ['category', 'type', 'class', 'group', 'status', 'level']
        }
        
        detected_categories = []
        field_names_lower = [name.lower() for name in schema.keys()]
        
        for category, keywords in field_categories.items():
            if any(keyword in field_name for field_name in field_names_lower for keyword in keywords):
                detected_categories.append(category)
        
        if detected_categories:
            return f"The dataset includes {', '.join(detected_categories)} information"
        
        return ""
    
    def _generate_fallback_description(self, dataset, processed_data: Dict[str, Any]) -> str:
        """Generate a basic fallback description."""
        parts = []
        
        title = dataset.title or "Dataset"
        parts.append(f"This dataset, '{title}', contains structured data")
        
        if processed_data:
            record_count = processed_data.get('record_count', 0)
            if record_count > 0:
                parts.append(f" with {record_count:,} records")
            
            schema = processed_data.get('schema', {})
            if schema:
                parts.append(f" and {len(schema)} data fields")
        
        parts.append(". The dataset is suitable for data analysis and research purposes.")
        
        return "".join(parts)
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords for classification."""
        return {
            'healthcare': ['health', 'medical', 'patient', 'hospital', 'disease', 'treatment', 'clinical', 'diagnosis', 'medicine', 'therapy'],
            'finance': ['financial', 'bank', 'money', 'investment', 'stock', 'market', 'trading', 'revenue', 'profit', 'economic'],
            'education': ['education', 'student', 'school', 'university', 'course', 'grade', 'learning', 'academic', 'teacher', 'curriculum'],
            'retail': ['sales', 'product', 'customer', 'purchase', 'order', 'inventory', 'retail', 'commerce', 'shopping', 'transaction'],
            'transportation': ['transport', 'vehicle', 'traffic', 'route', 'travel', 'logistics', 'shipping', 'delivery', 'mobility', 'transit'],
            'technology': ['software', 'hardware', 'computer', 'system', 'network', 'data', 'digital', 'tech', 'programming', 'algorithm'],
            'social media': ['social', 'media', 'post', 'comment', 'like', 'share', 'follower', 'tweet', 'facebook', 'instagram'],
            'environmental': ['environment', 'climate', 'weather', 'pollution', 'energy', 'sustainability', 'carbon', 'emission', 'green', 'renewable'],
            'sports': ['sport', 'game', 'player', 'team', 'score', 'match', 'competition', 'athlete', 'performance', 'tournament'],
            'government': ['government', 'public', 'policy', 'citizen', 'municipal', 'federal', 'state', 'administration', 'civic', 'political']
        }
    
    def _load_use_case_patterns(self) -> Dict[str, List[str]]:
        """Load use case patterns for different domains."""
        return {
            'healthcare': ['medical research', 'patient outcome analysis', 'epidemiological studies', 'clinical decision support'],
            'finance': ['risk assessment', 'fraud detection', 'investment analysis', 'credit scoring', 'market prediction'],
            'education': ['student performance analysis', 'curriculum optimization', 'learning analytics', 'educational research'],
            'retail': ['customer segmentation', 'demand forecasting', 'recommendation systems', 'inventory optimization'],
            'transportation': ['route optimization', 'traffic analysis', 'logistics planning', 'mobility studies'],
            'technology': ['system monitoring', 'performance optimization', 'user behavior analysis', 'software metrics'],
            'social media': ['sentiment analysis', 'social network analysis', 'content recommendation', 'user engagement studies'],
            'environmental': ['climate modeling', 'environmental monitoring', 'sustainability assessment', 'pollution tracking'],
            'sports': ['performance analytics', 'player evaluation', 'game strategy analysis', 'sports betting'],
            'government': ['policy analysis', 'public service optimization', 'citizen engagement studies', 'administrative efficiency']
        }
    
    def _load_data_type_descriptions(self) -> Dict[str, str]:
        """Load descriptions for different data types."""
        return {
            'int64': 'numeric integer',
            'float64': 'numeric decimal',
            'object': 'text/categorical',
            'bool': 'boolean',
            'datetime64[ns]': 'date/time',
            'category': 'categorical',
            'string': 'text',
            'number': 'numeric'
        }


# Global service instance
description_generator = DatasetDescriptionGenerator()
