"""
Data Visualization Service for AIMetaHarvest.

This service generates comprehensive visualizations from real dataset content including:
- Statistical charts and graphs
- Data distribution plots
- Quality assessment visualizations
- Interactive data exploration charts
- Export capabilities for web display
"""

import json
import statistics
import base64
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Optional imports for advanced visualizations
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    VISUALIZATION_LIBS_AVAILABLE = True
except ImportError:
    VISUALIZATION_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataVisualizationService:
    """
    Service for generating comprehensive data visualizations from real dataset content.
    
    Features:
    - Statistical distribution charts
    - Data quality visualizations
    - Interactive web-ready charts
    - Export to multiple formats
    - Real-time data analysis
    """
    
    def __init__(self):
        """Initialize the visualization service."""
        self.chart_colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f'
        ]
        self.quality_colors = {
            'excellent': '#27ae60',
            'good': '#2ecc71', 
            'fair': '#f39c12',
            'poor': '#e74c3c',
            'critical': '#c0392b'
        }
    
    def generate_comprehensive_visualizations(self, dataset, processed_data: Dict[str, Any], 
                                            quality_results: Dict[str, Any] = None,
                                            ai_compliance: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations for a dataset.
        
        Args:
            dataset: Dataset object
            processed_data: Processed dataset information
            quality_results: Quality assessment results
            ai_compliance: AI compliance assessment results
            
        Returns:
            Dictionary containing all generated visualizations
        """
        try:
            visualizations = {
                'generated_at': datetime.utcnow().isoformat(),
                'dataset_id': str(dataset.id),
                'charts': {},
                'summary': {},
                'export_formats': ['json', 'html']
            }
            
            # 1. Data Overview Visualizations
            if processed_data:
                overview_charts = self._generate_data_overview_charts(processed_data)
                visualizations['charts'].update(overview_charts)
            
            # 2. Quality Assessment Visualizations
            if quality_results:
                quality_charts = self._generate_quality_visualizations(quality_results)
                visualizations['charts'].update(quality_charts)
            
            # 3. Statistical Analysis Charts
            if processed_data and 'sample_data' in processed_data:
                stats_charts = self._generate_statistical_charts(processed_data)
                visualizations['charts'].update(stats_charts)
            
            # 4. AI Compliance Visualizations
            if ai_compliance:
                ai_charts = self._generate_ai_compliance_charts(ai_compliance)
                visualizations['charts'].update(ai_charts)
            
            # 5. Data Distribution Analysis
            if processed_data and 'schema' in processed_data:
                distribution_charts = self._generate_distribution_charts(processed_data)
                visualizations['charts'].update(distribution_charts)
            
            # 6. Generate summary statistics
            visualizations['summary'] = self._generate_visualization_summary(visualizations['charts'])
            
            # 7. Add advanced visualizations if libraries available
            if VISUALIZATION_LIBS_AVAILABLE and processed_data:
                advanced_charts = self._generate_advanced_visualizations(processed_data)
                visualizations['charts'].update(advanced_charts)
                visualizations['export_formats'].extend(['png', 'svg'])
            
            logger.info(f"Generated {len(visualizations['charts'])} visualizations for dataset {dataset.id}")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return self._generate_fallback_visualizations(dataset, processed_data)
    
    def _generate_data_overview_charts(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data overview charts."""
        charts = {}
        
        # Record count visualization
        record_count = processed_data.get('record_count', 0)

        # Ensure record_count is an integer
        if not isinstance(record_count, (int, float)):
            try:
                record_count = int(record_count) if record_count else 0
            except (ValueError, TypeError):
                record_count = 0

        if record_count > 0:
            charts['record_count'] = {
                'type': 'metric',
                'title': 'Dataset Size',
                'data': {
                    'value': record_count,
                    'label': 'Total Records',
                    'format': 'number'
                }
            }
        
        # Schema overview
        schema = processed_data.get('schema', {})
        if schema:
            # Data types distribution
            type_counts = {}
            for field_info in schema.values():
                if isinstance(field_info, dict):
                    field_type = field_info.get('type', 'unknown')
                    type_counts[field_type] = type_counts.get(field_type, 0) + 1
            
            if type_counts:
                charts['data_types'] = {
                    'type': 'pie',
                    'title': 'Data Types Distribution',
                    'data': [
                        {'name': data_type, 'value': count, 'color': self.chart_colors[i % len(self.chart_colors)]}
                        for i, (data_type, count) in enumerate(type_counts.items())
                    ]
                }
            
            # Field count
            charts['field_count'] = {
                'type': 'metric',
                'title': 'Schema Complexity',
                'data': {
                    'value': len(schema),
                    'label': 'Total Fields',
                    'format': 'number'
                }
            }
        
        return charts
    
    def _generate_quality_visualizations(self, quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality assessment visualizations."""
        charts = {}
        
        # Overall quality score
        quality_score = quality_results.get('quality_score', 0)
        charts['quality_score'] = {
            'type': 'gauge',
            'title': 'Overall Quality Score',
            'data': {
                'value': quality_score,
                'max': 100,
                'color': self._get_quality_color(quality_score),
                'label': f'{quality_score:.1f}/100'
            }
        }
        
        # Quality dimensions
        dimensions = ['completeness', 'consistency', 'accuracy', 'timeliness', 'conformity', 'integrity']
        dimension_data = []
        
        for dimension in dimensions:
            score = quality_results.get(dimension, 0)
            dimension_data.append({
                'dimension': dimension.capitalize(),
                'score': score,
                'color': self._get_quality_color(score)
            })
        
        if dimension_data:
            charts['quality_dimensions'] = {
                'type': 'bar',
                'title': 'Quality Dimensions',
                'data': dimension_data,
                'xAxis': 'dimension',
                'yAxis': 'score',
                'max': 100
            }
        
        # FAIR compliance
        fair_scores = quality_results.get('fair_scores', {})
        if fair_scores:
            fair_data = []
            fair_principles = ['findable', 'accessible', 'interoperable', 'reusable']
            
            for principle in fair_principles:
                score = fair_scores.get(principle, 0)
                fair_data.append({
                    'principle': principle.capitalize(),
                    'score': score,
                    'color': self._get_quality_color(score)
                })
            
            charts['fair_compliance'] = {
                'type': 'radar',
                'title': 'FAIR Principles Compliance',
                'data': fair_data,
                'max': 100
            }
        
        return charts
    
    def _generate_statistical_charts(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis charts from sample data."""
        charts = {}
        
        sample_data = processed_data.get('sample_data', [])
        schema = processed_data.get('schema', {})
        
        if not sample_data or not schema:
            return charts
        
        # Convert to DataFrame-like structure for analysis
        try:
            # Analyze numeric fields
            numeric_fields = []
            for field_name, field_info in schema.items():
                if isinstance(field_info, dict):
                    field_type = field_info.get('type', '')
                    if any(num_type in str(field_type).lower() for num_type in ['int', 'float', 'number']):
                        numeric_fields.append(field_name)
            
            if numeric_fields:
                # Generate statistics for numeric fields
                numeric_stats = {}
                for field in numeric_fields[:5]:  # Limit to first 5 numeric fields
                    values = []
                    for row in sample_data:
                        if isinstance(row, dict) and field in row:
                            try:
                                value = float(row[field])
                                if not (value != value):  # Check for NaN
                                    values.append(value)
                            except (ValueError, TypeError):
                                continue
                    
                    if len(values) >= 3:  # Need at least 3 values for statistics
                        numeric_stats[field] = {
                            'mean': statistics.mean(values),
                            'median': statistics.median(values),
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
                        
                        if len(values) > 1:
                            numeric_stats[field]['std'] = statistics.stdev(values)
                
                if numeric_stats:
                    # Create summary statistics chart
                    stats_data = []
                    for field, stats in numeric_stats.items():
                        stats_data.append({
                            'field': field,
                            'mean': round(stats['mean'], 2),
                            'median': round(stats['median'], 2),
                            'min': round(stats['min'], 2),
                            'max': round(stats['max'], 2),
                            'count': stats['count']
                        })
                    
                    charts['numeric_statistics'] = {
                        'type': 'table',
                        'title': 'Numeric Field Statistics',
                        'data': stats_data,
                        'columns': ['field', 'mean', 'median', 'min', 'max', 'count']
                    }
            
            # Analyze categorical fields
            categorical_fields = []
            for field_name, field_info in schema.items():
                if isinstance(field_info, dict):
                    field_type = field_info.get('type', '')
                    if 'str' in str(field_type).lower() or 'object' in str(field_type).lower():
                        categorical_fields.append(field_name)
            
            if categorical_fields:
                # Generate value counts for categorical fields
                for field in categorical_fields[:3]:  # Limit to first 3 categorical fields
                    value_counts = {}
                    for row in sample_data:
                        if isinstance(row, dict) and field in row:
                            value = str(row[field])
                            if value and value.lower() not in ['nan', 'null', 'none', '']:
                                value_counts[value] = value_counts.get(value, 0) + 1
                    
                    if value_counts:
                        # Sort by count and take top 10
                        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        chart_data = []
                        for i, (value, count) in enumerate(sorted_values):
                            chart_data.append({
                                'category': value[:20] + ('...' if len(value) > 20 else ''),
                                'count': count,
                                'color': self.chart_colors[i % len(self.chart_colors)]
                            })
                        
                        charts[f'categorical_{field}'] = {
                            'type': 'bar',
                            'title': f'Top Values in {field}',
                            'data': chart_data,
                            'xAxis': 'category',
                            'yAxis': 'count'
                        }
        
        except Exception as e:
            logger.warning(f"Error generating statistical charts: {e}")
        
        return charts
    
    def _generate_ai_compliance_charts(self, ai_compliance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI compliance visualizations."""
        charts = {}
        
        # AI readiness score
        ai_readiness = ai_compliance.get('ai_readiness_score', 0)
        charts['ai_readiness'] = {
            'type': 'gauge',
            'title': 'AI Readiness Score',
            'data': {
                'value': ai_readiness,
                'max': 100,
                'color': self._get_quality_color(ai_readiness),
                'label': f'{ai_readiness:.1f}/100'
            }
        }
        
        # Ethics compliance breakdown
        ethics_compliance = ai_compliance.get('ethics_compliance', {})
        if ethics_compliance:
            ethics_data = []
            ethics_metrics = ['transparency_score', 'accountability_score', 'fairness_score', 'privacy_score']
            
            for metric in ethics_metrics:
                score = ethics_compliance.get(metric, 0)
                ethics_data.append({
                    'metric': metric.replace('_score', '').capitalize(),
                    'score': score,
                    'color': self._get_quality_color(score)
                })
            
            charts['ethics_compliance'] = {
                'type': 'radar',
                'title': 'AI Ethics Compliance',
                'data': ethics_data,
                'max': 100
            }
        
        # Bias assessment
        bias_assessment = ai_compliance.get('bias_assessment', {})
        if bias_assessment:
            bias_risk = bias_assessment.get('overall_bias_risk', 0)
            charts['bias_risk'] = {
                'type': 'gauge',
                'title': 'Bias Risk Assessment',
                'data': {
                    'value': 100 - bias_risk,  # Invert so higher is better
                    'max': 100,
                    'color': self._get_quality_color(100 - bias_risk),
                    'label': f'{bias_risk:.1f}% Risk'
                }
            }
        
        return charts
    
    def _generate_distribution_charts(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data distribution analysis charts."""
        charts = {}
        
        schema = processed_data.get('schema', {})
        sample_data = processed_data.get('sample_data', [])
        
        if not schema or not sample_data:
            return charts
        
        # Data completeness analysis
        completeness_data = []
        for field_name, field_info in schema.items():
            if isinstance(field_info, dict):
                null_count = field_info.get('null_count', 0)

                # Ensure null_count is an integer
                if not isinstance(null_count, (int, float)):
                    try:
                        null_count = int(null_count) if null_count else 0
                    except (ValueError, TypeError):
                        null_count = 0

                total_count = len(sample_data)
                completeness = ((total_count - null_count) / total_count * 100) if total_count > 0 else 0
                
                completeness_data.append({
                    'field': field_name,
                    'completeness': round(completeness, 1),
                    'color': self._get_quality_color(completeness)
                })
        
        if completeness_data:
            charts['data_completeness'] = {
                'type': 'bar',
                'title': 'Data Completeness by Field',
                'data': completeness_data,
                'xAxis': 'field',
                'yAxis': 'completeness',
                'max': 100
            }
        
        # Unique values analysis
        uniqueness_data = []
        for field_name, field_info in schema.items():
            if isinstance(field_info, dict):
                unique_count = field_info.get('unique_count', 0)

                # Ensure unique_count is an integer
                if not isinstance(unique_count, (int, float)):
                    try:
                        unique_count = int(unique_count) if unique_count else 0
                    except (ValueError, TypeError):
                        unique_count = 0

                total_count = len(sample_data)
                uniqueness = (unique_count / total_count * 100) if total_count > 0 else 0
                
                uniqueness_data.append({
                    'field': field_name,
                    'uniqueness': round(uniqueness, 1),
                    'unique_count': unique_count,
                    'color': self.chart_colors[len(uniqueness_data) % len(self.chart_colors)]
                })
        
        if uniqueness_data:
            charts['data_uniqueness'] = {
                'type': 'scatter',
                'title': 'Data Uniqueness Analysis',
                'data': uniqueness_data,
                'xAxis': 'field',
                'yAxis': 'uniqueness'
            }
        
        return charts
    
    def _generate_advanced_visualizations(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced visualizations using matplotlib/seaborn."""
        charts = {}
        
        if not VISUALIZATION_LIBS_AVAILABLE:
            return charts
        
        try:
            sample_data = processed_data.get('sample_data', [])
            schema = processed_data.get('schema', {})
            
            if not sample_data or not schema:
                return charts
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(sample_data)
            
            # Generate correlation heatmap for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_columns].corr()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5)
                plt.title('Correlation Matrix of Numeric Fields')
                plt.tight_layout()
                
                # Convert to base64 for web display
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                charts['correlation_heatmap'] = {
                    'type': 'image',
                    'title': 'Correlation Matrix',
                    'data': {
                        'image': f'data:image/png;base64,{image_base64}',
                        'format': 'png'
                    }
                }
            
            # Generate distribution plots for numeric columns
            for i, column in enumerate(numeric_columns[:3]):  # Limit to first 3 columns
                plt.figure(figsize=(8, 6))
                
                # Remove NaN values
                clean_data = df[column].dropna()
                
                if len(clean_data) > 0:
                    plt.hist(clean_data, bins=20, alpha=0.7, color=self.chart_colors[i % len(self.chart_colors)])
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    
                    # Convert to base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
                    
                    charts[f'distribution_{column}'] = {
                        'type': 'image',
                        'title': f'Distribution of {column}',
                        'data': {
                            'image': f'data:image/png;base64,{image_base64}',
                            'format': 'png'
                        }
                    }
        
        except Exception as e:
            logger.warning(f"Error generating advanced visualizations: {e}")
        
        return charts
    
    def _generate_visualization_summary(self, charts: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all visualizations."""
        return {
            'total_charts': len(charts),
            'chart_types': list(set(chart.get('type', 'unknown') for chart in charts.values())),
            'has_quality_charts': any('quality' in name for name in charts.keys()),
            'has_statistical_charts': any('statistic' in name or 'distribution' in name for name in charts.keys()),
            'has_ai_charts': any('ai' in name or 'bias' in name or 'ethics' in name for name in charts.keys()),
            'interactive_charts': len([c for c in charts.values() if c.get('type') in ['bar', 'pie', 'radar', 'scatter']]),
            'static_images': len([c for c in charts.values() if c.get('type') == 'image'])
        }
    
    def _generate_fallback_visualizations(self, dataset, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic fallback visualizations."""
        return {
            'generated_at': datetime.utcnow().isoformat(),
            'dataset_id': str(dataset.id),
            'charts': {
                'basic_info': {
                    'type': 'metric',
                    'title': 'Dataset Information',
                    'data': {
                        'record_count': processed_data.get('record_count', 0) if processed_data else 0,
                        'field_count': len(processed_data.get('schema', {})) if processed_data else 0
                    }
                }
            },
            'summary': {
                'total_charts': 1,
                'chart_types': ['metric'],
                'status': 'fallback'
            },
            'export_formats': ['json']
        }
    
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score."""
        if score >= 80:
            return self.quality_colors['excellent']
        elif score >= 60:
            return self.quality_colors['good']
        elif score >= 40:
            return self.quality_colors['fair']
        elif score >= 20:
            return self.quality_colors['poor']
        else:
            return self.quality_colors['critical']
    
    def export_visualizations(self, visualizations: Dict[str, Any], format: str = 'json') -> Any:
        """Export visualizations in specified format."""
        if format == 'json':
            return json.dumps(visualizations, indent=2)
        elif format == 'html':
            return self._generate_html_report(visualizations)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_report(self, visualizations: Dict[str, Any]) -> str:
        """Generate HTML report with embedded visualizations."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualizations</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .metric { text-align: center; font-size: 24px; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Dataset Visualizations</h1>
        """
        
        charts = visualizations.get('charts', {})
        for chart_name, chart_data in charts.items():
            html += f'<div class="chart-container">'
            html += f'<h3>{chart_data.get("title", chart_name)}</h3>'
            
            if chart_data.get('type') == 'metric':
                data = chart_data.get('data', {})
                html += f'<div class="metric">{data.get("value", "N/A")} {data.get("label", "")}</div>'
            elif chart_data.get('type') == 'image':
                image_data = chart_data.get('data', {}).get('image', '')
                html += f'<img src="{image_data}" alt="{chart_data.get("title", "")}" style="max-width: 100%;">'
            else:
                html += f'<pre>{json.dumps(chart_data.get("data", {}), indent=2)}</pre>'
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html


# Global service instance
data_visualization_service = DataVisualizationService()
