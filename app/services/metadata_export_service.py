"""
Metadata Export Service for generating Markdown, JSON, and PDF exports of dataset metadata.
"""

import os
import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import current_app
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class MetadataExportService:
    """Service for exporting dataset metadata to various formats"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.export_dir = os.path.join(current_app.root_path, 'static', 'exports')
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_metadata_markdown(self, dataset_id: str) -> Optional[str]:
        """
        Export dataset metadata as a Markdown file.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Path to the generated Markdown file or None if failed
        """
        try:
            dataset = Dataset.find_by_id(dataset_id)
            if not dataset:
                return None
            
            metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
            
            # Generate Markdown content
            markdown_content = self._generate_markdown_content(dataset, metadata_quality)
            
            # Save to file
            filename = f"{dataset.title.replace(' ', '_')}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            file_path = os.path.join(self.export_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting metadata to Markdown: {e}")
            print(f"Error exporting metadata to Markdown: {e}")
            return None
    
    def export_metadata_pdf(self, dataset_id: str) -> Optional[str]:
        """
        Export dataset metadata as a PDF file.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Path to the generated PDF file or None if failed
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available for PDF generation")
            print("ReportLab not available for PDF generation")
            return None
        
        try:
            dataset = Dataset.find_by_id(dataset_id)
            if not dataset:
                return None
            
            metadata_quality = MetadataQuality.get_by_dataset(dataset_id)
            
            # Generate PDF
            filename = f"{dataset.title.replace(' ', '_')}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path = os.path.join(self.export_dir, filename)
            
            self._generate_pdf_content(dataset, metadata_quality, file_path)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting metadata to PDF: {e}")
            print(f"Error exporting metadata to PDF: {e}")
            return None

    def export_metadata_json(self, dataset_id: str) -> Optional[str]:
        """
        Export dataset metadata as a JSON file.

        Args:
            dataset_id: ID of the dataset

        Returns:
            Path to the generated JSON file or None if failed
        """
        try:
            dataset = Dataset.find_by_id(dataset_id)
            if not dataset:
                logger.error(f"Dataset not found: {dataset_id}")
                return None

            metadata_quality = MetadataQuality.get_by_dataset(dataset_id)

            # Generate comprehensive JSON content
            json_content = self._generate_comprehensive_metadata_dict(dataset, metadata_quality)

            # Save to file
            filename = f"{dataset.title.replace(' ', '_')}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(self.export_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Successfully exported metadata to JSON: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error exporting metadata to JSON: {e}")
            print(f"Error exporting metadata to JSON: {e}")
            return None

    def _generate_markdown_content(self, dataset: Dataset, metadata_quality: Optional[MetadataQuality]) -> str:
        """Generate Markdown content for dataset metadata"""
        
        content = f"""# {dataset.title}

## Dataset Metadata Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset ID:** {dataset.id}  
**Export Type:** Complete Metadata Export

---

## 📊 Basic Information

| Field | Value |
|-------|-------|
| **Title** | {dataset.title} |
| **Source** | {dataset.source or 'Not specified'} |
| **Category** | {dataset.category or 'Not specified'} |
| **Data Type** | {dataset.data_type or 'Not specified'} |
| **Format** | {dataset.format or 'Unknown'} |
| **File Size** | {getattr(dataset, 'file_size', 'Unknown')} |
| **License** | {dataset.license or 'Not specified'} |
| **Created** | {dataset.created_at.strftime('%Y-%m-%d %H:%M:%S') if dataset.created_at else 'Unknown'} |
| **Updated** | {dataset.updated_at.strftime('%Y-%m-%d %H:%M:%S') if dataset.updated_at else 'Unknown'} |

## 📝 Description

{dataset.description or 'No description available.'}

## 📈 Data Statistics

| Metric | Value |
|--------|-------|
| **Records** | {dataset.record_count or 'Unknown'} |
| **Fields** | {dataset.field_count or 'Unknown'} |
| **Status** | {dataset.status or 'Unknown'} |

"""

        # Add field information if available
        if dataset.field_names:
            content += "## 🏷️ Field Names\n\n"
            fields = dataset.field_names.split(',')
            for i, field in enumerate(fields[:20], 1):  # Limit to first 20 fields
                content += f"{i}. {field.strip()}\n"
            if len(fields) > 20:
                content += f"\n*... and {len(fields) - 20} more fields*\n"
            content += "\n"

        # Add data types if available
        if dataset.data_types:
            content += "## 🔢 Data Types\n\n"
            data_types = dataset.data_types.split(',')
            for dtype in data_types[:10]:  # Limit to first 10 types
                content += f"- {dtype.strip()}\n"
            content += "\n"

        # Add tags if available
        if dataset.tags:
            content += "## 🏷️ Tags\n\n"
            tags = dataset.tags.split(',')
            for tag in tags[:15]:  # Limit to first 15 tags
                content += f"- #{tag.strip()}\n"
            content += "\n"

        # Add keywords if available
        if dataset.keywords:
            content += "## 🔍 Keywords\n\n"
            keywords = dataset.keywords.split(',')
            for keyword in keywords[:15]:  # Limit to first 15 keywords
                content += f"- {keyword.strip()}\n"
            content += "\n"

        # Add use cases if available
        if dataset.use_cases:
            content += "## 💡 Use Cases\n\n"
            use_cases = dataset.use_cases.split(',')
            for use_case in use_cases[:10]:  # Limit to first 10 use cases
                content += f"- {use_case.strip()}\n"
            content += "\n"

        # Add quality metrics if available
        if metadata_quality:
            content += f"""## 📊 Quality Assessment

| Metric | Score |
|--------|-------|
| **Overall Quality** | {metadata_quality.quality_score or 0}% |
| **Completeness** | {metadata_quality.completeness or 0}% |
| **Consistency** | {metadata_quality.consistency or 0}% |
| **Accuracy** | {metadata_quality.accuracy or 'N/A'}% |

"""

        # Add FAIR compliance
        content += f"""## 🎯 FAIR Compliance

| Principle | Score | Status |
|-----------|-------|--------|
| **FAIR Score** | {dataset.fair_score or 0}% | {'✅ Compliant' if dataset.fair_compliant else '⚠️ Partial'} |
| **Findable** | {getattr(dataset, 'findable_score', 'N/A')} | {'✅' if getattr(dataset, 'findable_score', 0) >= 75 else '⚠️'} |
| **Accessible** | {getattr(dataset, 'accessible_score', 'N/A')} | {'✅' if getattr(dataset, 'accessible_score', 0) >= 75 else '⚠️'} |
| **Interoperable** | {getattr(dataset, 'interoperable_score', 'N/A')} | {'✅' if getattr(dataset, 'interoperable_score', 0) >= 75 else '⚠️'} |
| **Reusable** | {getattr(dataset, 'reusable_score', 'N/A')} | {'✅' if getattr(dataset, 'reusable_score', 0) >= 75 else '⚠️'} |

"""

        # Add compliance information
        content += f"""## ✅ Standards Compliance

- **Schema.org Compliant:** {'✅ Yes' if getattr(dataset, 'schema_org_compliant', False) else '❌ No'}
- **Persistent Identifier:** {'✅ Assigned' if dataset.persistent_identifier else '❌ Not assigned'}
- **Encoding Format:** {dataset.encoding_format or 'Not specified'}

"""

        # Add metadata standards
        if dataset.dublin_core or dataset.dcat_metadata or dataset.json_ld:
            content += "## 📋 Available Metadata Standards\n\n"
            if dataset.dublin_core:
                content += "- ✅ Dublin Core\n"
            if dataset.dcat_metadata:
                content += "- ✅ DCAT (Data Catalog Vocabulary)\n"
            if dataset.json_ld:
                content += "- ✅ Schema.org JSON-LD\n"
            content += "\n"

        # Add health report if available
        health_report = self._parse_json_field(getattr(dataset, 'health_report', None))
        if health_report:
            content += "## 🏥 Health Report\n\n"
            if isinstance(health_report, dict):
                for key, value in health_report.items():
                    content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
            else:
                content += f"{health_report}\n"
            content += "\n"

        # Add visualizations summary if available
        visualizations = self._parse_json_field(getattr(dataset, 'visualizations', None))
        if visualizations and isinstance(visualizations, dict):
            content += "## 📊 Visualizations\n\n"
            charts = visualizations.get('charts', {})
            if charts:
                content += f"**Available Charts**: {len(charts)} visualizations generated\n\n"
                for chart_name, chart_info in list(charts.items())[:5]:  # Show first 5
                    chart_type = chart_info.get('type', 'Unknown')
                    chart_title = chart_info.get('title', chart_name)
                    content += f"- **{chart_title}** ({chart_type})\n"
                if len(charts) > 5:
                    content += f"\n*... and {len(charts) - 5} more visualizations*\n"
            content += "\n"

        # Add Python code examples
        content += "## 🐍 Python Code Examples\n\n"
        python_examples = self._generate_python_examples_dict(dataset)
        for example_name, code in python_examples.items():
            content += f"### {example_name.replace('_', ' ').title()}\n\n"
            content += f"```python\n{code}\n```\n\n"

        # Add AI compliance if available
        ai_compliance = self._parse_json_field(getattr(dataset, 'ai_compliance', None))
        if ai_compliance and isinstance(ai_compliance, dict):
            content += "## 🤖 AI Compliance\n\n"
            for key, value in ai_compliance.items():
                if isinstance(value, (int, float)):
                    content += f"- **{key.replace('_', ' ').title()}**: {value}%\n"
                else:
                    content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
            content += "\n"

        # Add NLP analysis results
        entities = self._parse_json_field(getattr(dataset, 'entities', None))
        sentiment = self._parse_json_field(getattr(dataset, 'sentiment', None))
        if entities or sentiment:
            content += "## 🧠 NLP Analysis Results\n\n"

            if entities and isinstance(entities, list):
                content += "### Named Entities\n"
                entity_summary = {}
                for entity in entities[:10]:  # Show first 10 entities
                    label = entity.get('label', 'UNKNOWN')
                    entity_summary[label] = entity_summary.get(label, 0) + 1

                for label, count in entity_summary.items():
                    content += f"- **{label}**: {count} occurrences\n"
                content += "\n"

            if sentiment and isinstance(sentiment, dict):
                content += "### Sentiment Analysis\n"
                for key, value in sentiment.items():
                    if isinstance(value, (int, float)):
                        content += f"- **{key.replace('_', ' ').title()}**: {value:.2f}\n"
                    else:
                        content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                content += "\n"

        # Add data quality metrics
        content += "## 📊 Data Quality Metrics\n\n"
        quality_metrics = [
            ('completeness_score', 'Completeness'),
            ('consistency_score', 'Consistency'),
            ('accuracy_score', 'Accuracy'),
            ('validity_score', 'Validity'),
            ('uniqueness_score', 'Uniqueness')
        ]

        for attr, label in quality_metrics:
            value = getattr(dataset, attr, None)
            if value is not None:
                content += f"- **{label}**: {value:.1f}%\n"

        missing_count = getattr(dataset, 'missing_values_count', None)
        duplicate_count = getattr(dataset, 'duplicate_records_count', None)
        if missing_count is not None:
            content += f"- **Missing Values**: {missing_count:,}\n"
        if duplicate_count is not None:
            content += f"- **Duplicate Records**: {duplicate_count:,}\n"
        content += "\n"

        # Add technical metadata
        content += "## ⚙️ Technical Metadata\n\n"
        content += f"- **File Format**: {dataset.format.upper()}\n"

        encoding = getattr(dataset, 'encoding', None)
        if encoding:
            content += f"- **Encoding**: {encoding}\n"

        delimiter = getattr(dataset, 'delimiter', None)
        if delimiter:
            content += f"- **Delimiter**: `{delimiter}`\n"

        compression = getattr(dataset, 'compression', None)
        if compression:
            content += f"- **Compression**: {compression}\n"

        content += f"- **Upload Date**: {dataset.created_at.strftime('%Y-%m-%d %H:%M:%S') if dataset.created_at else 'Unknown'}\n"
        content += f"- **Last Modified**: {dataset.updated_at.strftime('%Y-%m-%d %H:%M:%S') if dataset.updated_at else 'Unknown'}\n"
        content += "\n"

        # Add footer
        content += f"""---

## 📄 Export Information

- **Export Format:** Markdown
- **Generated by:**  Metadata Harvesting System
- **Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset URL:** {dataset.source_url or 'N/A'}

*This metadata export contains comprehensive information about the dataset including quality metrics, FAIR compliance assessment, and standards compliance.*
"""

        return content
    
    def _generate_pdf_content(self, dataset: Dataset, metadata_quality: Optional[MetadataQuality], file_path: str):
        """Generate PDF content for dataset metadata"""
        
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph(f"Dataset Metadata Report: {dataset.title}", title_style))
        story.append(Spacer(1, 12))
        
        # Basic information table
        basic_info_data = [
            ['Field', 'Value'],
            ['Title', dataset.title],
            ['Source', dataset.source or 'Not specified'],
            ['Category', dataset.category or 'Not specified'],
            ['Format', dataset.format or 'Unknown'],
            ['Records', str(dataset.record_count or 'Unknown')],
            ['Fields', str(dataset.field_count or 'Unknown')],
            ['Created', dataset.created_at.strftime('%Y-%m-%d') if dataset.created_at else 'Unknown']
        ]
        
        basic_info_table = Table(basic_info_data, colWidths=[2*inch, 4*inch])
        basic_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Basic Information", styles['Heading2']))
        story.append(basic_info_table)
        story.append(Spacer(1, 12))
        
        # Description
        if dataset.description:
            story.append(Paragraph("Description", styles['Heading2']))
            story.append(Paragraph(dataset.description, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Quality metrics if available
        if metadata_quality:
            story.append(Paragraph("Quality Assessment", styles['Heading2']))
            quality_data = [
                ['Metric', 'Score'],
                ['Overall Quality', f"{metadata_quality.quality_score or 0}%"],
                ['Completeness', f"{metadata_quality.completeness or 0}%"],
                ['Consistency', f"{metadata_quality.consistency or 0}%"],
                ['FAIR Score', f"{dataset.fair_score or 0}%"]
            ]
            
            quality_table = Table(quality_data, colWidths=[3*inch, 2*inch])
            quality_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(quality_table)
            story.append(Spacer(1, 12))
        
        # Footer
        story.append(Spacer(1, 24))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Metadata Generation System To Support AI Research Initiative", footer_style))
        
        # Build PDF
        doc.build(story)

    def _generate_comprehensive_metadata_dict(self, dataset: Dataset, metadata_quality: Optional[MetadataQuality]) -> Dict[str, Any]:
        """Generate comprehensive metadata dictionary including all available fields"""

        # Basic dataset information
        metadata = {
            'export_info': {
                'export_format': 'JSON',
                'export_date': datetime.now().isoformat(),
                'generated_by': ' Metadata Harvesting System',
                'version': '2.0'
            },
            'basic_info': {
                'id': str(dataset.id),
                'title': dataset.title,
                'source': dataset.source,
                'category': dataset.category,
                'data_type': dataset.data_type,
                'format': dataset.format,
                'file_size': getattr(dataset, 'file_size', None),
                'license': dataset.license,
                'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
                'updated_at': dataset.updated_at.isoformat() if dataset.updated_at else None,
                'source_url': dataset.source_url,
                'persistent_identifier': dataset.persistent_identifier,
                'encoding_format': dataset.encoding_format,
                'status': dataset.status
            },
            'description': {
                'main_description': dataset.description,
                'structured_description': self._parse_json_field(dataset.structured_description),
                'auto_generated_description': getattr(dataset, 'auto_generated_description', None)
            },
            'data_statistics': {
                'record_count': dataset.record_count,
                'field_count': dataset.field_count,
                'field_names': dataset.field_names.split(',') if dataset.field_names else [],
                'data_types': dataset.data_types.split(',') if dataset.data_types else [],
                'data_distribution_types': self._parse_json_field(getattr(dataset, 'data_distribution_types', None))
            },
            'metadata_fields': {
                'tags': dataset.tags.split(',') if dataset.tags else [],
                'keywords': dataset.keywords.split(',') if dataset.keywords else [],
                'use_cases': dataset.use_cases.split(',') if dataset.use_cases else [],
                'entities': self._parse_json_field(getattr(dataset, 'entities', None)),
                'sentiment': self._parse_json_field(getattr(dataset, 'sentiment', None))
            }
        }

        # Quality assessment
        if metadata_quality:
            metadata['quality_assessment'] = {
                'overall_quality_score': metadata_quality.quality_score,
                'completeness': metadata_quality.completeness,
                'consistency': metadata_quality.consistency,
                'accuracy': metadata_quality.accuracy,
                'timeliness': metadata_quality.timeliness,
                'conformity': metadata_quality.conformity,
                'integrity': metadata_quality.integrity,
                'issues': metadata_quality.issues_list,
                'recommendations': metadata_quality.recommendations_list,
                'assessment_date': metadata_quality.assessment_date.isoformat() if metadata_quality.assessment_date else None
            }

        # FAIR compliance
        metadata['fair_compliance'] = {
            'overall_score': dataset.fair_score,
            'is_compliant': dataset.fair_compliant,
            'findable_score': getattr(dataset, 'findable_score', None),
            'accessible_score': getattr(dataset, 'accessible_score', None),
            'interoperable_score': getattr(dataset, 'interoperable_score', None),
            'reusable_score': getattr(dataset, 'reusable_score', None)
        }

        # Standards compliance
        metadata['standards_compliance'] = {
            'schema_org_compliant': getattr(dataset, 'schema_org_compliant', False),
            'dublin_core': self._parse_json_field(dataset.dublin_core),
            'dcat_metadata': self._parse_json_field(dataset.dcat_metadata),
            'json_ld': self._parse_json_field(dataset.json_ld)
        }

        # Visualizations
        metadata['visualizations'] = self._parse_json_field(getattr(dataset, 'visualizations', None))

        # Health reports and additional metadata
        metadata['health_report'] = self._parse_json_field(getattr(dataset, 'health_report', None))
        metadata['ai_compliance'] = self._parse_json_field(getattr(dataset, 'ai_compliance', None))
        metadata['processing_metadata'] = self._parse_json_field(getattr(dataset, 'processing_metadata', None))

        # Python code examples
        metadata['python_examples'] = self._generate_python_examples_dict(dataset)

        # Collection information if applicable
        if dataset.is_collection or dataset.collection_id:
            metadata['collection_info'] = {
                'is_collection': dataset.is_collection,
                'collection_id': dataset.collection_id,
                'parent_collection': str(dataset.parent_collection.id) if dataset.parent_collection else None,
                'collection_files': self._parse_json_field(dataset.collection_files),
                'auto_generated_title': dataset.auto_generated_title
            }

        # Advanced NLP analysis results
        metadata['nlp_analysis'] = {
            'entities': self._parse_json_field(getattr(dataset, 'entities', None)),
            'sentiment': self._parse_json_field(getattr(dataset, 'sentiment', None)),
            'summary': getattr(dataset, 'summary', None),
            'language_detected': getattr(dataset, 'language', 'en'),
            'content_analysis': self._parse_json_field(getattr(dataset, 'content_analysis', None))
        }

        # Technical metadata
        metadata['technical_metadata'] = {
            'file_format': dataset.format,
            'encoding': getattr(dataset, 'encoding', None),
            'delimiter': getattr(dataset, 'delimiter', None),
            'compression': getattr(dataset, 'compression', None),
            'checksum': getattr(dataset, 'checksum', None),
            'file_path': getattr(dataset, 'file_path', None),
            'upload_date': dataset.created_at.isoformat() if dataset.created_at else None,
            'last_modified': dataset.updated_at.isoformat() if dataset.updated_at else None
        }

        # Data quality metrics
        metadata['data_quality_metrics'] = {
            'completeness_score': getattr(dataset, 'completeness_score', None),
            'consistency_score': getattr(dataset, 'consistency_score', None),
            'accuracy_score': getattr(dataset, 'accuracy_score', None),
            'validity_score': getattr(dataset, 'validity_score', None),
            'uniqueness_score': getattr(dataset, 'uniqueness_score', None),
            'missing_values_count': getattr(dataset, 'missing_values_count', None),
            'duplicate_records_count': getattr(dataset, 'duplicate_records_count', None)
        }

        # Statistical summary
        metadata['statistical_summary'] = self._parse_json_field(getattr(dataset, 'statistical_summary', None))

        # Data lineage and provenance
        metadata['data_lineage'] = {
            'source_system': getattr(dataset, 'source_system', None),
            'collection_method': getattr(dataset, 'collection_method', None),
            'processing_history': self._parse_json_field(getattr(dataset, 'processing_history', None)),
            'transformation_applied': self._parse_json_field(getattr(dataset, 'transformations', None)),
            'validation_rules': self._parse_json_field(getattr(dataset, 'validation_rules', None))
        }

        return metadata

    def _parse_json_field(self, field_value) -> Any:
        """Parse JSON field value safely"""
        if not field_value:
            return None

        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except (json.JSONDecodeError, ValueError):
                return field_value

        return field_value

    def _generate_python_examples_dict(self, dataset: Dataset) -> Dict[str, str]:
        """Generate Python code examples for the dataset"""
        examples = {}

        # Get dataset-specific information
        field_names = dataset.field_names.split(',') if dataset.field_names else []
        field_names = [f.strip() for f in field_names[:5]]  # Limit to first 5 fields

        # Basic loading example
        examples['basic_loading'] = f"""# Load {dataset.title} dataset
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('{dataset.title.replace(' ', '_').lower()}.csv')

# Basic information
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(df.head())"""

        # Data exploration example
        if field_names:
            examples['data_exploration'] = f"""# Data exploration for {dataset.title}
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset overview
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Basic visualizations
{chr(10).join([f"plt.figure(figsize=(10, 6))" if i == 0 else f"# Analyze {field}" for i, field in enumerate(field_names[:3])])}"""

        return examples


def get_metadata_export_service():
    """Get metadata export service instance"""
    return MetadataExportService()
