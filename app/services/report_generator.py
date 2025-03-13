"""
Report Generator - Generate comprehensive dataset health reports
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Union
import base64
from io import BytesIO

from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality
from app.services.quality_assessment_service import quality_assessment_service

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Service for generating comprehensive dataset health reports.
    
    This service:
    1. Collects dataset information and quality metrics
    2. Generates PDF or HTML reports
    3. Provides downloadable and shareable reports
    """
    
    def __init__(self, templates_folder: str):
        """
        Initialize report generator.
        
        Args:
            templates_folder: Path to report templates
        """
        self.templates_folder = templates_folder
    
    def generate_report(self, dataset_id: int, report_format: str = 'html') -> Dict[str, Any]:
        """
        Generate a health report for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            report_format: Format of the report ('html' or 'pdf')
            
        Returns:
            Dictionary with report data
        """
        try:
            # Get dataset
            dataset = Dataset.find_by_id(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Get quality assessment
            quality = MetadataQuality.get_by_dataset(dataset_id)
            if not quality:
                raise ValueError(f"Quality assessment for dataset with ID {dataset_id} not found")
            
            # Generate timestamp
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Generate report ID
            report_id = f"report_{dataset_id}_{timestamp}"
            
            # Generate report data
            report_data = self._collect_report_data(dataset, quality)
            
            # Create report metadata
            report_metadata = {
                'id': report_id,
                'dataset_id': dataset_id,
                'dataset_title': dataset.title,
                'generated_at': datetime.datetime.utcnow().isoformat(),
                'format': report_format
            }
            
            # Generate report in requested format
            if report_format == 'html':
                report_content = self._generate_html_report(report_data, report_metadata)
                report_path = self._save_html_report(report_content, report_id)
            elif report_format == 'pdf':
                report_content = self._generate_pdf_report(report_data, report_metadata)
                report_path = self._save_pdf_report(report_content, report_id)
            else:
                raise ValueError(f"Unsupported report format: {report_format}")
            
            # Create report response
            report_response = {
                'id': report_id,
                'dataset_id': dataset_id,
                'dataset_title': dataset.title,
                'format': report_format,
                'generated_at': report_metadata['generated_at'],
                'path': report_path
            }
            
            return report_response
            
        except Exception as e:
            logger.error(f"Error generating report for dataset {dataset_id}: {str(e)}")
            raise
    
    def _collect_report_data(self, dataset: Dataset, quality: MetadataQuality) -> Dict[str, Any]:
        """
        Collect all data needed for the report.
        
        Args:
            dataset: Dataset object
            quality: MetadataQuality object
            
        Returns:
            Dictionary with all report data
        """
        # Dataset information
        dataset_info = dataset.to_dict()
        
        # Quality metrics
        quality_metrics = quality.to_dict()
        
        # Get dimension details from quality assessment service
        dimension_details = self._get_dimension_details(dataset.id)
        
        # Prepare recommendations
        recommendations = quality.recommendations_list
        
        # Prepare issues
        issues = quality.issues_list
        
        # Collect all data
        report_data = {
            'dataset': dataset_info,
            'quality': quality_metrics,
            'dimension_details': dimension_details,
            'recommendations': recommendations,
            'issues': issues
        }
        
        return report_data
    
    def _get_dimension_details(self, dataset_id: int) -> Dict[str, Any]:
        """
        Get detailed information for each quality dimension.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary with dimension details
        """
        # Get dimension scores
        dimension_scores = quality_assessment_service.get_dimension_scores(dataset_id)
        
        # Get FAIR scores
        fair_scores = quality_assessment_service.get_fair_scores(dataset_id)
        
        # Create dimension details
        dimension_details = {
            'dimensions': dimension_scores,
            'fair': fair_scores
        }
        
        return dimension_details
    
    def _generate_html_report(self, report_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Generate HTML report.
        
        Args:
            report_data: Report data
            metadata: Report metadata
            
        Returns:
            HTML content as string
        """
        # Get HTML template
        template_path = os.path.join(self.templates_folder, 'report_template.html')
        
        try:
            with open(template_path, 'r') as file:
                template = file.read()
        except FileNotFoundError:
            # Use inline template if file not found
            template = self._get_default_html_template()
        
        # Format dataset info
        dataset_info = report_data['dataset']
        dataset_html = self._format_dataset_info_html(dataset_info)
        
        # Format quality metrics
        quality_metrics = report_data['quality']
        quality_html = self._format_quality_metrics_html(quality_metrics, report_data['dimension_details'])
        
        # Format recommendations
        recommendations = report_data['recommendations']
        recommendations_html = self._format_recommendations_html(recommendations)
        
        # Format issues
        issues = report_data['issues']
        issues_html = self._format_issues_html(issues)
        
        # Replace placeholders in template
        html_content = template.replace('{{REPORT_TITLE}}', f"Dataset Health Report: {dataset_info['title']}")
        html_content = html_content.replace('{{REPORT_DATE}}', metadata['generated_at'])
        html_content = html_content.replace('{{DATASET_INFO}}', dataset_html)
        html_content = html_content.replace('{{QUALITY_METRICS}}', quality_html)
        html_content = html_content.replace('{{RECOMMENDATIONS}}', recommendations_html)
        html_content = html_content.replace('{{ISSUES}}', issues_html)
        
        return html_content
    
    def _format_dataset_info_html(self, dataset_info: Dict[str, Any]) -> str:
        """
        Format dataset information as HTML.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            HTML string
        """
        html = '<div class="section">'
        html += '<h2>Dataset Information</h2>'
        html += '<table class="info-table">'
        
        # Basic info
        html += f'<tr><td class="label">Title:</td><td>{dataset_info.get("title", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Description:</td><td>{dataset_info.get("description", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Source:</td><td>{dataset_info.get("source", "N/A")}</td></tr>'
        
        # Format
        html += f'<tr><td class="label">Format:</td><td>{dataset_info.get("format", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Size:</td><td>{dataset_info.get("size", "N/A")}</td></tr>'
        
        # Type and category
        html += f'<tr><td class="label">Data Type:</td><td>{dataset_info.get("data_type", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Category:</td><td>{dataset_info.get("category", "N/A")}</td></tr>'
        
        # Dates
        created_at = dataset_info.get('created_at')
        if created_at:
            if isinstance(created_at, str):
                created_at_str = created_at
            else:
                created_at_str = created_at.strftime('%Y-%m-%d %H:%M:%S')
            html += f'<tr><td class="label">Created At:</td><td>{created_at_str}</td></tr>'
        
        updated_at = dataset_info.get('updated_at')
        if updated_at:
            if isinstance(updated_at, str):
                updated_at_str = updated_at
            else:
                updated_at_str = updated_at.strftime('%Y-%m-%d %H:%M:%S')
            html += f'<tr><td class="label">Updated At:</td><td>{updated_at_str}</td></tr>'
        
        # Tags
        tags = dataset_info.get("tags", [])
        if tags:
            tags_html = ', '.join([f'<span class="tag">{tag}</span>' for tag in tags])
            html += f'<tr><td class="label">Tags:</td><td>{tags_html}</td></tr>'
        
        html += '</table>'
        html += '</div>'
        
        return html
    
    def _format_quality_metrics_html(self, metrics: Dict[str, Any], dimension_details: Dict[str, Any]) -> str:
        """
        Format quality metrics as HTML.
        
        Args:
            metrics: Quality metrics
            dimension_details: Dimension details
            
        Returns:
            HTML string
        """
        html = '<div class="section">'
        html += '<h2>Quality Assessment</h2>'
        
        # Overall score
        quality_score = metrics.get('quality_score', 0)
        score_class = self._get_score_class(quality_score)
        
        html += '<div class="overall-score">'
        html += f'<div class="score-circle {score_class}">{int(quality_score)}</div>'
        html += '<div class="score-label">Overall Quality Score</div>'
        html += '</div>'
        
        # Dimension scores
        html += '<h3>Quality Dimensions</h3>'
        html += '<div class="dimensions">'
        
        dimensions = dimension_details.get('dimensions', {})
        for dim_name, dim_score in dimensions.items():
            score_class = self._get_score_class(dim_score)
            html += f'<div class="dimension-item">'
            html += f'<div class="dimension-name">{dim_name.capitalize()}</div>'
            html += f'<div class="progress-bar-container">'
            html += f'<div class="progress-bar {score_class}" style="width: {dim_score}%;"></div>'
            html += f'</div>'
            html += f'<div class="dimension-score">{int(dim_score)}</div>'
            html += f'</div>'
        
        html += '</div>'
        
        # FAIR compliance
        html += '<h3>FAIR Principles Compliance</h3>'
        html += '<div class="fair-scores">'
        
        fair_scores = dimension_details.get('fair', {})
        
        # FAIR compliance status
        fair_compliant = metrics.get('fair_compliant', False)
        schema_org_compliant = metrics.get('schema_org_compliant', False)
        
        html += '<div class="compliance-status">'
        html += f'<div class="status-item"><span class="status-label">FAIR Compliant:</span> <span class="status-value {self._get_compliance_class(fair_compliant)}">{self._format_compliance(fair_compliant)}</span></div>'
        html += f'<div class="status-item"><span class="status-label">Schema.org Compliant:</span> <span class="status-value {self._get_compliance_class(schema_org_compliant)}">{self._format_compliance(schema_org_compliant)}</span></div>'
        html += '</div>'
        
        # FAIR scores
        for fair_name, fair_score in fair_scores.items():
            if fair_name == 'overall':
                continue
                
            score_class = self._get_score_class(fair_score)
            fair_letter = fair_name[0].upper() if fair_name != 'overall' else 'FAIR'
            
            html += f'<div class="fair-item">'
            html += f'<div class="fair-icon {score_class}">{fair_letter}</div>'
            html += f'<div>'
            html += f'<div class="fair-name">{fair_name.capitalize()}</div>'
            html += f'<div class="progress-bar-container">'
            html += f'<div class="progress-bar {score_class}" style="width: {fair_score}%;"></div>'
            html += f'</div>'
            html += f'</div>'
            html += f'<div class="fair-score">{int(fair_score)}</div>'
            html += f'</div>'
        
        html += '</div>'
        html += '</div>'
        
        return html
    
    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """
        Format recommendations as HTML.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            HTML string
        """
        html = '<div class="section">'
        html += '<h2>Recommendations</h2>'
        
        if recommendations:
            html += '<ul class="recommendations-list">'
            for rec in recommendations:
                html += f'<li><div class="recommendation-item"><i class="icon">💡</i> {rec}</div></li>'
            html += '</ul>'
        else:
            html += '<p class="empty-message">No recommendations available.</p>'
        
        html += '</div>'
        
        return html
    
    def _format_issues_html(self, issues: List[str]) -> str:
        """
        Format issues as HTML.
        
        Args:
            issues: List of issues
            
        Returns:
            HTML string
        """
        html = '<div class="section">'
        html += '<h2>Identified Issues</h2>'
        
        if issues:
            html += '<ul class="issues-list">'
            for issue in issues:
                html += f'<li><div class="issue-item"><i class="icon">⚠️</i> {issue}</div></li>'
            html += '</ul>'
        else:
            html += '<p class="empty-message">No issues identified.</p>'
        
        html += '</div>'
        
        return html
    
    def _get_score_class(self, score: float) -> str:
        """
        Get CSS class for a score.
        
        Args:
            score: Score value
            
        Returns:
            CSS class name
        """
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"
    
    def _get_compliance_class(self, compliant: bool) -> str:
        """
        Get CSS class for compliance status.
        
        Args:
            compliant: Whether compliant
            
        Returns:
            CSS class name
        """
        return "compliant" if compliant else "non-compliant"
    
    def _format_compliance(self, compliant: bool) -> str:
        """
        Format compliance status as string.
        
        Args:
            compliant: Whether compliant
            
        Returns:
            Formatted string
        """
        return "Yes" if compliant else "No"
    
    def _get_default_html_template(self) -> str:
        """
        Get default HTML template for reports.
        
        Returns:
            HTML template string
        """
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{REPORT_TITLE}}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        
        .report-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        
        .report-title {
            font-size: 28px;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .report-date {
            color: #7f8c8d;
            font-size: 14px;
        }
        
        .section {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 22px;
            margin-top: 0;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        h3 {
            color: #34495e;
            font-size: 18px;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        .info-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .info-table td {
            padding: 10px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }
        
        .info-table .label {
            font-weight: bold;
            width: 150px;
            color: #34495e;
        }
        
        .tag {
            display: inline-block;
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 5px;
        }
        
        .overall-score {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 30px 0;
        }
        
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 36px;
            font-weight: bold;
            color: white;
            margin-bottom: 10px;
        }
        
        .score-label {
            font-size: 16px;
            color: #555;
        }
        
        .excellent {
            background-color: #10b981;
        }
        
        .good {
            background-color: #3b82f6;
        }
        
        .fair {
            background-color: #f59e0b;
        }
        
        .poor {
            background-color: #ef4444;
        }
        
        .dimensions, .fair-scores {
            margin-top: 20px;
        }
        
        .dimension-item, .fair-item {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .dimension-name, .fair-name {
            width: 150px;
            font-weight: 500;
        }
        
        .fair-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
            font-weight: bold;
            margin-right: 15px;
        }
        
        .progress-bar-container {
            flex-grow: 1;
            height: 8px;
            background-color: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin: 0 15px;
        }
        
        .progress-bar {
            height: 100%;
            border-radius: 4px;
        }
        
        .dimension-score, .fair-score {
            width: 40px;
            text-align: right;
            font-weight: bold;
        }
        
        .compliance-status {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .status-item {
            background-color: #f8fafc;
            padding: 10px 15px;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }
        
        .status-label {
            font-weight: 500;
        }
        
        .status-value {
            margin-left: 5px;
            font-weight: bold;
        }
        
        .compliant {
            color: #10b981;
        }
        
        .non-compliant {
            color: #f59e0b;
        }
        
        .recommendations-list, .issues-list {
            list-style-type: none;
            padding-left: 0;
        }
        
        .recommendation-item, .issue-item {
            padding: 12px 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            display: flex;
            align-items: flex-start;
        }
        
        .recommendation-item {
            background-color: #f0fdf4;
            border-left: 4px solid #10b981;
        }
        
        .issue-item {
            background-color: #fff7ed;
            border-left: 4px solid #f59e0b;
        }
        
        .icon {
            margin-right: 12px;
            font-style: normal;
        }
        
        .empty-message {
            padding: 20px;
            text-align: center;
            color: #6b7280;
            font-style: italic;
            background-color: #f8fafc;
            border-radius: 6px;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #6b7280;
            font-size: 14px;
        }
        
        @media print {
            body {
                background-color: #fff;
                padding: 0;
            }
            
            .section {
                box-shadow: none;
                border: 1px solid #ddd;
                break-inside: avoid;
            }
            
            .report-header {
                break-after: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="report-header">
        <h1 class="report-title">{{REPORT_TITLE}}</h1>
        <div class="report-date">Generated on {{REPORT_DATE}}</div>
    </div>
    
    {{DATASET_INFO}}
    
    {{QUALITY_METRICS}}
    
    {{RECOMMENDATIONS}}
    
    {{ISSUES}}
    
    <footer>
        <p>Report generated by Dataset Metadata Manager</p>
    </footer>
</body>
</html>'''
    
    def _save_html_report(self, html_content: str, report_id: str) -> str:
        """
        Save HTML report to file.
        
        Args:
            html_content: HTML content
            report_id: Report ID
            
        Returns:
            Path to saved report
        """
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.getcwd(), 'app', 'static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save report
        report_filename = f"{report_id}.html"
        report_path = os.path.join(reports_dir, report_filename)
        
        with open(report_path, 'w') as file:
            file.write(html_content)
        
        # Return relative path for URL
        return f"/static/reports/{report_filename}"
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], metadata: Dict[str, Any]) -> bytes:
        """
        Generate PDF report.
        
        Args:
            report_data: Report data
            metadata: Report metadata
            
        Returns:
            PDF content as bytes
        """
        try:
            import pdfkit
            from jinja2 import Template
            
            # Get HTML content first
            html_content = self._generate_html_report(report_data, metadata)
            
            # Convert HTML to PDF
            pdf_content = pdfkit.from_string(html_content, False)
            
            return pdf_content
            
        except ImportError:
            # Fallback if pdfkit or jinja2 is not available
            logger.warning("pdfkit or jinja2 not available, falling back to HTML report")
            
            # Generate HTML report
            html_content = self._generate_html_report(report_data, metadata)
            
            # Convert to bytes
            return html_content.encode('utf-8')
    
    def _save_pdf_report(self, pdf_content: bytes, report_id: str) -> str:
        """
        Save PDF report to file.
        
        Args:
            pdf_content: PDF content as bytes
            report_id: Report ID
            
        Returns:
            Path to saved report
        """
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.getcwd(), 'app', 'static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save report
        report_filename = f"{report_id}.pdf"
        report_path = os.path.join(reports_dir, report_filename)
        
        with open(report_path, 'wb') as file:
            file.write(pdf_content)
        
        # Return relative path for URL
        return f"/static/reports/{report_filename}"

# Singleton instance for application-wide use
report_generator = None

def get_report_generator(templates_folder):
    """
    Get or create the report generator instance.
    
    Args:
        templates_folder: Path to templates folder
        
    Returns:
        ReportGenerator instance
    """
    global report_generator
    
    if report_generator is None:
        report_generator = ReportGenerator(templates_folder)
    
    return report_generator