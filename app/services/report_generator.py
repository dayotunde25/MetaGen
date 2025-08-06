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
        dimension_details = self._get_dimension_details(str(dataset.id))

        # Prepare recommendations
        recommendations = quality.recommendations_list

        # Prepare issues
        issues = quality.issues_list

        # Generate Python code examples
        python_examples = self._generate_python_examples(dataset)

        # Generate visualizations data
        visualizations = self._get_visualizations_data(dataset)

        # Collect all data
        report_data = {
            'dataset': dataset_info,
            'quality': quality_metrics,
            'dimension_details': dimension_details,
            'recommendations': recommendations,
            'issues': issues,
            'python_examples': python_examples,
            'visualizations': visualizations
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
            with open(template_path, 'r', encoding='utf-8') as file:
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

        # Format Python examples
        python_examples = report_data.get('python_examples', {})
        python_html = self._format_python_examples_html(python_examples)

        # Format visualizations
        visualizations = report_data.get('visualizations', {})
        visualizations_html = self._format_visualizations_html(visualizations)

        # Replace placeholders in template
        html_content = template.replace('{{REPORT_TITLE}}', f"Dataset Health Report: {dataset_info['title']}")
        html_content = html_content.replace('{{REPORT_DATE}}', metadata['generated_at'])
        html_content = html_content.replace('{{DATASET_INFO}}', dataset_html)
        html_content = html_content.replace('{{QUALITY_METRICS}}', quality_html)
        html_content = html_content.replace('{{RECOMMENDATIONS}}', recommendations_html)
        html_content = html_content.replace('{{ISSUES}}', issues_html)
        html_content = html_content.replace('{{PYTHON_EXAMPLES}}', python_html)
        html_content = html_content.replace('{{VISUALIZATIONS}}', visualizations_html)

        return html_content

    def _format_dataset_info_html(self, dataset_info: Dict[str, Any]) -> str:
        """
        Format dataset information as HTML with comprehensive metadata.

        Args:
            dataset_info: Dataset information

        Returns:
            HTML string
        """
        html = '<div class="section">'
        html += '<h2>Dataset Information</h2>'

        # Basic Information Table
        html += '<h3>Basic Information</h3>'
        html += '<table class="info-table">'
        html += f'<tr><td class="label">Title:</td><td>{dataset_info.get("title", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Source:</td><td>{dataset_info.get("source", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Category:</td><td>{dataset_info.get("category", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Data Type:</td><td>{dataset_info.get("data_type", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Format:</td><td>{dataset_info.get("format", "N/A")}</td></tr>'
        html += f'<tr><td class="label">Size:</td><td>{dataset_info.get("size", "N/A")}</td></tr>'
        html += '</table>'

        # Description Section
        html += '<h3>Description</h3>'
        description = dataset_info.get("description", "No description available")
        if description and description != "N/A":
            # Format description with proper line breaks
            formatted_description = description.replace('\n', '<br>')
            html += f'<div class="description-content">{formatted_description}</div>'
        else:
            html += '<p class="empty-message">No description provided.</p>'

        # Data Structure Information
        html += '<h3>Data Structure</h3>'
        html += '<table class="info-table">'
        html += f'<tr><td class="label">Record Count:</td><td>{dataset_info.get("record_count", "N/A"):,}</td></tr>'
        html += f'<tr><td class="label">Field Count:</td><td>{dataset_info.get("field_count", "N/A")}</td></tr>'

        # Field Names
        field_names = dataset_info.get("field_names", "")
        if field_names:
            field_list = [f.strip() for f in field_names.split(',')]
            field_html = ', '.join([f'<code>{field}</code>' for field in field_list[:10]])
            if len(field_list) > 10:
                field_html += f' <em>(and {len(field_list) - 10} more)</em>'
            html += f'<tr><td class="label">Field Names:</td><td>{field_html}</td></tr>'

        # Data Types
        data_types = dataset_info.get("data_types", "")
        if data_types:
            type_list = [t.strip() for t in data_types.split(',')]
            type_html = ', '.join([f'<code>{dtype}</code>' for dtype in type_list])
            html += f'<tr><td class="label">Data Types:</td><td>{type_html}</td></tr>'

        html += '</table>'

        # Use Case Suggestions
        use_cases = dataset_info.get("use_cases", "")
        if use_cases:
            html += '<h3>Use Case Suggestions</h3>'
            use_case_list = [u.strip() for u in use_cases.split(',')]
            html += '<ul class="use-cases-list">'
            for use_case in use_case_list:
                html += f'<li>{use_case}</li>'
            html += '</ul>'

        # Keywords and Tags
        html += '<h3>Keywords and Tags</h3>'
        html += '<table class="info-table">'

        # Keywords
        keywords = dataset_info.get("keywords", "")
        if keywords:
            if isinstance(keywords, str) and keywords.startswith('['):
                # Handle JSON string
                try:
                    import json
                    keyword_list = json.loads(keywords)
                    keyword_html = ', '.join([f'<span class="keyword">{kw}</span>' for kw in keyword_list[:15]])
                except:
                    keyword_html = keywords
            else:
                keyword_list = [k.strip() for k in str(keywords).split(',')]
                keyword_html = ', '.join([f'<span class="keyword">{kw}</span>' for kw in keyword_list[:15]])
            html += f'<tr><td class="label">Keywords:</td><td>{keyword_html}</td></tr>'

        # Tags
        tags = dataset_info.get("tags", "")
        if tags:
            if isinstance(tags, list):
                tag_html = ', '.join([f'<span class="tag">{tag}</span>' for tag in tags])
            else:
                tag_list = [t.strip() for t in str(tags).split(',')]
                tag_html = ', '.join([f'<span class="tag">{tag}</span>' for tag in tag_list])
            html += f'<tr><td class="label">Tags:</td><td>{tag_html}</td></tr>'

        html += '</table>'

        # Dates
        html += '<h3>Metadata</h3>'
        html += '<table class="info-table">'

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
                html += f'<li><div class="recommendation-item"><i class="icon">&#128161;</i> {rec}</div></li>'
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
                html += f'<li><div class="issue-item"><i class="icon">&#9888;</i> {issue}</div></li>'
            html += '</ul>'
        else:
            html += '<p class="empty-message">No issues identified.</p>'

        html += '</div>'

        return html

    def _format_python_examples_html(self, python_examples: Dict[str, str]) -> str:
        """Format Python examples as HTML."""
        html = '<div class="section">'
        html += '<h2>Python Code Examples</h2>'

        if python_examples:
            for example_type, code in python_examples.items():
                title = example_type.replace('_', ' ').title()
                html += f'<h3>{title}</h3>'
                html += f'<pre class="code-block"><code class="python">{code}</code></pre>'
        else:
            html += '<p class="empty-message">No Python examples available.</p>'

        html += '</div>'
        return html

    def _format_visualizations_html(self, visualizations: Dict[str, Any]) -> str:
        """Format visualizations information as HTML."""
        html = '<div class="section">'
        html += '<h2>Data Visualizations</h2>'

        if visualizations:
            # Available visualizations
            if 'available' in visualizations:
                html += '<h3>Available Visualizations</h3>'
                html += '<ul class="visualization-list">'
                for viz in visualizations['available']:
                    html += f'<li>{viz}</li>'
                html += '</ul>'

            # Description
            if 'description' in visualizations:
                html += '<h3>Visualization Overview</h3>'
                html += f'<p>{visualizations["description"]}</p>'

            # Recommended visualizations
            if 'recommended' in visualizations:
                html += '<h3>Recommended Visualizations</h3>'
                html += '<ul class="recommendations-list">'
                for rec in visualizations['recommended']:
                    html += f'<li><i class="icon">📊</i> {rec}</li>'
                html += '</ul>'
        else:
            html += '<p class="empty-message">No visualization information available.</p>'

        html += '</div>'
        return html

    def _generate_python_examples(self, dataset: Dataset) -> Dict[str, str]:
        """Generate dynamic Python code examples tailored to the specific dataset."""
        examples = {}

        # Get dataset-specific information
        field_names = dataset.field_names.split(',') if dataset.field_names else []
        field_names = [f.strip() for f in field_names]
        data_types = dataset.data_types.split(',') if dataset.data_types else []
        data_types = [t.strip() for t in data_types]

        # Create field-type mapping
        field_type_map = {}
        for i, field in enumerate(field_names):
            if i < len(data_types):
                field_type_map[field] = data_types[i]

        # Identify numeric and categorical fields
        numeric_fields = [f for f, t in field_type_map.items() if 'int' in t.lower() or 'float' in t.lower()]
        categorical_fields = [f for f, t in field_type_map.items() if 'object' in t.lower() or 'string' in t.lower()]
        datetime_fields = [f for f, t in field_type_map.items() if 'datetime' in t.lower() or 'date' in t.lower()]
        boolean_fields = [f for f, t in field_type_map.items() if 'bool' in t.lower()]

        # Generate dataset-specific filename
        filename = dataset.title.lower().replace(" ", "_").replace("-", "_")
        filename = ''.join(c for c in filename if c.isalnum() or c == '_')

        # Basic data loading example (format-specific)
        if dataset.format and dataset.format.lower() == 'csv':
            examples['loading'] = f'''# Load the {dataset.title} dataset
import pandas as pd
import numpy as np

# Load data from CSV file
df = pd.read_csv('{filename}.csv')

# Display basic information about the dataset
print(f"Dataset: {dataset.title}")
print(f"Shape: {{df.shape}} ({dataset.record_count:,} rows, {dataset.field_count} columns)")
print("\\nColumn names and types:")
print(df.dtypes)
print("\\nFirst 5 rows:")
print(df.head())
print("\\nDataset summary:")
print(df.info())'''
        elif dataset.format and dataset.format.lower() == 'json':
            examples['loading'] = f'''# Load the {dataset.title} dataset
import pandas as pd
import json

# Load data from JSON file
with open('{filename}.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Display basic information
print(f"Dataset: {dataset.title}")
print(f"Shape: {{df.shape}} ({dataset.record_count:,} rows, {dataset.field_count} columns)")
print("\\nDataset info:")
print(df.info())'''
        else:
            examples['loading'] = f'''# Load the {dataset.title} dataset
import pandas as pd

# Load data (adjust file path and format as needed)
df = pd.read_csv('{filename}.csv')  # or .json, .xlsx, etc.

# Display basic information
print(f"Dataset: {dataset.title}")
print(f"Shape: {{df.shape}}")
print("\\nDataset info:")
print(df.info())'''

        # Dynamic data exploration based on actual fields
        if field_names:
            exploration_code = f'''# Data exploration for {dataset.title}
import matplotlib.pyplot as plt
import seaborn as sns

# Basic statistics for the entire dataset
print("Dataset Overview:")
print(f"Total records: {dataset.record_count:,}")
print(f"Total fields: {dataset.field_count}")
print("\\nDataset statistics:")
print(df.describe())

# Check for missing values
print("\\nMissing values analysis:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])
if missing_data.sum() == 0:
    print("✅ No missing values found!")

# Data quality check
print("\\nData quality overview:")
print(f"Duplicate rows: {{df.duplicated().sum()}}")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")'''

            # Add field-specific analysis
            if numeric_fields:
                exploration_code += f'''

# Numeric fields analysis
numeric_fields = {numeric_fields}
print("\\n📊 Numeric Fields Analysis:")
for field in numeric_fields:
    print(f"\\n{field}:")
    print(f"  Range: {{df[field].min()}} to {{df[field].max()}}")
    print(f"  Mean: {{df[field].mean():.2f}}")
    print(f"  Std: {{df[field].std():.2f}}")
    print(f"  Unique values: {{df[field].nunique()}}")'''

            if categorical_fields:
                exploration_code += f'''

# Categorical fields analysis
categorical_fields = {categorical_fields}
print("\\n📝 Categorical Fields Analysis:")
for field in categorical_fields:
    print(f"\\n{field}:")
    print(f"  Unique values: {{df[field].nunique()}}")
    print(f"  Most common values:")
    print(df[field].value_counts().head(3))'''

            if datetime_fields:
                exploration_code += f'''

# DateTime fields analysis
datetime_fields = {datetime_fields}
print("\\n📅 DateTime Fields Analysis:")
for field in datetime_fields:
    df[field] = pd.to_datetime(df[field])
    print(f"\\n{field}:")
    print(f"  Date range: {{df[field].min()}} to {{df[field].max()}}")
    print(f"  Time span: {{(df[field].max() - df[field].min()).days}} days")'''

            if boolean_fields:
                exploration_code += f'''

# Boolean fields analysis
boolean_fields = {boolean_fields}
print("\\n✅ Boolean Fields Analysis:")
for field in boolean_fields:
    print(f"\\n{field}:")
    print(df[field].value_counts())
    print(f"  True percentage: {{(df[field].sum() / len(df) * 100):.1f}}%")'''

            examples['exploration'] = exploration_code

        # Dynamic visualization based on field types
        if field_names:
            viz_code = f'''# Dynamic visualizations for {dataset.title}
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 12)

# Create comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 15))
fig.suptitle('{dataset.title} - Data Visualization Dashboard', fontsize=16, fontweight='bold')

plot_count = 0'''

            # Add numeric field visualizations
            if numeric_fields:
                viz_code += f'''

# Numeric fields visualizations
numeric_fields = {numeric_fields}
if len(numeric_fields) > 0:
    # Distribution plots for numeric fields
    for i, field in enumerate(numeric_fields[:3]):  # Show first 3 numeric fields
        plot_count += 1
        plt.subplot(3, 3, plot_count)
        df[field].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {{field}}')
        plt.xlabel(field)
        plt.ylabel('Frequency')

        # Add statistics text
        mean_val = df[field].mean()
        std_val = df[field].std()
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {{mean_val:.2f}}')
        plt.legend()

    # Box plots for numeric fields
    if len(numeric_fields) >= 2:
        plot_count += 1
        plt.subplot(3, 3, plot_count)
        df[numeric_fields[:4]].boxplot()
        plt.title('Box Plots - Numeric Fields Comparison')
        plt.xticks(rotation=45)

    # Correlation heatmap
    if len(numeric_fields) > 1:
        plot_count += 1
        plt.subplot(3, 3, plot_count)
        correlation_matrix = df[numeric_fields].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Heatmap - Numeric Fields')

    # Scatter plot for top 2 numeric fields
    if len(numeric_fields) >= 2:
        plot_count += 1
        plt.subplot(3, 3, plot_count)
        x_field, y_field = numeric_fields[0], numeric_fields[1]
        plt.scatter(df[x_field], df[y_field], alpha=0.6, color='green')
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.title(f'{{x_field}} vs {{y_field}}')

        # Add trend line
        z = np.polyfit(df[x_field].dropna(), df[y_field].dropna(), 1)
        p = np.poly1d(z)
        plt.plot(df[x_field], p(df[x_field]), "r--", alpha=0.8)'''

            # Add categorical field visualizations
            if categorical_fields:
                viz_code += f'''

# Categorical fields visualizations
categorical_fields = {categorical_fields}
if len(categorical_fields) > 0:
    # Bar plots for categorical fields
    for i, field in enumerate(categorical_fields[:2]):  # Show first 2 categorical fields
        plot_count += 1
        plt.subplot(3, 3, plot_count)
        top_values = df[field].value_counts().head(10)
        top_values.plot(kind='bar', color='lightcoral')
        plt.title(f'Top Values in {{field}}')
        plt.xlabel(field)
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for i, v in enumerate(top_values.values):
            plt.text(i, v + 0.01 * max(top_values.values), str(v),
                    ha='center', va='bottom')

    # Pie chart for a categorical field
    if len(categorical_fields) > 0:
        plot_count += 1
        plt.subplot(3, 3, plot_count)
        field = categorical_fields[0]
        top_categories = df[field].value_counts().head(5)
        plt.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
        plt.title(f'Distribution of {{field}} (Top 5)')'''

            # Add time series visualization if datetime fields exist
            if datetime_fields:
                viz_code += f'''

# Time series visualizations
datetime_fields = {datetime_fields}
if len(datetime_fields) > 0:
    # Convert datetime fields
    for field in datetime_fields:
        df[field] = pd.to_datetime(df[field])

    # Time series plot
    plot_count += 1
    plt.subplot(3, 3, plot_count)
    date_field = datetime_fields[0]

    # Group by date and count records
    daily_counts = df.groupby(df[date_field].dt.date).size()
    daily_counts.plot(kind='line', color='purple')
    plt.title(f'Records Over Time ({{date_field}})')
    plt.xlabel('Date')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45)'''

            viz_code += '''

plt.tight_layout()
plt.show()

# Additional specific visualizations
print("\\n📊 Generating additional insights...")

# Summary statistics visualization
plt.figure(figsize=(12, 8))
df.describe().T.plot(kind='bar', y=['mean', 'std'],
                    title='Mean and Standard Deviation Comparison')
plt.xticks(rotation=45)
plt.ylabel('Value')
plt.legend(['Mean', 'Standard Deviation'])
plt.tight_layout()
plt.show()'''

            examples['visualization'] = viz_code

        # Dynamic machine learning code based on dataset characteristics
        if field_names and len(field_names) > 1:
            # Determine the most likely ML approach based on dataset characteristics
            ml_approach = self._determine_ml_approach(dataset, numeric_fields, categorical_fields)

            ml_code = f'''# Machine Learning Analysis for {dataset.title}
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

print("🤖 Machine Learning Analysis for {dataset.title}")
print("=" * 50)

# Data preprocessing
df_ml = df.copy()

# Handle missing values
print("\\n1. Data Preprocessing:")
print(f"Original shape: {{df_ml.shape}}")

# Remove rows with too many missing values
missing_threshold = 0.5  # Remove rows with >50% missing values
df_ml = df_ml.dropna(thresh=int(missing_threshold * len(df_ml.columns)))
print(f"After removing sparse rows: {{df_ml.shape}}")

# Fill remaining missing values
numeric_columns = {numeric_fields}
categorical_columns = {categorical_fields}

for col in numeric_columns:
    if col in df_ml.columns:
        df_ml[col].fillna(df_ml[col].median(), inplace=True)

for col in categorical_columns:
    if col in df_ml.columns:
        df_ml[col].fillna(df_ml[col].mode()[0] if not df_ml[col].mode().empty else 'Unknown', inplace=True)

print("✅ Missing values handled")'''

            # Add domain-specific ML approach
            if ml_approach == 'classification':
                ml_code += f'''

# 2. Classification Analysis
print("\\n2. Setting up Classification Problem:")

# For this example, we'll predict {categorical_fields[0] if categorical_fields else 'a categorical variable'}
# You should replace this with your actual target variable
target_column = '{categorical_fields[0] if categorical_fields else 'target'}'

if target_column in df_ml.columns:
    # Prepare features and target
    X = df_ml.drop(columns=[target_column])
    y = df_ml[target_column]

    # Create preprocessing pipeline
    numeric_features = [col for col in numeric_columns if col in X.columns]
    categorical_features = [col for col in categorical_columns if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])

    # Create ML pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set: {{X_train.shape}}")
    print(f"Test set: {{X_test.shape}}")
    print(f"Target classes: {{y.nunique()}}")

    # Train the model
    print("\\n3. Training Random Forest Classifier...")
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\\n4. Model Performance:")
    print(f"Accuracy: {{accuracy:.3f}}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"\\nCross-validation scores: {{cv_scores}}")
    print(f"Mean CV accuracy: {{cv_scores.mean():.3f}} (+/- {{cv_scores.std() * 2:.3f}})")

    # Feature importance
    feature_names = (numeric_features +
                    list(pipeline.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(categorical_features)))

    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({{
        'feature': feature_names,
        'importance': importances
    }}).sort_values('importance', ascending=False)

    print("\\n5. Top 10 Most Important Features:")
    print(feature_importance_df.head(10))
else:
    print(f"Target column '{{target_column}}' not found. Please specify the correct target variable.")'''

            elif ml_approach == 'regression':
                ml_code += f'''

# 2. Regression Analysis
print("\\n2. Setting up Regression Problem:")

# For this example, we'll predict {numeric_fields[0] if numeric_fields else 'a numeric variable'}
# You should replace this with your actual target variable
target_column = '{numeric_fields[0] if numeric_fields else 'target'}'

if target_column in df_ml.columns:
    # Prepare features and target
    X = df_ml.drop(columns=[target_column])
    y = df_ml[target_column]

    # Create preprocessing pipeline
    numeric_features = [col for col in numeric_columns if col in X.columns]
    categorical_features = [col for col in categorical_columns if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])

    # Create ML pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {{X_train.shape}}")
    print(f"Test set: {{X_test.shape}}")
    print(f"Target range: {{y.min():.2f}} to {{y.max():.2f}}")

    # Train the model
    print("\\n3. Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\\n4. Model Performance:")
    print(f"R² Score: {{r2:.3f}}")
    print(f"RMSE: {{rmse:.3f}}")
    print(f"Mean Absolute Error: {{np.mean(np.abs(y_test - y_pred)):.3f}}")

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    print(f"\\nCross-validation R² scores: {{cv_scores}}")
    print(f"Mean CV R²: {{cv_scores.mean():.3f}} (+/- {{cv_scores.std() * 2:.3f}})")
else:
    print(f"Target column '{{target_column}}' not found. Please specify the correct target variable.")'''

            else:  # clustering or general analysis
                ml_code += f'''

# 2. Unsupervised Learning - Clustering Analysis
print("\\n2. Clustering Analysis:")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Prepare data for clustering (numeric features only)
numeric_features = {numeric_fields}
if len(numeric_features) >= 2:
    X_cluster = df_ml[numeric_features].select_dtypes(include=[np.number])

    # Handle any remaining missing values
    X_cluster = X_cluster.fillna(X_cluster.mean())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Determine optimal number of clusters using elbow method
    inertias = []
    k_range = range(2, min(11, len(X_cluster)//10 + 2))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Apply K-means clustering
    optimal_k = 3  # You can adjust this based on elbow method results
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    print(f"Applied K-means clustering with {{optimal_k}} clusters")
    print(f"Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {{cluster}}: {{count}} samples ({{count/len(cluster_labels)*100:.1f}}%)")

    # PCA for visualization
    if X_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        print(f"\\nPCA explained variance ratio: {{pca.explained_variance_ratio_}}")
        print(f"Total variance explained: {{pca.explained_variance_ratio_.sum():.3f}}")

        # Add cluster labels to original dataframe
        df_ml['cluster'] = cluster_labels

        print("\\n✅ Clustering analysis complete. Cluster labels added to dataframe.")
    else:
        print("Not enough numeric features for meaningful clustering analysis.")
else:
    print("Insufficient numeric features for clustering analysis.")'''

            ml_code += '''

# 6. Next Steps and Recommendations
print("\\n" + "="*50)
print("🎯 RECOMMENDATIONS FOR FURTHER ANALYSIS:")
print("="*50)

recommendations = [
    "1. Feature Engineering: Create new features from existing ones",
    "2. Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV",
    "3. Try Different Algorithms: Compare performance across multiple models",
    "4. Handle Class Imbalance: Use SMOTE or class weights if applicable",
    "5. Feature Selection: Use SelectKBest or RFE for feature selection",
    "6. Ensemble Methods: Combine multiple models for better performance",
    "7. Cross-validation: Use more sophisticated CV strategies",
    "8. Domain-specific Features: Incorporate domain knowledge"
]

for rec in recommendations:
    print(rec)

print("\\n✅ Machine Learning analysis complete!")'''

            examples['ml_analysis'] = ml_code

        # Add domain-specific analysis
        domain_analysis = self._generate_domain_specific_analysis(dataset, field_names, numeric_fields, categorical_fields)
        if domain_analysis:
            examples['domain_analysis'] = domain_analysis

        return examples

    def _determine_ml_approach(self, dataset: Dataset, numeric_fields: list, categorical_fields: list) -> str:
        """Determine the most appropriate ML approach based on dataset characteristics."""

        # Analyze dataset category and title for domain clues
        title_lower = dataset.title.lower() if dataset.title else ""
        category_lower = dataset.category.lower() if dataset.category else ""

        # Check for classification indicators
        classification_keywords = ['class', 'category', 'type', 'label', 'status', 'outcome', 'result', 'grade']
        regression_keywords = ['price', 'cost', 'amount', 'score', 'rating', 'value', 'income', 'salary', 'revenue']

        # Check if there are good categorical targets for classification
        if categorical_fields and any(keyword in title_lower or keyword in category_lower for keyword in classification_keywords):
            return 'classification'

        # Check if there are good numeric targets for regression
        if numeric_fields and any(keyword in title_lower or keyword in category_lower for keyword in regression_keywords):
            return 'regression'

        # Default based on field types
        if len(categorical_fields) > len(numeric_fields):
            return 'classification'
        elif len(numeric_fields) > len(categorical_fields):
            return 'regression'
        else:
            return 'clustering'

    def _generate_domain_specific_analysis(self, dataset: Dataset, field_names: list, numeric_fields: list, categorical_fields: list) -> str:
        """Generate domain-specific analysis code based on dataset category."""

        if not dataset.category:
            return ""

        category_lower = dataset.category.lower()
        title_lower = dataset.title.lower() if dataset.title else ""

        # Education domain analysis
        if 'education' in category_lower or 'student' in title_lower or 'course' in title_lower:
            return f'''# Domain-Specific Analysis: Education Data
print("\\n🎓 EDUCATION DOMAIN ANALYSIS")
print("=" * 40)

# Education-specific metrics
education_metrics = {{}}

# Performance analysis
performance_fields = [col for col in df.columns if any(keyword in col.lower()
                     for keyword in ['score', 'grade', 'performance', 'result', 'mark'])]

if performance_fields:
    print("\\n📊 Academic Performance Analysis:")
    for field in performance_fields:
        if df[field].dtype in ['int64', 'float64']:
            education_metrics[field] = {{
                'mean': df[field].mean(),
                'median': df[field].median(),
                'std': df[field].std(),
                'pass_rate': (df[field] >= df[field].quantile(0.6)).mean() * 100
            }}
            print(f"{{field}}:")
            print(f"  Average: {{education_metrics[field]['mean']:.2f}}")
            print(f"  Pass Rate (>60th percentile): {{education_metrics[field]['pass_rate']:.1f}}%")

# Enrollment analysis
enrollment_fields = [col for col in df.columns if any(keyword in col.lower()
                    for keyword in ['enrollment', 'student', 'participant', 'learner'])]

if enrollment_fields:
    print("\\n👥 Enrollment Analysis:")
    for field in enrollment_fields:
        if df[field].dtype in ['int64', 'float64']:
            print(f"{{field}}: Total = {{df[field].sum():,}}, Average = {{df[field].mean():.0f}}")

# Course/Subject analysis
subject_fields = [col for col in df.columns if any(keyword in col.lower()
                 for keyword in ['subject', 'course', 'topic', 'category'])]

if subject_fields:
    print("\\n📚 Subject/Course Distribution:")
    for field in subject_fields:
        if df[field].dtype == 'object':
            top_subjects = df[field].value_counts().head(5)
            print(f"\\nTop 5 {{field}}:")
            for subject, count in top_subjects.items():
                print(f"  {{subject}}: {{count}} ({{count/len(df)*100:.1f}}%)")

print("\\n✅ Education domain analysis complete!")'''

        # Business/Finance domain analysis
        elif any(keyword in category_lower for keyword in ['business', 'finance', 'market', 'sales']):
            return f'''# Domain-Specific Analysis: Business/Finance Data
print("\\n💼 BUSINESS/FINANCE DOMAIN ANALYSIS")
print("=" * 45)

# Financial metrics
financial_metrics = {{}}

# Revenue/Sales analysis
revenue_fields = [col for col in df.columns if any(keyword in col.lower()
                 for keyword in ['revenue', 'sales', 'income', 'profit', 'price', 'cost'])]

if revenue_fields:
    print("\\n💰 Financial Performance Analysis:")
    for field in revenue_fields:
        if df[field].dtype in ['int64', 'float64']:
            financial_metrics[field] = {{
                'total': df[field].sum(),
                'average': df[field].mean(),
                'median': df[field].median(),
                'growth_potential': df[field].quantile(0.9) / df[field].quantile(0.1)
            }}
            print(f"{{field}}:")
            print(f"  Total: ${{financial_metrics[field]['total']:,.2f}}")
            print(f"  Average: ${{financial_metrics[field]['average']:,.2f}}")
            print(f"  Median: ${{financial_metrics[field]['median']:,.2f}}")

# Customer analysis
customer_fields = [col for col in df.columns if any(keyword in col.lower()
                  for keyword in ['customer', 'client', 'user', 'subscriber'])]

if customer_fields:
    print("\\n👥 Customer Analysis:")
    for field in customer_fields:
        if df[field].dtype in ['int64', 'float64']:
            print(f"{{field}}: {{df[field].sum():,}} total")
        elif df[field].dtype == 'object':
            print(f"{{field}}: {{df[field].nunique()}} unique values")

# Market analysis
market_fields = [col for col in df.columns if any(keyword in col.lower()
                for keyword in ['market', 'segment', 'category', 'region'])]

if market_fields:
    print("\\n🌍 Market Segmentation:")
    for field in market_fields:
        if df[field].dtype == 'object':
            segments = df[field].value_counts()
            print(f"\\n{{field}} distribution:")
            for segment, count in segments.head(5).items():
                print(f"  {{segment}}: {{count}} ({{count/len(df)*100:.1f}}%)")

print("\\n✅ Business/Finance domain analysis complete!")'''

        # Agriculture domain analysis
        elif 'farm' in category_lower or 'agriculture' in category_lower or 'crop' in title_lower:
            return f'''# Domain-Specific Analysis: Agriculture Data
print("\\n🌾 AGRICULTURE DOMAIN ANALYSIS")
print("=" * 40)

# Agricultural metrics
agri_metrics = {{}}

# Crop yield analysis
yield_fields = [col for col in df.columns if any(keyword in col.lower()
               for keyword in ['yield', 'production', 'harvest', 'output'])]

if yield_fields:
    print("\\n🌱 Crop Yield Analysis:")
    for field in yield_fields:
        if df[field].dtype in ['int64', 'float64']:
            agri_metrics[field] = {{
                'total_production': df[field].sum(),
                'average_yield': df[field].mean(),
                'yield_variability': df[field].std() / df[field].mean()
            }}
            print(f"{{field}}:")
            print(f"  Total Production: {{agri_metrics[field]['total_production']:,.0f}}")
            print(f"  Average Yield: {{agri_metrics[field]['average_yield']:.2f}}")
            print(f"  Yield Variability: {{agri_metrics[field]['yield_variability']:.2f}}")

# Area analysis
area_fields = [col for col in df.columns if any(keyword in col.lower()
              for keyword in ['area', 'hectare', 'acre', 'land'])]

if area_fields:
    print("\\n🗺️ Land Area Analysis:")
    for field in area_fields:
        if df[field].dtype in ['int64', 'float64']:
            print(f"{{field}}: Total = {{df[field].sum():,.0f}}, Average = {{df[field].mean():.2f}}")

# Crop type analysis
crop_fields = [col for col in df.columns if any(keyword in col.lower()
              for keyword in ['crop', 'plant', 'variety', 'species'])]

if crop_fields:
    print("\\n🌾 Crop Type Distribution:")
    for field in crop_fields:
        if df[field].dtype == 'object':
            crops = df[field].value_counts()
            print(f"\\nTop crops in {{field}}:")
            for crop, count in crops.head(5).items():
                print(f"  {{crop}}: {{count}} records")

print("\\n✅ Agriculture domain analysis complete!")'''

        # Health domain analysis
        elif 'health' in category_lower or 'medical' in category_lower:
            return f'''# Domain-Specific Analysis: Health/Medical Data
print("\\n🏥 HEALTH/MEDICAL DOMAIN ANALYSIS")
print("=" * 42)

# Health metrics
health_metrics = {{}}

# Patient analysis
patient_fields = [col for col in df.columns if any(keyword in col.lower()
                 for keyword in ['patient', 'age', 'gender', 'diagnosis'])]

if patient_fields:
    print("\\n👤 Patient Demographics:")
    for field in patient_fields:
        if 'age' in field.lower() and df[field].dtype in ['int64', 'float64']:
            age_groups = pd.cut(df[field], bins=[0, 18, 35, 50, 65, 100],
                              labels=['<18', '18-35', '36-50', '51-65', '65+'])
            print(f"Age distribution:")
            print(age_groups.value_counts())
        elif df[field].dtype == 'object':
            print(f"\\n{{field}} distribution:")
            print(df[field].value_counts().head())

print("\\n✅ Health domain analysis complete!")'''

        return ""

    def _get_visualizations_data(self, dataset: Dataset) -> Dict[str, Any]:
        """Get visualization data for the dataset."""
        visualizations = {}

        # Check if dataset has visualizations stored
        if hasattr(dataset, 'visualizations') and dataset.visualizations:
            try:
                import json
                visualizations = json.loads(dataset.visualizations)
            except:
                visualizations = {}

        # Add basic visualization descriptions
        if not visualizations:
            visualizations = {
                'available': ['Distribution plots', 'Box plots', 'Correlation heatmap', 'Scatter plots'],
                'description': 'Various statistical visualizations can be generated from this dataset to explore data patterns and relationships.',
                'recommended': [
                    'Histogram for numerical distributions',
                    'Bar charts for categorical data',
                    'Correlation matrix for feature relationships',
                    'Time series plots if temporal data exists'
                ]
            }

        return visualizations

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

        .code-block {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
        }

        .code-block code {
            background: none;
            padding: 0;
            border: none;
            color: #1f2937;
        }

        .visualization-list, .use-cases-list {
            list-style-type: none;
            padding-left: 0;
        }

        .visualization-list li, .use-cases-list li {
            padding: 8px 12px;
            margin-bottom: 5px;
            background-color: #f0f9ff;
            border-left: 3px solid #0ea5e9;
            border-radius: 4px;
        }

        .description-content {
            background-color: #f8fafc;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3b82f6;
            margin: 10px 0;
            line-height: 1.6;
        }

        .keyword, .tag {
            display: inline-block;
            background-color: #e0e7ff;
            color: #3730a3;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }

        .tag {
            background-color: #fef3c7;
            color: #92400e;
        }

        code {
            background-color: #f1f5f9;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #1e293b;
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

    {{PYTHON_EXAMPLES}}

    {{VISUALIZATIONS}}

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

        with open(report_path, 'w', encoding='utf-8') as file:
            file.write(html_content)

        # Return relative path for URL
        return f"/static/reports/{report_filename}"

    def _generate_pdf_report(self, report_data: Dict[str, Any], metadata: Dict[str, Any]) -> bytes:
        """
        Generate PDF report with multiple fallback options.

        Args:
            report_data: Report data
            metadata: Report metadata

        Returns:
            PDF content as bytes or HTML as fallback
        """
        # Try multiple PDF generation methods
        pdf_methods = [
            self._try_weasyprint_pdf,
            self._try_pdfkit_pdf,
            self._try_reportlab_pdf
        ]

        html_content = self._generate_html_report(report_data, metadata)

        for method in pdf_methods:
            try:
                pdf_content = method(html_content, report_data, metadata)
                if pdf_content:
                    logger.info(f"PDF generated successfully using {method.__name__}")
                    return pdf_content
            except Exception as e:
                logger.warning(f"PDF method {method.__name__} failed: {e}")
                continue

        # Final fallback to HTML
        logger.warning("All PDF generation methods failed, falling back to HTML")
        return html_content.encode('utf-8')

    def _try_weasyprint_pdf(self, html_content: str, report_data: Dict, metadata: Dict) -> bytes:
        """Try generating PDF using WeasyPrint."""
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration

            # Create PDF with WeasyPrint
            font_config = FontConfiguration()
            html_doc = HTML(string=html_content)
            pdf_bytes = html_doc.write_pdf(font_config=font_config)
            return pdf_bytes
        except ImportError:
            raise Exception("WeasyPrint not available")

    def _try_pdfkit_pdf(self, html_content: str, report_data: Dict, metadata: Dict) -> bytes:
        """Try generating PDF using pdfkit."""
        try:
            import pdfkit

            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None,
                'disable-smart-shrinking': '',
                'print-media-type': ''
            }

            pdf_content = pdfkit.from_string(html_content, False, options=options)
            return pdf_content
        except ImportError:
            raise Exception("pdfkit not available")

    def _try_reportlab_pdf(self, html_content: str, report_data: Dict, metadata: Dict) -> bytes:
        """Try generating PDF using ReportLab with simplified content."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from io import BytesIO

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph(f"Dataset Health Report: {report_data['dataset']['title']}", title_style))
            story.append(Spacer(1, 20))

            # Dataset Information
            story.append(Paragraph("Dataset Information", styles['Heading2']))
            dataset_info = report_data['dataset']

            data = [
                ['Title', dataset_info.get('title', 'N/A')],
                ['Source', dataset_info.get('source', 'N/A')],
                ['Category', dataset_info.get('category', 'N/A')],
                ['Format', dataset_info.get('format', 'N/A')],
                ['Records', f"{dataset_info.get('record_count', 0):,}"],
                ['Fields', str(dataset_info.get('field_count', 'N/A'))],
            ]

            table = Table(data, colWidths=[2*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))

            # Description
            if dataset_info.get('description'):
                story.append(Paragraph("Description", styles['Heading2']))
                story.append(Paragraph(dataset_info['description'], styles['Normal']))
                story.append(Spacer(1, 20))

            # Use Cases
            if dataset_info.get('use_cases'):
                story.append(Paragraph("Use Case Suggestions", styles['Heading2']))
                use_cases = dataset_info['use_cases'].split(',')
                for i, use_case in enumerate(use_cases[:6], 1):
                    story.append(Paragraph(f"{i}. {use_case.strip()}", styles['Normal']))
                story.append(Spacer(1, 20))

            # Quality Metrics
            story.append(Paragraph("Quality Assessment", styles['Heading2']))
            quality_metrics = report_data['quality']
            quality_score = quality_metrics.get('quality_score', 0)
            story.append(Paragraph(f"Overall Quality Score: {int(quality_score)}/100", styles['Normal']))
            story.append(Spacer(1, 20))

            # Build PDF
            doc.build(story)
            pdf_content = buffer.getvalue()
            buffer.close()
            return pdf_content

        except ImportError:
            raise Exception("ReportLab not available")

    def _save_pdf_report(self, content: bytes, report_id: str) -> str:
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
            file.write(content)

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