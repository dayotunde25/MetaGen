"""
Quality Visualization Service - Generate visualizations for dataset quality metrics
"""
import json
import os
import base64
import io
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm as cm
from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality


class QualityVisualization:
    """Service for generating visualizations of dataset quality metrics"""

    def __init__(self, static_folder: str):
        """
        Initialize visualization service
        
        Args:
            static_folder: Path to static folder for saving generated images
        """
        self.static_folder = static_folder
        self.img_folder = os.path.join(static_folder, 'img', 'quality')
        
        # Create folder if it doesn't exist
        os.makedirs(self.img_folder, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Define colors for FAIR principles
        self.fair_colors = {
            'findable': '#4CAF50',      # Green
            'accessible': '#2196F3',    # Blue
            'interoperable': '#FFC107', # Amber/Yellow
            'reusable': '#9C27B0'       # Purple
        }
        
        # Define colors for quality dimensions
        self.dimension_colors = {
            'completeness': '#3949AB',  # Indigo
            'consistency': '#00ACC1',   # Cyan
            'accuracy': '#43A047',      # Light Green
            'timeliness': '#FB8C00',    # Orange
            'fairness': '#EC407A'       # Pink
        }

    def generate_fair_radar_chart(self, metadata_quality: MetadataQuality, dataset_id: int) -> str:
        """
        Generate a radar chart for FAIR principles scores
        
        Args:
            metadata_quality: The MetadataQuality object
            dataset_id: The dataset ID for file naming
            
        Returns:
            Path to the generated image file
        """
        # Create figure and polar axis
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, polar=True)
        
        # Define the categories and scores
        categories = ['Findable', 'Accessible', 'Interoperable', 'Reusable']
        scores = [
            metadata_quality.findable_score,
            metadata_quality.accessible_score,
            metadata_quality.interoperable_score,
            metadata_quality.reusable_score
        ]
        
        # Number of categories
        N = len(categories)
        
        # We need the first value to repeat to close the circle
        values = scores + [scores[0]]
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Add first angle to close the loop
        
        # Draw the chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
                color='#3949AB', label='FAIR Scores')
        ax.fill(angles, values, alpha=0.3, color='#3949AB')
        
        # Fix axis to go in the right order and start at top
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Set y-axis limit
        ax.set_ylim(0, 100)
        
        # Draw a circle at each percentage
        for percentage in [20, 40, 60, 80, 100]:
            ax.plot(angles, [percentage] * len(angles), '--', color='gray', alpha=0.3, linewidth=0.5)
        
        # Add percentage labels along y-axis
        # These are tricky in polar plots, so we'll add them manually at key points
        plt.annotate('20%', xy=(0, 20), xytext=(10, 20), color='gray', size=8)
        plt.annotate('40%', xy=(0, 40), xytext=(10, 40), color='gray', size=8)
        plt.annotate('60%', xy=(0, 60), xytext=(10, 60), color='gray', size=8)
        plt.annotate('80%', xy=(0, 80), xytext=(10, 80), color='gray', size=8)
        plt.annotate('100%', xy=(0, 100), xytext=(10, 100), color='gray', size=8)
        
        # Add score annotations near each point
        for i, score in enumerate(scores):
            angle = angles[i]
            x = angle
            y = score
            
            # Calculate text position based on angle
            # A bit of a simplification, but works for basic positioning
            plt.annotate(f'{int(score)}%', 
                         xy=(x, y), 
                         xytext=(1.2 * np.cos(x) * y, 1.2 * np.sin(x) * y),
                         size=11,
                         weight='bold',
                         color='#3949AB')
        
        # Add title
        plt.title('FAIR Principles Assessment', size=14, y=1.08)
        
        # Save figure
        filename = f'fair_radar_{dataset_id}.png'
        filepath = os.path.join(self.img_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Return path relative to static folder
        return os.path.join('img', 'quality', filename)

    def generate_dimension_bar_chart(self, assessment: Dict, dataset_id: int) -> str:
        """
        Generate a bar chart for quality dimensions
        
        Args:
            assessment: Quality assessment dictionary
            dataset_id: The dataset ID for file naming
            
        Returns:
            Path to the generated image file
        """
        # Create figure and axis
        fig, ax = plt.figure(figsize=(8, 5)), plt.axes()
        
        # Define dimensions and scores
        dimensions = ['Completeness', 'Consistency', 'Accuracy', 'Timeliness', 'Overall Quality']
        scores = [
            assessment.get('completeness', 0),
            assessment.get('consistency', 0),
            assessment.get('accuracy', 0),
            assessment.get('timeliness', 0),
            assessment.get('quality_score', 0)
        ]
        
        # Colors for each bar
        colors = [
            self.dimension_colors.get('completeness', '#3949AB'),
            self.dimension_colors.get('consistency', '#00ACC1'),
            self.dimension_colors.get('accuracy', '#43A047'),
            self.dimension_colors.get('timeliness', '#FB8C00'),
            '#5D4037'  # Brown for overall quality
        ]
        
        # Create horizontal bar chart
        bars = ax.barh(dimensions, scores, color=colors, height=0.6)
        
        # Add data labels to the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{int(width)}%', 
                     va='center', fontsize=10, fontweight='bold')
        
        # Customize chart
        ax.set_xlabel('Score (%)', fontsize=12)
        ax.set_xlim(0, 105)  # Allow space for labels
        plt.title('Dataset Quality Dimensions', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        filename = f'quality_dimensions_{dataset_id}.png'
        filepath = os.path.join(self.img_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Return path relative to static folder
        return os.path.join('img', 'quality', filename)

    def generate_quality_gauge(self, quality_score: float, dataset_id: int) -> str:
        """
        Generate a gauge chart for overall quality score
        
        Args:
            quality_score: The overall quality score
            dataset_id: The dataset ID for file naming
            
        Returns:
            Path to the generated image file
        """
        # Create figure
        fig = plt.figure(figsize=(8, 4))
        
        # Create an axis at a specific position to draw the gauge
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        
        # Define the gauge parameters
        theta = np.linspace(0, 180, 100) * np.pi / 180  # Convert to radians
        r = np.ones_like(theta)
        
        # Color gradient for gauge background
        cmap = cm.get_cmap('RdYlGn')  # Red-Yellow-Green colormap
        norm = plt.Normalize(0, 100)
        colors = cmap(norm(np.linspace(0, 100, len(theta))))
        
        # Draw the gauge background
        ax.scatter(theta, r, c=colors, s=300, linewidths=0, alpha=0.8)
        
        # Calculate the angle for the quality score needle
        score_angle = quality_score * np.pi / 100  # Map 0-100 to 0-π
        
        # Draw the needle
        ax.plot([0, score_angle], [0, 0.9], color='#333333', linewidth=4)
        
        # Draw the center point of the gauge
        ax.scatter(0, 0, s=200, color='#333333')
        
        # Add score text
        ax.text(0, -0.2, f'{int(quality_score)}', fontsize=24, ha='center', va='center', weight='bold')
        
        # Customize the gauge
        ax.set_theta_zero_location('S')  # Set 0 degrees at the bottom
        ax.set_theta_direction(-1)      # Go clockwise
        ax.set_rlim(0, 1)               # Set radius limit
        
        # Remove grid lines and axis labels
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add gauge labels
        plt.text(0, 1.2, 'Poor', fontsize=10, ha='center')
        plt.text(np.pi/4, 1.2, 'Fair', fontsize=10, ha='center')
        plt.text(np.pi/2, 1.2, 'Good', fontsize=10, ha='center')
        plt.text(3*np.pi/4, 1.2, 'Very Good', fontsize=10, ha='center')
        plt.text(np.pi, 1.2, 'Excellent', fontsize=10, ha='center')
        
        # Add quality categories
        if quality_score < 30:
            quality_label = 'Poor'
        elif quality_score < 50:
            quality_label = 'Fair'
        elif quality_score < 70:
            quality_label = 'Good'
        elif quality_score < 90:
            quality_label = 'Very Good'
        else:
            quality_label = 'Excellent'
            
        plt.text(0, -0.35, quality_label, fontsize=14, ha='center', va='center')
        
        # Add title
        plt.title('Overall Quality Score', fontsize=14, y=1.2)
        
        # Save figure
        filename = f'quality_gauge_{dataset_id}.png'
        filepath = os.path.join(self.img_folder, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Return path relative to static folder
        return os.path.join('img', 'quality', filename)

    def generate_comparison_chart(self, comparison_data: Dict) -> str:
        """
        Generate a comparative chart for multiple datasets
        
        Args:
            comparison_data: Dictionary with comparison results
            
        Returns:
            Path to the generated image file
        """
        # Number of datasets to compare
        n_datasets = len(comparison_data.get('datasets', []))
        
        if n_datasets == 0:
            return ""
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define dimensions to compare
        dimensions = ['Overall Quality', 'Completeness', 'Consistency', 
                     'Findable', 'Accessible', 'Interoperable', 'Reusable']
        
        # X positions for bars
        x = np.arange(len(dimensions))
        width = 0.8 / n_datasets  # Width of bars
        
        # Plot bars for each dataset
        for i, dataset in enumerate(comparison_data.get('datasets', [])):
            # Get scores for this dataset
            scores = [
                dataset.get('quality_score', 0),
                dataset.get('completeness', 0),
                dataset.get('consistency', 0),
                dataset.get('findable_score', 0),
                dataset.get('accessible_score', 0),
                dataset.get('interoperable_score', 0),
                dataset.get('reusable_score', 0)
            ]
            
            # Calculate position for this dataset's bars
            pos = x - 0.4 + (i + 0.5) * width
            
            # Pick a color from a colormap
            color = plt.cm.tab10(i)
            
            # Create bars
            bars = ax.bar(pos, scores, width, label=dataset.get('title', f'Dataset {i+1}'), color=color, alpha=0.8)
            
            # Add data labels for overall quality
            if i == 0:  # Only for the first dimension (Overall Quality)
                plt.text(pos[0], scores[0] + 2, f'{int(scores[0])}', 
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize chart
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_ylim(0, 105)  # Allow space for labels
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title('Dataset Quality Comparison', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(loc='upper right', fontsize=9)
        
        # Save figure
        filename = f'quality_comparison.png'
        filepath = os.path.join(self.img_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Return path relative to static folder
        return os.path.join('img', 'quality', filename)

    def generate_fair_compliance_chart(self, datasets: List[Dataset], fair_results: List[bool]) -> str:
        """
        Generate a FAIR compliance chart for multiple datasets
        
        Args:
            datasets: List of datasets
            fair_results: List of boolean fair compliance results
            
        Returns:
            Path to the generated image file
        """
        # Count compliant and non-compliant datasets
        compliant = sum(fair_results)
        non_compliant = len(fair_results) - compliant
        
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Pie chart of compliance status
        labels = ['FAIR Compliant', 'Not FAIR Compliant']
        sizes = [compliant, non_compliant]
        colors = ['#4CAF50', '#F44336']
        explode = (0.1, 0)  # Explode the 1st slice (compliant)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.set_title('FAIR Compliance Status', fontsize=14)
        
        # Bar chart of individual datasets
        dataset_names = [d.title if hasattr(d, 'title') else f"Dataset {i+1}" for i, d in enumerate(datasets)]
        # Limit to 10 datasets for readability
        if len(dataset_names) > 10:
            dataset_names = dataset_names[:9] + ['Others']
            fair_results = fair_results[:9] + [any(fair_results[9:])]
            
        # Shorten long names
        dataset_names = [name[:20] + '...' if len(name) > 20 else name for name in dataset_names]
        
        # Create bar colors
        bar_colors = ['#4CAF50' if result else '#F44336' for result in fair_results]
        
        y_pos = np.arange(len(dataset_names))
        ax2.barh(y_pos, [1] * len(dataset_names), color=bar_colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(dataset_names)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel('FAIR Compliance Status')
        ax2.set_xticks([])  # Hide x ticks
        ax2.set_title('Individual Dataset Compliance', fontsize=14)
        
        # Add text labels
        for i, result in enumerate(fair_results):
            text = 'Compliant' if result else 'Not Compliant'
            ax2.text(0.5, i, text, ha='center', va='center', color='white', fontweight='bold')
        
        # Save figure
        filename = f'fair_compliance_summary.png'
        filepath = os.path.join(self.img_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Return path relative to static folder
        return os.path.join('img', 'quality', filename)

    def generate_quality_trend_chart(self, dataset_id: int, quality_history: List[Dict]) -> str:
        """
        Generate a chart showing quality score trends over time
        
        Args:
            dataset_id: The dataset ID
            quality_history: List of quality assessments with timestamps
            
        Returns:
            Path to the generated image file
        """
        if not quality_history:
            return ""
            
        # Sort by timestamp
        quality_history.sort(key=lambda x: x.get('timestamp', 0))
        
        # Extract timestamps and scores
        timestamps = [entry.get('timestamp') for entry in quality_history]
        quality_scores = [entry.get('quality_score', 0) for entry in quality_history]
        fair_scores = [
            (entry.get('findable_score', 0) +
             entry.get('accessible_score', 0) +
             entry.get('interoperable_score', 0) +
             entry.get('reusable_score', 0)) / 4
            for entry in quality_history
        ]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot quality score trend
        ax.plot(timestamps, quality_scores, 'o-', color='#3949AB', linewidth=2, label='Overall Quality')
        
        # Plot FAIR score trend
        ax.plot(timestamps, fair_scores, 's-', color='#43A047', linewidth=2, label='Avg. FAIR Score')
        
        # Customize chart
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_ylim(0, 105)  # Allow space for labels
        ax.set_title(f'Quality Score Trends Over Time', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Add legend
        plt.legend(loc='lower right')
        
        # Save figure
        filename = f'quality_trend_{dataset_id}.png'
        filepath = os.path.join(self.img_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Return path relative to static folder
        return os.path.join('img', 'quality', filename)

    def generate_quality_report(self, dataset: Dataset, assessment: Dict) -> Dict:
        """
        Generate a complete quality report with multiple visualizations
        
        Args:
            dataset: The dataset
            assessment: Quality assessment dictionary
            
        Returns:
            Dictionary with paths to all generated visualizations
        """
        dataset_id = dataset.id if hasattr(dataset, 'id') else 0
        
        # Create MetadataQuality object from assessment for compatibility
        metadata_quality = MetadataQuality(
            dataset_id=dataset_id,
            quality_score=assessment.get('quality_score', 0),
            completeness=assessment.get('completeness', 0),
            consistency=assessment.get('consistency', 0),
            findable_score=assessment.get('findable_score', 0),
            accessible_score=assessment.get('accessible_score', 0),
            interoperable_score=assessment.get('interoperable_score', 0),
            reusable_score=assessment.get('reusable_score', 0),
            fair_compliant=assessment.get('fair_compliant', False),
            schema_org_compliant=assessment.get('schema_org_compliant', False)
        )
        
        # Generate all visualizations
        radar_chart = self.generate_fair_radar_chart(metadata_quality, dataset_id)
        dimension_chart = self.generate_dimension_bar_chart(assessment, dataset_id)
        gauge_chart = self.generate_quality_gauge(assessment.get('quality_score', 0), dataset_id)
        
        # Return paths to all visualizations
        return {
            'radar_chart': radar_chart,
            'dimension_chart': dimension_chart,
            'gauge_chart': gauge_chart
        }

    def generate_base64_chart(self, figure: Figure) -> str:
        """
        Generate a base64-encoded PNG image from a matplotlib figure
        
        Args:
            figure: Matplotlib figure object
            
        Returns:
            Base64-encoded string of the image
        """
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(figure)
        
        return f"data:image/png;base64,{img_str}"


# Singleton instance
_quality_visualization = None

def get_quality_visualization(static_folder: str):
    """Get singleton instance of QualityVisualization"""
    global _quality_visualization
    if _quality_visualization is None:
        _quality_visualization = QualityVisualization(static_folder)
    return _quality_visualization