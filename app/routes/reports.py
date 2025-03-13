"""
Routes for dataset health reports.
"""

import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, send_file, jsonify, send_from_directory
from flask_login import login_required, current_user

from app.models.dataset import Dataset
from app.models.metadata import MetadataQuality
from app.services.report_generator import get_report_generator

# Create blueprint
reports = Blueprint('reports', __name__)


@reports.route('/datasets/<int:dataset_id>/report')
def view_report(dataset_id):
    """
    View dataset health report.
    
    Args:
        dataset_id: ID of the dataset
    """
    dataset = Dataset.find_by_id(dataset_id)
    
    if not dataset:
        flash("Dataset not found", "danger")
        return redirect(url_for('datasets.list'))
    
    # Get metadata quality
    quality = MetadataQuality.get_by_dataset(dataset_id)
    
    if not quality:
        flash("Dataset quality assessment not available", "warning")
        return redirect(url_for('datasets.quality', dataset_id=dataset_id))
    
    # Get report generator
    templates_folder = os.path.join(current_app.root_path, 'templates', 'reports')
    report_generator = get_report_generator(templates_folder)
    
    # Generate report
    try:
        report_format = request.args.get('format', 'html')
        report = report_generator.generate_report(dataset_id, report_format)
        
        # Redirect to static report
        if report_format == 'html':
            # For HTML reports, redirect to the static file
            return redirect(report['path'])
        else:
            # For PDF reports, serve the file for download
            filename = os.path.basename(report['path'])
            directory = os.path.dirname(os.path.join(current_app.root_path, report['path'].lstrip('/')))
            return send_from_directory(
                directory,
                filename,
                as_attachment=True,
                attachment_filename=f"health_report_{dataset.title}_{filename}"
            )
    
    except Exception as e:
        current_app.logger.error(f"Error generating report: {str(e)}")
        flash(f"Error generating report: {str(e)}", "danger")
        return redirect(url_for('datasets.quality', dataset_id=dataset_id))


@reports.route('/api/datasets/<int:dataset_id>/report', methods=['POST'])
@login_required
def generate_report(dataset_id):
    """
    Generate dataset health report via API.
    
    Args:
        dataset_id: ID of the dataset
    """
    # Check dataset exists
    dataset = Dataset.find_by_id(dataset_id)
    
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Check authorization
    if dataset.user_id and dataset.user_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Unauthorized to generate report for this dataset"}), 403
    
    # Check quality assessment exists
    quality = MetadataQuality.get_by_dataset(dataset_id)
    
    if not quality:
        return jsonify({"error": "Dataset quality assessment not available"}), 400
    
    # Get report format
    report_format = request.json.get('format', 'html')
    if report_format not in ['html', 'pdf']:
        return jsonify({"error": f"Unsupported report format: {report_format}"}), 400
    
    # Get report generator
    templates_folder = os.path.join(current_app.root_path, 'templates', 'reports')
    report_generator = get_report_generator(templates_folder)
    
    # Generate report
    try:
        report = report_generator.generate_report(dataset_id, report_format)
        
        # Return report info
        return jsonify({
            "success": True,
            "report": report
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error generating report: {str(e)}")
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500


@reports.route('/api/datasets/<int:dataset_id>/report/download')
def download_report(dataset_id):
    """
    Download dataset health report.
    
    Args:
        dataset_id: ID of the dataset
    """
    # Check dataset exists
    dataset = Dataset.find_by_id(dataset_id)
    
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Get report format
    report_format = request.args.get('format', 'html')
    if report_format not in ['html', 'pdf']:
        return jsonify({"error": f"Unsupported report format: {report_format}"}), 400
    
    # Get report path from request
    report_path = request.args.get('path')
    
    if not report_path:
        # If path not provided, generate report
        templates_folder = os.path.join(current_app.root_path, 'templates', 'reports')
        report_generator = get_report_generator(templates_folder)
        
        try:
            report = report_generator.generate_report(dataset_id, report_format)
            report_path = report['path']
        except Exception as e:
            current_app.logger.error(f"Error generating report: {str(e)}")
            return jsonify({"error": f"Error generating report: {str(e)}"}), 500
    
    # Get full path
    file_path = os.path.join(current_app.root_path, report_path.lstrip('/'))
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({"error": "Report file not found"}), 404
    
    # Determine filename
    filename = os.path.basename(file_path)
    dataset_title = dataset.title.replace(' ', '_')
    download_filename = f"health_report_{dataset_title}_{filename}"
    
    # Serve file
    try:
        return send_file(
            file_path,
            as_attachment=True,
            attachment_filename=download_filename,
            mimetype='text/html' if report_format == 'html' else 'application/pdf'
        )
    except Exception as e:
        current_app.logger.error(f"Error sending file: {str(e)}")
        return jsonify({"error": f"Error sending file: {str(e)}"}), 500