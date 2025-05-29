/**
 * Real-time progress tracking for dataset processing
 */

class ProgressTracker {
    constructor(datasetId, options = {}) {
        this.datasetId = datasetId;
        this.options = {
            pollInterval: 2000, // Poll every 2 seconds
            progressBarId: 'processing-progress',
            statusTextId: 'processing-status',
            messageTextId: 'processing-message',
            onComplete: null,
            onError: null,
            ...options
        };
        
        this.isPolling = false;
        this.pollTimer = null;
        this.progressBar = null;
        this.statusText = null;
        this.messageText = null;
        
        this.init();
    }
    
    init() {
        // Get DOM elements
        this.progressBar = document.getElementById(this.options.progressBarId);
        this.statusText = document.getElementById(this.options.statusTextId);
        this.messageText = document.getElementById(this.options.messageTextId);
        
        // Start polling if elements exist
        if (this.progressBar || this.statusText || this.messageText) {
            this.startPolling();
        }
    }
    
    startPolling() {
        if (this.isPolling) return;
        
        this.isPolling = true;
        this.poll();
    }
    
    stopPolling() {
        this.isPolling = false;
        if (this.pollTimer) {
            clearTimeout(this.pollTimer);
            this.pollTimer = null;
        }
    }
    
    async poll() {
        if (!this.isPolling) return;
        
        try {
            const response = await fetch(`/api/datasets/${this.datasetId}/progress`);
            const data = await response.json();
            
            if (response.ok) {
                this.updateProgress(data);
                
                // Continue polling if still active
                if (data.active && data.status !== 'completed' && data.status !== 'failed') {
                    this.scheduleNextPoll();
                } else {
                    this.stopPolling();
                    this.handleCompletion(data);
                }
            } else {
                this.handleError(data.error || 'Failed to get progress');
            }
        } catch (error) {
            this.handleError(error.message);
        }
    }
    
    scheduleNextPoll() {
        this.pollTimer = setTimeout(() => {
            this.poll();
        }, this.options.pollInterval);
    }
    
    updateProgress(data) {
        const { progress, status, message } = data;
        
        // Update progress bar
        if (this.progressBar) {
            this.progressBar.style.width = `${progress}%`;
            this.progressBar.setAttribute('aria-valuenow', progress);
            
            // Update progress bar color based on status
            this.progressBar.className = this.progressBar.className.replace(/bg-\w+/, '');
            if (status === 'completed') {
                this.progressBar.classList.add('bg-success');
            } else if (status === 'failed') {
                this.progressBar.classList.add('bg-danger');
            } else if (status === 'processing') {
                this.progressBar.classList.add('bg-primary');
            } else {
                this.progressBar.classList.add('bg-info');
            }
        }
        
        // Update status text
        if (this.statusText) {
            this.statusText.textContent = this.formatStatus(status);
            this.statusText.className = this.statusText.className.replace(/text-\w+/, '');
            
            if (status === 'completed') {
                this.statusText.classList.add('text-success');
            } else if (status === 'failed') {
                this.statusText.classList.add('text-danger');
            } else if (status === 'processing') {
                this.statusText.classList.add('text-primary');
            } else {
                this.statusText.classList.add('text-info');
            }
        }
        
        // Update message text
        if (this.messageText && message) {
            this.messageText.textContent = message;
        }
        
        // Update progress percentage display
        const progressPercent = document.getElementById('progress-percent');
        if (progressPercent) {
            progressPercent.textContent = `${progress}%`;
        }
    }
    
    formatStatus(status) {
        const statusMap = {
            'pending': 'Pending',
            'processing': 'Processing',
            'completed': 'Completed',
            'failed': 'Failed',
            'error': 'Error',
            'not_started': 'Not Started'
        };
        
        return statusMap[status] || status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    handleCompletion(data) {
        if (data.status === 'completed') {
            this.showNotification('Processing completed successfully!', 'success');
            
            // Refresh the page after a short delay to show updated data
            setTimeout(() => {
                window.location.reload();
            }, 2000);
        } else if (data.status === 'failed') {
            this.showNotification(`Processing failed: ${data.message || data.error}`, 'error');
        }
        
        // Call completion callback if provided
        if (this.options.onComplete) {
            this.options.onComplete(data);
        }
    }
    
    handleError(error) {
        console.error('Progress tracking error:', error);
        this.showNotification(`Error tracking progress: ${error}`, 'error');
        
        // Stop polling on error
        this.stopPolling();
        
        // Call error callback if provided
        if (this.options.onError) {
            this.options.onError(error);
        }
    }
    
    showNotification(message, type = 'info') {
        // Create a simple notification
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.zIndex = '9999';
        notification.style.minWidth = '300px';
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    // Public method to manually trigger progress check
    checkProgress() {
        this.poll();
    }
    
    // Public method to restart polling
    restart() {
        this.stopPolling();
        this.startPolling();
    }
}

// Auto-initialize progress tracker if dataset ID is available
document.addEventListener('DOMContentLoaded', function() {
    const datasetIdElement = document.querySelector('[data-dataset-id]');
    if (datasetIdElement) {
        const datasetId = datasetIdElement.getAttribute('data-dataset-id');
        
        // Check if there's a processing indicator on the page
        const processingIndicator = document.getElementById('processing-progress') || 
                                  document.getElementById('processing-status');
        
        if (processingIndicator && datasetId) {
            window.progressTracker = new ProgressTracker(datasetId);
        }
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProgressTracker;
} else {
    window.ProgressTracker = ProgressTracker;
}
