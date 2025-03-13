// Main JavaScript file for Dataset Metadata Manager

// Document ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
    
    // Enable confirm dialogs
    const confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(function(button) {
        button.addEventListener('click', function(e) {
            if (!confirm(this.getAttribute('data-confirm'))) {
                e.preventDefault();
            }
        });
    });
    
    // Handle dataset search form
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            const searchQuery = document.getElementById('query').value.trim();
            if (searchQuery === '') {
                e.preventDefault();
                showSearchError('Please enter a search query');
            }
        });
    }
    
    // Function to show search error
    function showSearchError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger alert-dismissible fade show mt-2';
        errorDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const searchForm = document.getElementById('searchForm');
        searchForm.after(errorDiv);
        
        // Auto remove after 3 seconds
        setTimeout(function() {
            errorDiv.remove();
        }, 3000);
    }
    
    // Dynamic FAIR score visualization
    const fairScores = document.querySelectorAll('.fair-score');
    fairScores.forEach(function(scoreElement) {
        const score = parseFloat(scoreElement.getAttribute('data-score'));
        const category = scoreElement.getAttribute('data-category');
        let color;
        
        if (score >= 80) {
            color = '#4CAF50'; // Green for high
        } else if (score >= 50) {
            color = '#FFC107'; // Yellow for medium
        } else {
            color = '#F44336'; // Red for low
        }
        
        scoreElement.style.width = score + '%';
        scoreElement.style.backgroundColor = color;
        
        // Add tooltip with category
        if (category) {
            scoreElement.setAttribute('title', category + ': ' + score + '%');
            new bootstrap.Tooltip(scoreElement);
        }
    });
});