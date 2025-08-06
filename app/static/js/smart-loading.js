/**
 * Smart Loading Animation System
 * Shows loading animations only when processes take longer than expected
 */

class SmartLoader {
    constructor() {
        this.loadingThreshold = 300; // Show loading after 300ms
        this.activeLoaders = new Map();
        this.loadingTemplates = {
            search: this.createSearchLoadingTemplate(),
            processing: this.createProcessingLoadingTemplate(),
            upload: this.createUploadLoadingTemplate(),
            general: this.createGeneralLoadingTemplate()
        };
        this.init();
    }

    init() {
        // Add CSS styles
        this.addStyles();
        
        // Auto-detect forms and add loading
        this.setupFormLoading();
        
        // Setup AJAX loading detection
        this.setupAjaxLoading();
    }

    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .smart-loader {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 255, 255, 0.95);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease, visibility 0.3s ease;
            }

            .smart-loader.show {
                opacity: 1;
                visibility: visible;
            }

            .smart-loader-content {
                text-align: center;
                padding: 2rem;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                width: 90%;
            }

            .smart-loader-spinner {
                width: 60px;
                height: 60px;
                margin: 0 auto 1rem;
                position: relative;
            }

            .smart-loader-spinner::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: 4px solid #e3f2fd;
                border-top: 4px solid #2196f3;
                border-radius: 50%;
                animation: smart-loader-spin 1s linear infinite;
            }

            @keyframes smart-loader-spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .smart-loader-title {
                font-size: 1.2rem;
                font-weight: 600;
                color: #333;
                margin-bottom: 0.5rem;
            }

            .smart-loader-subtitle {
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 1rem;
            }

            .smart-loader-progress {
                width: 100%;
                height: 4px;
                background: #e3f2fd;
                border-radius: 2px;
                overflow: hidden;
                margin-bottom: 1rem;
            }

            .smart-loader-progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #2196f3, #21cbf3);
                border-radius: 2px;
                animation: smart-loader-progress 2s ease-in-out infinite;
            }

            @keyframes smart-loader-progress {
                0% { transform: translateX(-100%); }
                50% { transform: translateX(0%); }
                100% { transform: translateX(100%); }
            }

            .smart-loader-tips {
                font-size: 0.8rem;
                color: #888;
                font-style: italic;
            }

            /* Search-specific animations */
            .search-loader .smart-loader-spinner::before {
                border-top-color: #4caf50;
            }

            .search-loader .smart-loader-progress-bar {
                background: linear-gradient(90deg, #4caf50, #8bc34a);
            }

            /* Processing-specific animations */
            .processing-loader .smart-loader-spinner::before {
                border-top-color: #ff9800;
            }

            .processing-loader .smart-loader-progress-bar {
                background: linear-gradient(90deg, #ff9800, #ffc107);
            }

            /* Upload-specific animations */
            .upload-loader .smart-loader-spinner::before {
                border-top-color: #9c27b0;
            }

            .upload-loader .smart-loader-progress-bar {
                background: linear-gradient(90deg, #9c27b0, #e91e63);
            }

            /* Inline loader for smaller areas */
            .smart-loader-inline {
                position: relative;
                padding: 2rem;
                text-align: center;
                background: #f8f9fa;
                border-radius: 8px;
                margin: 1rem 0;
            }

            .smart-loader-inline .smart-loader-spinner {
                width: 40px;
                height: 40px;
            }
        `;
        document.head.appendChild(style);
    }

    createSearchLoadingTemplate() {
        return {
            title: "🔍 Searching Datasets",
            subtitle: "Analyzing content with advanced NLP techniques...",
            tips: "Using BERT embeddings, entity recognition, and semantic analysis",
            className: "search-loader"
        };
    }

    createProcessingLoadingTemplate() {
        return {
            title: "⚙️ Processing Dataset",
            subtitle: "Running quality assessment and metadata generation...",
            tips: "This may take a few moments for large datasets",
            className: "processing-loader"
        };
    }

    createUploadLoadingTemplate() {
        return {
            title: "📤 Uploading Dataset",
            subtitle: "Uploading and validating your dataset...",
            tips: "Please don't close this window",
            className: "upload-loader"
        };
    }

    createGeneralLoadingTemplate() {
        return {
            title: "⏳ Loading",
            subtitle: "Please wait while we process your request...",
            tips: "This should only take a moment",
            className: "general-loader"
        };
    }

    show(type = 'general', customOptions = {}) {
        const loaderId = Date.now().toString();
        const template = { ...this.loadingTemplates[type], ...customOptions };
        
        // Create loader element
        const loader = document.createElement('div');
        loader.className = `smart-loader ${template.className}`;
        loader.id = `smart-loader-${loaderId}`;
        
        loader.innerHTML = `
            <div class="smart-loader-content">
                <div class="smart-loader-spinner"></div>
                <div class="smart-loader-title">${template.title}</div>
                <div class="smart-loader-subtitle">${template.subtitle}</div>
                <div class="smart-loader-progress">
                    <div class="smart-loader-progress-bar"></div>
                </div>
                <div class="smart-loader-tips">${template.tips}</div>
            </div>
        `;
        
        document.body.appendChild(loader);
        
        // Show with delay to avoid flash for fast operations
        const showTimeout = setTimeout(() => {
            loader.classList.add('show');
        }, this.loadingThreshold);
        
        this.activeLoaders.set(loaderId, {
            element: loader,
            showTimeout: showTimeout,
            startTime: Date.now()
        });
        
        return loaderId;
    }

    hide(loaderId) {
        const loader = this.activeLoaders.get(loaderId);
        if (!loader) return;
        
        // Clear show timeout if it hasn't triggered yet
        clearTimeout(loader.showTimeout);
        
        const element = loader.element;
        const duration = Date.now() - loader.startTime;
        
        // If the operation was very fast, don't show the loader at all
        if (duration < this.loadingThreshold) {
            element.remove();
        } else {
            // Fade out the loader
            element.classList.remove('show');
            setTimeout(() => {
                if (element.parentNode) {
                    element.remove();
                }
            }, 300);
        }
        
        this.activeLoaders.delete(loaderId);
    }

    showInline(container, type = 'general', customOptions = {}) {
        const template = { ...this.loadingTemplates[type], ...customOptions };
        
        const loader = document.createElement('div');
        loader.className = `smart-loader-inline ${template.className}`;
        
        loader.innerHTML = `
            <div class="smart-loader-spinner"></div>
            <div class="smart-loader-title">${template.title}</div>
            <div class="smart-loader-subtitle">${template.subtitle}</div>
        `;
        
        container.innerHTML = '';
        container.appendChild(loader);
        
        return loader;
    }

    setupFormLoading() {
        // Auto-detect search forms
        const searchForms = document.querySelectorAll('form[action*="search"], #searchForm');
        searchForms.forEach(form => {
            form.addEventListener('submit', (e) => {
                const formData = new FormData(form);
                const hasQuery = formData.get('query') && formData.get('query').trim();
                
                if (hasQuery) {
                    const loaderId = this.show('search', {
                        subtitle: `Searching for "${formData.get('query')}"...`,
                        tips: "Using advanced semantic search with NLP analysis"
                    });
                    
                    // Store loader ID for cleanup on page unload
                    window.currentSearchLoader = loaderId;
                }
            });
        });

        // Auto-detect upload forms
        const uploadForms = document.querySelectorAll('form[enctype*="multipart"], form input[type="file"]');
        uploadForms.forEach(form => {
            const actualForm = form.tagName === 'FORM' ? form : form.closest('form');
            if (actualForm) {
                actualForm.addEventListener('submit', () => {
                    const loaderId = this.show('upload');
                    window.currentUploadLoader = loaderId;
                });
            }
        });

        // Auto-detect processing forms
        const processingForms = document.querySelectorAll('form[action*="process"], .process-form');
        processingForms.forEach(form => {
            form.addEventListener('submit', () => {
                const loaderId = this.show('processing');
                window.currentProcessingLoader = loaderId;
            });
        });
    }

    setupAjaxLoading() {
        // Intercept fetch requests
        const originalFetch = window.fetch;
        window.fetch = (...args) => {
            const url = args[0];
            let loaderId = null;
            
            // Determine loading type based on URL
            if (url.includes('search')) {
                loaderId = this.show('search');
            } else if (url.includes('process')) {
                loaderId = this.show('processing');
            } else if (url.includes('upload')) {
                loaderId = this.show('upload');
            }
            
            return originalFetch(...args).finally(() => {
                if (loaderId) {
                    this.hide(loaderId);
                }
            });
        };

        // Intercept XMLHttpRequest
        const originalXHROpen = XMLHttpRequest.prototype.open;
        XMLHttpRequest.prototype.open = function(method, url, ...args) {
            this._smartLoaderUrl = url;
            return originalXHROpen.call(this, method, url, ...args);
        };

        const originalXHRSend = XMLHttpRequest.prototype.send;
        XMLHttpRequest.prototype.send = function(...args) {
            let loaderId = null;
            
            if (this._smartLoaderUrl) {
                if (this._smartLoaderUrl.includes('search')) {
                    loaderId = smartLoader.show('search');
                } else if (this._smartLoaderUrl.includes('process')) {
                    loaderId = smartLoader.show('processing');
                } else if (this._smartLoaderUrl.includes('upload')) {
                    loaderId = smartLoader.show('upload');
                }
            }
            
            const cleanup = () => {
                if (loaderId) {
                    smartLoader.hide(loaderId);
                }
            };
            
            this.addEventListener('load', cleanup);
            this.addEventListener('error', cleanup);
            this.addEventListener('abort', cleanup);
            
            return originalXHRSend.call(this, ...args);
        };
    }

    // Utility methods for manual control
    showSearchLoader(query = '') {
        return this.show('search', {
            subtitle: query ? `Searching for "${query}"...` : "Searching datasets...",
            tips: "Using advanced semantic search with NLP analysis"
        });
    }

    showProcessingLoader(datasetName = '') {
        return this.show('processing', {
            subtitle: datasetName ? `Processing "${datasetName}"...` : "Processing dataset...",
            tips: "Running quality assessment and metadata generation"
        });
    }

    showUploadLoader(fileName = '') {
        return this.show('upload', {
            subtitle: fileName ? `Uploading "${fileName}"...` : "Uploading dataset...",
            tips: "Please don't close this window"
        });
    }
}

// Initialize smart loader when DOM is ready
let smartLoader;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        smartLoader = new SmartLoader();
        window.smartLoader = smartLoader;
    });
} else {
    smartLoader = new SmartLoader();
    window.smartLoader = smartLoader;
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.currentSearchLoader) {
        smartLoader.hide(window.currentSearchLoader);
    }
    if (window.currentUploadLoader) {
        smartLoader.hide(window.currentUploadLoader);
    }
    if (window.currentProcessingLoader) {
        smartLoader.hide(window.currentProcessingLoader);
    }
});
