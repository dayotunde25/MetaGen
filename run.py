#!/usr/bin/env python3
"""
Dataset Metadata Manager - Flask Application Runner

This is the main entry point for the Flask application.
Run this file to start the development server.
"""

import os
from app import create_app

def main():
    """Main function to run the Flask application"""
    
    # Create the Flask application
    app = create_app()
    
    # Print startup information
    print("🚀 Dataset Metadata Manager")
    print("=" * 50)
    print("📊 Admin credentials: admin / admin123")
    print("🌐 Access: http://127.0.0.1:5001")
    print("✅ Features:")
    print("   - File upload with automatic processing")
    print("   - Real-time progress tracking")
    print("   - NLP analysis and quality scoring")
    print("   - Metadata generation")
    print("   - FAIR compliance assessment")
    print("🔄 Processing queue enabled")
    print("🔒 CSRF protection enabled")
    print("=" * 50)
    
    # Run the application
    try:
        app.run(
            host='127.0.0.1',
            port=5001,
            debug=True,
            use_reloader=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
