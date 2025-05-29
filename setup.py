#!/usr/bin/env python3
"""
AIMetaHarvest Local Setup Script

This script automates the setup process for running AIMetaHarvest locally.
It handles virtual environment creation, dependency installation, and basic configuration.
"""

import os
import sys
import subprocess
import platform
import urllib.request
import json
from pathlib import Path

class AIMetaHarvestSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_cmd = self._get_python_command()
        self.pip_cmd = self._get_pip_command()
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        
    def _get_python_command(self):
        """Get the appropriate Python command for the system."""
        if self.system == "windows":
            return "python"
        else:
            return "python3"
    
    def _get_pip_command(self):
        """Get the appropriate pip command for the system."""
        if self.system == "windows":
            return str(self.venv_path / "Scripts" / "pip")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def _run_command(self, command, shell=True, check=True):
        """Run a system command and handle errors."""
        try:
            print(f"🔧 Running: {command}")
            result = subprocess.run(command, shell=shell, check=check, 
                                  capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running command: {command}")
            print(f"Error: {e.stderr}")
            return None
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        print("🐍 Checking Python version...")
        try:
            result = subprocess.run([self.python_cmd, "--version"], 
                                  capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            print(f"✅ Found {version}")
            
            # Extract version number
            version_parts = version.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major < 3 or (major == 3 and minor < 8):
                print("❌ Python 3.8 or higher is required!")
                return False
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Python not found! Please install Python 3.8 or higher.")
            return False
    
    def check_mongodb(self):
        """Check if MongoDB is installed and running."""
        print("🍃 Checking MongoDB...")
        try:
            # Try to connect to MongoDB
            result = subprocess.run(["mongo", "--eval", "db.adminCommand('ismaster')"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ MongoDB is running")
                return True
            else:
                print("⚠️  MongoDB is installed but not running")
                return self._start_mongodb()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ MongoDB not found or not running")
            print("📋 Please install MongoDB following the guide in LOCAL_SETUP_GUIDE.md")
            return False
    
    def _start_mongodb(self):
        """Attempt to start MongoDB service."""
        print("🔄 Attempting to start MongoDB...")
        try:
            if self.system == "windows":
                result = self._run_command("net start MongoDB", check=False)
            elif self.system == "darwin":  # macOS
                result = self._run_command("brew services start mongodb/brew/mongodb-community", check=False)
            else:  # Linux
                result = self._run_command("sudo systemctl start mongod", check=False)
            
            if result and result.returncode == 0:
                print("✅ MongoDB started successfully")
                return True
            else:
                print("⚠️  Could not start MongoDB automatically")
                return False
        except Exception as e:
            print(f"⚠️  Error starting MongoDB: {e}")
            return False
    
    def create_virtual_environment(self):
        """Create Python virtual environment."""
        print("📦 Creating virtual environment...")
        if self.venv_path.exists():
            print("✅ Virtual environment already exists")
            return True
        
        result = self._run_command(f"{self.python_cmd} -m venv venv")
        if result and result.returncode == 0:
            print("✅ Virtual environment created")
            return True
        else:
            print("❌ Failed to create virtual environment")
            return False
    
    def install_dependencies(self):
        """Install Python dependencies."""
        print("📚 Installing dependencies...")
        
        # Upgrade pip first
        print("🔧 Upgrading pip...")
        self._run_command(f"{self.pip_cmd} install --upgrade pip")
        
        # Core Flask dependencies
        core_deps = [
            "Flask==2.3.3",
            "Flask-Login==0.6.3", 
            "Flask-WTF==1.1.1",
            "WTForms==3.0.1",
            "mongoengine==0.27.0",
            "pymongo==4.5.0",
            "Werkzeug==2.3.7",
            "python-dotenv==1.0.0"
        ]
        
        print("🔧 Installing core Flask dependencies...")
        for dep in core_deps:
            result = self._run_command(f"{self.pip_cmd} install {dep}")
            if not result or result.returncode != 0:
                print(f"⚠️  Warning: Failed to install {dep}")
        
        # Data processing dependencies
        data_deps = [
            "pandas==2.1.1",
            "numpy==1.24.3",
            "openpyxl==3.1.2",
            "lxml==4.9.3",
            "reportlab==4.0.4",
            "Pillow==10.0.0",
            "python-dateutil==2.8.2"
        ]
        
        print("🔧 Installing data processing dependencies...")
        for dep in data_deps:
            result = self._run_command(f"{self.pip_cmd} install {dep}")
            if not result or result.returncode != 0:
                print(f"⚠️  Warning: Failed to install {dep}")
        
        # ML/NLP dependencies (optional but recommended)
        print("🤖 Installing ML/NLP dependencies (this may take a while)...")
        ml_deps = [
            "scikit-learn==1.3.0",
            "sentence-transformers==2.2.2",
            "transformers==4.33.2",
            "torch==2.0.1"
        ]
        
        for dep in ml_deps:
            print(f"📥 Installing {dep}...")
            result = self._run_command(f"{self.pip_cmd} install {dep}")
            if not result or result.returncode != 0:
                print(f"⚠️  Warning: Failed to install {dep} (semantic search may not work)")
        
        print("✅ Dependencies installation completed")
        return True
    
    def create_directories(self):
        """Create necessary directories."""
        print("📁 Creating directories...")
        directories = [
            "uploads",
            "app/cache",
            "app/cache/search"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        return True
    
    def create_env_file(self):
        """Create .env configuration file."""
        print("⚙️  Creating .env configuration file...")
        env_path = self.project_root / ".env"
        
        if env_path.exists():
            print("✅ .env file already exists")
            return True
        
        env_content = """# Flask Configuration
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=dataset_metadata_manager

# Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216

# Semantic Search Configuration
ENABLE_SEMANTIC_SEARCH=true
CACHE_DIR=app/cache

# Debug Configuration
DEBUG=true
"""
        
        try:
            with open(env_path, 'w') as f:
                f.write(env_content)
            print("✅ .env file created")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def test_installation(self):
        """Test if the installation is working."""
        print("🧪 Testing installation...")
        
        # Test Python imports
        test_script = """
import sys
try:
    import flask
    import mongoengine
    import pandas
    import numpy
    print("✅ Core dependencies imported successfully")
    
    try:
        import sklearn
        import sentence_transformers
        print("✅ ML/NLP dependencies imported successfully")
    except ImportError as e:
        print(f"⚠️  ML/NLP dependencies not available: {e}")
    
    print("✅ Installation test passed")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
        
        # Write test script to temporary file
        test_file = self.project_root / "test_imports.py"
        try:
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            # Run test with virtual environment Python
            if self.system == "windows":
                python_path = self.venv_path / "Scripts" / "python"
            else:
                python_path = self.venv_path / "bin" / "python"
            
            result = self._run_command(f"{python_path} test_imports.py")
            
            # Clean up test file
            test_file.unlink()
            
            return result and result.returncode == 0
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process."""
        print("🚀 Starting AIMetaHarvest Local Setup")
        print("=" * 50)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Checking MongoDB", self.check_mongodb),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Creating directories", self.create_directories),
            ("Creating configuration", self.create_env_file),
            ("Testing installation", self.test_installation)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                print("\n📖 Please check LOCAL_SETUP_GUIDE.md for manual setup instructions")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate virtual environment:")
        if self.system == "windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Start the application:")
        print("   python run.py")
        print("3. Open browser to: http://localhost:5001")
        print("4. Login with admin/admin123")
        print("\n📖 For detailed instructions, see LOCAL_SETUP_GUIDE.md")
        
        return True

if __name__ == "__main__":
    setup = AIMetaHarvestSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)
