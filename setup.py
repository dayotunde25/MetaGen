import os
import sys
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    print("Downloading NLTK data...")
    for package in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.download(package, quiet=True)
            print(f"✓ Downloaded NLTK {package}")
        except Exception as e:
            print(f"! Failed to download NLTK {package}: {e}")

def download_spacy_model():
    """Download SpaCy models"""
    print("Downloading SpaCy models...")
    models = ['en_core_web_md']  # We use medium model as default
    for model in models:
        try:
            os.system(f"{sys.executable} -m spacy download {model}")
            print(f"✓ Downloaded SpaCy model {model}")
        except Exception as e:
            print(f"! Failed to download SpaCy model {model}: {e}")

def download_transformers_models():
    """Download and cache transformer models"""
    print("Downloading transformer models (this may take a while)...")
    try:
        from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
        
        # Download FLAN-T5 Base for offline description generation
        print("Downloading FLAN-T5 model...")
        T5Tokenizer.from_pretrained('google/flan-t5-base')
        T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
        print("✓ Downloaded FLAN-T5 model")

        # Download DistilBERT for semantic search (smaller but efficient)
        print("Downloading DistilBERT model...")
        AutoTokenizer.from_pretrained('distilbert-base-uncased')
        AutoModel.from_pretrained('distilbert-base-uncased')
        print("✓ Downloaded DistilBERT model")
        
    except Exception as e:
        print(f"! Failed to download transformer models: {e}")

class PostDevelopCommand(develop):
    """Post-development command to download models"""
    def run(self):
        develop.run(self)
        self.execute(download_nltk_data, (), "Downloading NLTK data")
        self.execute(download_spacy_model, (), "Downloading SpaCy model")
        self.execute(download_transformers_models, (), "Downloading transformer models")

class PostInstallCommand(install):
    """Post-installation command to download models"""
    def run(self):
        install.run(self)
        self.execute(download_nltk_data, (), "Downloading NLTK data")
        self.execute(download_spacy_model, (), "Downloading SpaCy model")
        self.execute(download_transformers_models, (), "Downloading transformer models")

setup(
    name="aimetaharvest",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.8,<3.11",
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    }
)
