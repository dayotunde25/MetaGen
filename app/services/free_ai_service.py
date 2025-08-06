"""
Free AI Models Service for Dataset Description Generation
Supports Mistral AI and Groq with robust error handling and text cleaning
"""

import os
import re
import time
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeAIService:
    """Service for generating descriptions using free AI models"""
    
    def __init__(self):
        """Initialize the Free AI Service with API clients"""
        self.mistral_client = None
        self.groq_client = None
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize API clients for free AI models"""
        # Initialize Mistral AI
        try:
            from mistralai.client import MistralClient
            api_key = os.getenv('MISTRAL_API_KEY')
            if api_key and api_key != 'your_mistral_key_here':
                self.mistral_client = MistralClient(api_key=api_key)
                logger.info("✅ Mistral AI client initialized successfully")
            else:
                logger.warning("⚠️ Mistral API key not found or invalid")
        except ImportError:
            logger.warning("⚠️ Mistral AI package not installed")
        except Exception as e:
            logger.error(f"⚠️ Mistral AI initialization failed: {e}")
        
        # Initialize Groq
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if api_key and api_key != 'your_groq_key_here':
                self.groq_client = Groq(api_key=api_key)
                logger.info("✅ Groq client initialized successfully")
            else:
                logger.warning("⚠️ Groq API key not found or invalid")
        except ImportError:
            logger.warning("⚠️ Groq package not installed")
        except Exception as e:
            logger.error(f"⚠️ Groq initialization failed: {e}")
    
    def clean_generated_text(self, text: str) -> str:
        """
        Clean generated text by removing unnecessary characters and formatting
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text without special characters
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[#@$%^&*+=\[\]{}|\\<>~`]', '', text)
        
        # Clean up multiple spaces and newlines
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure proper sentence structure
        if text and not text.endswith('.'):
            text += '.'
        
        return text
    
    def generate_description_mistral(self, dataset_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate description using Mistral AI
        
        Args:
            dataset_info: Dictionary containing dataset information
            
        Returns:
            Generated description or None if failed
        """
        if not self.mistral_client:
            return None
        
        try:
            # Prepare prompt for Mistral
            prompt = self._create_description_prompt(dataset_info)
            
            from mistralai.models.chat_completion import ChatMessage
            
            messages = [
                ChatMessage(
                    role="system",
                    content="You are a data science expert. Generate clear, professional dataset descriptions without markdown formatting or special characters. Give detailed explanation with focus on content, structure, and potential use cases."
                ),
                ChatMessage(
                    role="user",
                    content=prompt
                )
            ]
            
            # Generate response
            response = self.mistral_client.chat(
                model="mistral-small",
                messages=messages,
                max_tokens=800,
                temperature=0.3
            )
            
            if response and response.choices:
                description = response.choices[0].message.content
                cleaned_description = self.clean_generated_text(description)
                logger.info(f"✅ Mistral description generated: {len(cleaned_description)} chars")
                return cleaned_description
            
        except Exception as e:
            logger.error(f"⚠️ Mistral description generation failed: {e}")
        
        return None
    
    def generate_description_groq(self, dataset_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate description using Groq
        
        Args:
            dataset_info: Dictionary containing dataset information
            
        Returns:
            Generated description or None if failed
        """
        if not self.groq_client:
            return None
        
        try:
            # Prepare prompt for Groq
            prompt = self._create_description_prompt(dataset_info)
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data science expert. Generate clear, professional dataset descriptions without markdown formatting or special characters. Give detailed explanation with focus on content, structure, and potential use cases."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            if response and response.choices:
                description = response.choices[0].message.content
                cleaned_description = self.clean_generated_text(description)
                logger.info(f"✅ Groq description generated: {len(cleaned_description)} chars")
                return cleaned_description
            
        except Exception as e:
            logger.error(f"⚠️ Groq description generation failed: {e}")
        
        return None
    
    def _create_description_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """
        Create a comprehensive prompt for AI description generation
        
        Args:
            dataset_info: Dictionary containing dataset information
            
        Returns:
            Formatted prompt string
        """
        title = dataset_info.get('title', 'Dataset')
        record_count = dataset_info.get('record_count', 0)
        field_count = len(dataset_info.get('field_names', []))
        field_names = dataset_info.get('field_names', [])
        data_types = dataset_info.get('data_types', {})
        # Ensure data_types is a dictionary
        if not isinstance(data_types, dict):
            data_types = {}
        category = dataset_info.get('category', 'General')
        
        prompt = f"""Generate a comprehensive description for this dataset:

Title: {title}
Category: {category}
Records: {record_count:,}
Fields: {field_count}

Field Information:
"""
        
        # Add field details
        if field_names:
            for field in field_names[:10]:  # Limit to first 10 fields
                field_type = data_types.get(field, 'unknown')
                prompt += f"- {field} ({field_type})\n"
        
        prompt += f"""
Please provide a very detailed description that includes:
1. What this dataset contains and represents
2. The structure and organization of the data
3. Potential research applications and use cases
4. Key insights about the data characteristics
5. Suitable analysis methods or techniques

Write in clear, professional language without markdown formatting or special characters. Make it expressive, informative and suitable for academic or research purposes."""
        
        return prompt
    
    def generate_enhanced_description(self, dataset_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate enhanced description using available free AI models
        Priority order: GrokAI -> MistralAI -> FLAN-T5 -> other fallbacks

        Args:
            dataset_info: Dictionary containing dataset information

        Returns:
            Generated description or None if all models fail
        """
        # Try GrokAI first (as requested by user)
        description = self.generate_description_groq(dataset_info)
        if description and len(description) > 100:
            logger.info("✅ GrokAI description generated successfully")
            return description

        # Fallback to MistralAI (second priority)
        description = self.generate_description_mistral(dataset_info)
        if description and len(description) > 100:
            logger.info("✅ MistralAI description generated successfully")
            return description

        logger.warning("⚠️ All free AI models failed or returned short descriptions")
        return None

    def generate_python_code_mistral(self, dataset_info: Dict[str, Any]) -> Optional[str]:
        """Generate Python analysis code using Mistral AI"""
        if not self.mistral_client:
            return None

        try:
            prompt = self._create_python_code_prompt(dataset_info)

            from mistralai.models.chat_completion import ChatMessage

            messages = [
                ChatMessage(
                    role="system",
                    content="You are a Python data science expert. Generate clean, well-commented Python code for dataset analysis. Include imports, data loading, exploration, visualization, and basic analysis. Use pandas, numpy, matplotlib, and seaborn. Make the code practical and executable."
                ),
                ChatMessage(
                    role="user",
                    content=prompt
                )
            ]

            response = self.mistral_client.chat(
                model="mistral-small",
                messages=messages,
                max_tokens=1000,
                temperature=0.2
            )

            if response and response.choices:
                code = response.choices[0].message.content.strip()
                code = self._clean_python_code(code, dataset_info)
                logger.info("✅ Mistral AI Python code generation successful")
                return code

        except Exception as e:
            logger.error(f"❌ Mistral AI Python code generation failed: {e}")

        return None

    def _create_python_code_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for Python code generation"""
        title = dataset_info.get('title', 'Dataset')
        record_count = dataset_info.get('record_count', 0)
        field_names = dataset_info.get('field_names', [])
        category = dataset_info.get('category', 'General')
        format_type = dataset_info.get('format', 'csv')

        field_context = ""
        if field_names:
            key_fields = field_names[:5]
            field_context = f"The dataset has {len(field_names)} columns including: {', '.join(key_fields)}"

        prompt = f"""Generate comprehensive Python code to analyze the "{title}" dataset:

Dataset Details:
- Title: {title}
- Records: {record_count:,}
- Format: {format_type.upper()}
- Category: {category}
{field_context}

Requirements:
1. Import necessary libraries (pandas, numpy, matplotlib, seaborn)
2. Load the dataset from file
3. Basic data exploration (shape, info, describe, head)
4. Check for missing values and data types
5. Generate appropriate visualizations based on data type
6. Basic statistical analysis
7. Data quality checks

Make the code practical, well-commented, and ready to execute."""

        return prompt

    def _clean_python_code(self, raw_code: str, dataset_info: Dict[str, Any]) -> str:
        """Clean and format the generated Python code"""
        title = dataset_info.get('title', 'Dataset')

        # Remove markdown code blocks if present
        if '```python' in raw_code:
            raw_code = raw_code.split('```python')[1]
        if '```' in raw_code:
            raw_code = raw_code.split('```')[0]

        # Clean up the code
        lines = raw_code.strip().split('\n')
        cleaned_lines = []

        # Add header if not present
        if not any('# Python Analysis Code' in line for line in lines[:3]):
            cleaned_lines.append(f"# Python Analysis Code for {title}")
            cleaned_lines.append("# Auto-generated by AI")
            cleaned_lines.append("")

        # Process each line
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def generate_enhanced_python_code(self, dataset_info: Dict[str, Any]) -> Optional[str]:
        """Generate enhanced Python code using available free AI models"""
        # Try Mistral first (usually higher quality)
        code = self.generate_python_code_mistral(dataset_info)
        if code and len(code) > 200:
            return code

        logger.warning("⚠️ Free AI Python code generation failed")
        return None

    def is_available(self) -> bool:
        """Check if any free AI models are available"""
        return self.mistral_client is not None or self.groq_client is not None


# Global instance
free_ai_service = FreeAIService()
