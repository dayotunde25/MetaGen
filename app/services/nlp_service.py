import os
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class NLPService:
    """Service for natural language processing tasks related to dataset metadata"""
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Load small English model for spaCy (faster)
        self.nlp = None  # Lazy-loaded
    
    def _ensure_spacy_model(self):
        """Ensure spaCy model is loaded (lazy loading)"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                # If model not found, download it
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not text:
            return []
            
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        tokens = [t for t in tokens if t.isalnum() and t not in self.stopwords]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def extract_keywords(self, text, num_keywords=10):
        """Extract keywords from text using TF-IDF-like approach"""
        if not text:
            return []
            
        tokens = self.preprocess_text(text)
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Get the most common tokens
        keywords = [kw for kw, _ in token_counts.most_common(num_keywords)]
        
        return keywords
    
    def compute_text_similarity(self, text1, text2):
        """Compute similarity between two texts using token overlap"""
        tokens1 = set(self.preprocess_text(text1))
        tokens2 = set(self.preprocess_text(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
            
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def semantic_search(self, query, documents, document_ids=None):
        """Perform semantic search across documents"""
        self._ensure_spacy_model()
        
        if not query or not documents:
            return []
            
        # Process query with spaCy
        query_doc = self.nlp(query)
        
        # Process all documents with spaCy and compute similarities
        results = []
        for i, doc_text in enumerate(documents):
            doc_id = document_ids[i] if document_ids else i
            if not doc_text:
                continue
                
            spacy_doc = self.nlp(doc_text)
            similarity = query_doc.similarity(spacy_doc)
            
            results.append({
                'id': doc_id,
                'similarity': similarity
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def suggest_tags(self, text, num_tags=5):
        """Suggest tags based on text content"""
        self._ensure_spacy_model()
        
        if not text:
            return []
            
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Combine and count frequencies
        candidates = entities + noun_phrases
        candidates = [c.lower() for c in candidates if len(c) > 2]
        
        # Count and get most common
        tag_counter = Counter(candidates)
        suggested_tags = [tag for tag, _ in tag_counter.most_common(num_tags)]
        
        return suggested_tags

# Initialize singleton instance
nlp_service = NLPService()