"""
NLP Service for dataset processing and analysis.
"""

import re
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class NLPService:
    """Service for NLP processing of dataset content and metadata"""
    
    def __init__(self):
        self.nlp = None
        self.stemmer = None
        self.stop_words = set()
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP libraries if available"""
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        
        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                self.stemmer = PorterStemmer()
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                print("Warning: NLTK data not found. Download with: nltk.download('stopwords') and nltk.download('punkt')")
                self.stemmer = None
                self.stop_words = set()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using available NLP libraries"""
        if not text:
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        if self.nlp:
            return self._extract_keywords_spacy(text, max_keywords)
        elif NLTK_AVAILABLE:
            return self._extract_keywords_nltk(text, max_keywords)
        else:
            return self._extract_keywords_simple(text, max_keywords)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def _extract_keywords_spacy(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using spaCy"""
        doc = self.nlp(text)
        
        # Extract nouns, proper nouns, and adjectives
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Count frequency and return most common
        keyword_counts = Counter(keywords)
        return [word for word, _ in keyword_counts.most_common(max_keywords)]
    
    def _extract_keywords_nltk(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using NLTK"""
        try:
            tokens = word_tokenize(text)
            
            # Filter tokens
            keywords = []
            for token in tokens:
                if (len(token) > 2 and 
                    token.lower() not in self.stop_words and 
                    token.isalpha()):
                    if self.stemmer:
                        keywords.append(self.stemmer.stem(token.lower()))
                    else:
                        keywords.append(token.lower())
            
            # Count frequency and return most common
            keyword_counts = Counter(keywords)
            return [word for word, _ in keyword_counts.most_common(max_keywords)]
        except:
            return self._extract_keywords_simple(text, max_keywords)
    
    def _extract_keywords_simple(self, text: str, max_keywords: int) -> List[str]:
        """Simple keyword extraction without external libraries"""
        # Basic stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words = text.split()
        keywords = []
        
        for word in words:
            if (len(word) > 2 and 
                word.lower() not in stop_words and 
                word.isalpha()):
                keywords.append(word.lower())
        
        # Count frequency and return most common
        keyword_counts = Counter(keywords)
        return [word for word, _ in keyword_counts.most_common(max_keywords)]
    
    def suggest_tags(self, text: str, num_tags: int = 5) -> List[str]:
        """Suggest tags based on text content"""
        keywords = self.extract_keywords(text, num_tags * 2)
        
        # Filter and clean keywords for tags
        tags = []
        for keyword in keywords:
            if len(keyword) > 2 and keyword.isalpha():
                tags.append(keyword.replace('_', ' ').title())
        
        return tags[:num_tags]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis"""
        if not text:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        # Simple sentiment analysis using word lists
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'positive', 'beneficial', 'useful', 'valuable', 'important', 'significant'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'poor', 'useless', 
            'problematic', 'difficult', 'challenging', 'limited', 'incomplete'
        }
        
        words = self._clean_text(text).split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / total_sentiment_words
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = negative_count / total_sentiment_words
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        if not text or not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_
            })
        
        return entities
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        # Score sentences based on keyword frequency
        keywords = self.extract_keywords(text, 10)
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_words = self._clean_text(sentence).split()
            for word in sentence_words:
                if word in keywords:
                    score += 1
            sentence_scores[i] = score
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sentences.sort(key=lambda x: x[0])  # Sort by original order
        
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        return '. '.join(summary_sentences) + '.'


# Global service instance
nlp_service = NLPService()
