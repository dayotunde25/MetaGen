import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from collections import Counter

class NLPService:
    """Service for natural language processing tasks related to dataset metadata"""
    
    def __init__(self):
        """Initialize NLP service"""
        self.nlp = None
        self.stop_words = None
        self.stemmer = PorterStemmer()
        
    def _ensure_spacy_model(self):
        """Ensure spaCy model is loaded (lazy loading)"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                # If model isn't available, try to download it
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
                self.nlp = spacy.load('en_core_web_sm')
    
    def _ensure_nltk_resources(self):
        """Ensure NLTK resources are downloaded"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not text:
            return []
        
        # Ensure resources are available
        self._ensure_nltk_resources()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def _calculate_tf(self, term, document):
        """Calculate term frequency"""
        term_count = document.count(term)
        return term_count / len(document) if document else 0
    
    def _calculate_idf(self, term, documents):
        """Calculate inverse document frequency"""
        doc_count = sum(1 for doc in documents if term in doc)
        return np.log(len(documents) / (doc_count + 1))
    
    def _calculate_tfidf(self, term, document, documents):
        """Calculate TF-IDF score"""
        return self._calculate_tf(term, document) * self._calculate_idf(term, documents)
    
    def extract_keywords(self, text, num_keywords=10):
        """Extract keywords from text using TF-IDF-like approach"""
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        # If text is too short, return the tokens
        if len(tokens) < num_keywords * 2:
            return tokens[:num_keywords]
        
        # Split tokens into sentences (simplified approach)
        sentences = []
        sentence_size = max(5, len(tokens) // 10)
        for i in range(0, len(tokens), sentence_size):
            sentences.append(tokens[i:i+sentence_size])
        
        # Calculate TF-IDF for each term
        tfidf_scores = {}
        for term in set(tokens):
            tfidf_scores[term] = self._calculate_tfidf(term, tokens, sentences)
        
        # Return top keywords
        keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in keywords[:num_keywords]]
    
    def compute_text_similarity(self, text1, text2):
        """Compute similarity between two texts using token overlap"""
        # Preprocess texts
        tokens1 = set(self.preprocess_text(text1))
        tokens2 = set(self.preprocess_text(text2))
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def semantic_search(self, query, documents, document_ids=None):
        """Perform semantic search across documents"""
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        # If no tokens in query, return empty results
        if not query_tokens:
            return []
        
        # Calculate similarity scores
        scores = []
        for i, doc in enumerate(documents):
            # Calculate similarity score
            sim_score = self.compute_text_similarity(query, doc)
            
            # Store score and document ID
            doc_id = document_ids[i] if document_ids else i
            scores.append((doc_id, sim_score))
        
        # Sort by similarity score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def suggest_tags(self, text, num_tags=5):
        """Suggest tags based on text content"""
        # Make sure spaCy is loaded
        self._ensure_spacy_model()
        
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract nouns and named entities
        nouns = [token.text.lower() for token in doc if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 3]
        entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 3]
        
        # Count frequencies
        all_terms = nouns + entities
        term_counts = Counter(all_terms)
        
        # Return most common terms as tags
        tags = [term for term, _ in term_counts.most_common(num_tags)]
        
        # If we don't have enough tags, add keywords
        if len(tags) < num_tags:
            keywords = self.extract_keywords(text, num_tags - len(tags))
            tags.extend([kw for kw in keywords if kw not in tags])
        
        return tags[:num_tags]


# Create a singleton instance
nlp_service = NLPService()