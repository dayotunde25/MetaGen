"""
Semantic Search Service using NLP techniques for better dataset discovery.

This service implements multiple search algorithms:
- TF-IDF for traditional text matching
- BERT embeddings for semantic similarity
- Hybrid scoring combining multiple techniques
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SemanticSearchService:
    """
    Advanced semantic search service for datasets using multiple NLP techniques.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the semantic search service.
        
        Args:
            cache_dir: Directory to cache models and embeddings
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'app', 'cache', 'search')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.dataset_embeddings = {}
        self.dataset_texts = {}
        self.is_initialized = False
        
        # Search weights for hybrid scoring
        self.weights = {
            'tfidf': 0.3,
            'bert': 0.4,
            'metadata': 0.2,
            'popularity': 0.1
        }
    
    def initialize(self):
        """Initialize the search models and load cached data if available."""
        try:
            self._load_tfidf_model()
            self._load_bert_model()
            self.is_initialized = True
            logger.info("Semantic search service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic search: {e}")
            # Fall back to basic search
            self.is_initialized = False
    
    def _load_tfidf_model(self):
        """Load or initialize TF-IDF vectorizer."""
        try:
            tfidf_path = os.path.join(self.cache_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded cached TF-IDF vectorizer")
            else:
                # Initialize new TF-IDF vectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                logger.info("Initialized new TF-IDF vectorizer")
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {e}")
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def _load_bert_model(self):
        """Load BERT model for semantic embeddings."""
        try:
            # Try to import sentence-transformers (lightweight BERT alternative)
            from sentence_transformers import SentenceTransformer
            
            model_path = os.path.join(self.cache_dir, 'sentence_transformer')
            if os.path.exists(model_path):
                self.bert_model = SentenceTransformer(model_path)
                logger.info("Loaded cached BERT model")
            else:
                # Use a lightweight, fast model for production
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Cache the model
                self.bert_model.save(model_path)
                logger.info("Downloaded and cached BERT model")
                
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to TF-IDF only")
            self.bert_model = None
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.bert_model = None
    
    def index_datasets(self, datasets: List[Any]) -> bool:
        """
        Index datasets for semantic search.
        
        Args:
            datasets: List of dataset objects to index
            
        Returns:
            True if indexing successful, False otherwise
        """
        try:
            if not datasets:
                logger.warning("No datasets to index")
                return False
            
            # Extract text content from datasets
            dataset_texts = []
            dataset_ids = []
            
            for dataset in datasets:
                text_content = self._extract_dataset_text(dataset)
                dataset_texts.append(text_content)
                dataset_ids.append(str(dataset.id))
                self.dataset_texts[str(dataset.id)] = text_content
            
            # Build TF-IDF matrix
            if self.tfidf_vectorizer and dataset_texts:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(dataset_texts)
                
                # Save TF-IDF vectorizer
                tfidf_path = os.path.join(self.cache_dir, 'tfidf_vectorizer.pkl')
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                
                logger.info(f"Built TF-IDF matrix for {len(dataset_texts)} datasets")
            
            # Build BERT embeddings
            if self.bert_model and dataset_texts:
                embeddings = self.bert_model.encode(dataset_texts, show_progress_bar=False)
                
                for i, dataset_id in enumerate(dataset_ids):
                    self.dataset_embeddings[dataset_id] = embeddings[i]
                
                # Save embeddings
                embeddings_path = os.path.join(self.cache_dir, 'bert_embeddings.pkl')
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(self.dataset_embeddings, f)
                
                logger.info(f"Built BERT embeddings for {len(dataset_texts)} datasets")
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing datasets: {e}")
            return False
    
    def _extract_dataset_text(self, dataset) -> str:
        """
        Extract searchable text content from a dataset.
        
        Args:
            dataset: Dataset object
            
        Returns:
            Combined text content for search indexing
        """
        text_parts = []
        
        # Add title (with higher weight by repeating)
        if hasattr(dataset, 'title') and dataset.title:
            text_parts.extend([dataset.title] * 3)  # Title gets 3x weight
        
        # Add description
        if hasattr(dataset, 'description') and dataset.description:
            text_parts.append(dataset.description)
        
        # Add tags
        if hasattr(dataset, 'tags') and dataset.tags:
            if isinstance(dataset.tags, str):
                tags = [tag.strip() for tag in dataset.tags.split(',')]
            else:
                tags = dataset.tags
            text_parts.extend(tags)
        
        # Add category and data type
        if hasattr(dataset, 'category') and dataset.category:
            text_parts.append(dataset.category)
        
        if hasattr(dataset, 'data_type') and dataset.data_type:
            text_parts.append(dataset.data_type)
        
        # Add source information
        if hasattr(dataset, 'source') and dataset.source:
            text_parts.append(dataset.source)
        
        return ' '.join(text_parts)
    
    def search(self, query: str, datasets: List[Any], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform semantic search on datasets.
        
        Args:
            query: Search query
            datasets: List of dataset objects to search
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores
        """
        if not query or not datasets:
            return []
        
        try:
            # If not initialized, fall back to basic search
            if not self.is_initialized:
                return self._basic_search(query, datasets, limit)
            
            # Get scores from different algorithms
            tfidf_scores = self._get_tfidf_scores(query, datasets)
            bert_scores = self._get_bert_scores(query, datasets)
            metadata_scores = self._get_metadata_scores(query, datasets)
            popularity_scores = self._get_popularity_scores(datasets)
            
            # Combine scores
            combined_scores = self._combine_scores(
                datasets, tfidf_scores, bert_scores, metadata_scores, popularity_scores
            )
            
            # Sort by combined score and return top results
            results = sorted(combined_scores, key=lambda x: x['score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._basic_search(query, datasets, limit)
    
    def _get_tfidf_scores(self, query: str, datasets: List[Any]) -> Dict[str, float]:
        """Get TF-IDF similarity scores for the query."""
        scores = {}
        
        try:
            if not self.tfidf_vectorizer or not self.tfidf_matrix:
                return scores
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            for i, dataset in enumerate(datasets):
                if i < len(similarities):
                    scores[str(dataset.id)] = float(similarities[i])
                    
        except Exception as e:
            logger.error(f"Error calculating TF-IDF scores: {e}")
        
        return scores
    
    def _get_bert_scores(self, query: str, datasets: List[Any]) -> Dict[str, float]:
        """Get BERT semantic similarity scores for the query."""
        scores = {}
        
        try:
            if not self.bert_model or not self.dataset_embeddings:
                return scores
            
            # Get query embedding
            query_embedding = self.bert_model.encode([query])[0]
            
            # Calculate similarities
            for dataset in datasets:
                dataset_id = str(dataset.id)
                if dataset_id in self.dataset_embeddings:
                    dataset_embedding = self.dataset_embeddings[dataset_id]
                    similarity = cosine_similarity(
                        [query_embedding], [dataset_embedding]
                    )[0][0]
                    scores[dataset_id] = float(similarity)
                    
        except Exception as e:
            logger.error(f"Error calculating BERT scores: {e}")
        
        return scores
    
    def _get_metadata_scores(self, query: str, datasets: List[Any]) -> Dict[str, float]:
        """Get metadata-based scores (exact matches in categories, tags, etc.)."""
        scores = {}
        query_lower = query.lower()
        
        for dataset in datasets:
            score = 0.0
            
            # Exact category match
            if hasattr(dataset, 'category') and dataset.category:
                if query_lower in dataset.category.lower():
                    score += 0.5
            
            # Exact data type match
            if hasattr(dataset, 'data_type') and dataset.data_type:
                if query_lower in dataset.data_type.lower():
                    score += 0.3
            
            # Tag matches
            if hasattr(dataset, 'tags') and dataset.tags:
                tags = dataset.tags.split(',') if isinstance(dataset.tags, str) else dataset.tags
                for tag in tags:
                    if query_lower in tag.lower().strip():
                        score += 0.2
            
            scores[str(dataset.id)] = score
        
        return scores
    
    def _get_popularity_scores(self, datasets: List[Any]) -> Dict[str, float]:
        """Get popularity-based scores (based on creation date, views, etc.)."""
        scores = {}
        
        # Simple popularity based on recency (newer datasets get higher scores)
        if datasets:
            sorted_datasets = sorted(datasets, key=lambda d: d.created_at, reverse=True)
            total = len(sorted_datasets)
            
            for i, dataset in enumerate(sorted_datasets):
                # Normalize score between 0 and 1
                popularity_score = (total - i) / total
                scores[str(dataset.id)] = popularity_score
        
        return scores
    
    def _combine_scores(self, datasets: List[Any], tfidf_scores: Dict[str, float], 
                       bert_scores: Dict[str, float], metadata_scores: Dict[str, float],
                       popularity_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Combine different scoring algorithms into final results."""
        results = []
        
        for dataset in datasets:
            dataset_id = str(dataset.id)
            
            # Get individual scores
            tfidf_score = tfidf_scores.get(dataset_id, 0.0)
            bert_score = bert_scores.get(dataset_id, 0.0)
            metadata_score = metadata_scores.get(dataset_id, 0.0)
            popularity_score = popularity_scores.get(dataset_id, 0.0)
            
            # Calculate weighted combined score
            combined_score = (
                tfidf_score * self.weights['tfidf'] +
                bert_score * self.weights['bert'] +
                metadata_score * self.weights['metadata'] +
                popularity_score * self.weights['popularity']
            )
            
            results.append({
                'dataset': dataset,
                'score': combined_score,
                'scores': {
                    'tfidf': tfidf_score,
                    'bert': bert_score,
                    'metadata': metadata_score,
                    'popularity': popularity_score,
                    'combined': combined_score
                }
            })
        
        return results
    
    def _basic_search(self, query: str, datasets: List[Any], limit: int) -> List[Dict[str, Any]]:
        """Fallback basic search when semantic search is not available."""
        results = []
        query_lower = query.lower()
        
        for dataset in datasets:
            score = 0.0
            
            # Simple text matching
            if hasattr(dataset, 'title') and dataset.title:
                if query_lower in dataset.title.lower():
                    score += 0.5
            
            if hasattr(dataset, 'description') and dataset.description:
                if query_lower in dataset.description.lower():
                    score += 0.3
            
            if hasattr(dataset, 'tags') and dataset.tags:
                if query_lower in dataset.tags.lower():
                    score += 0.2
            
            if score > 0:
                results.append({
                    'dataset': dataset,
                    'score': score,
                    'scores': {'basic': score}
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]


# Global service instance
_semantic_search_service = None

def get_semantic_search_service() -> SemanticSearchService:
    """Get or create the semantic search service instance."""
    global _semantic_search_service
    if _semantic_search_service is None:
        _semantic_search_service = SemanticSearchService()
        _semantic_search_service.initialize()
    return _semantic_search_service
