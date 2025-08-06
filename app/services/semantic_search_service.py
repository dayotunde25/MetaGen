"""
Enhanced Semantic Search Service using Advanced NLP techniques for better dataset discovery.

This service implements multiple search algorithms:
- TF-IDF for traditional text matching with n-grams
- BERT embeddings for semantic similarity
- Named Entity Recognition (NER) for entity matching
- Keyword extraction and matching from generated content
- Metadata and description analysis
- Hybrid scoring combining multiple techniques
"""

import os
import pickle
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Try to import NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class EnhancedSemanticSearchService:
    """
    Advanced semantic search service for datasets using multiple NLP techniques including:
    - TF-IDF with n-grams and custom preprocessing
    - BERT embeddings for semantic similarity
    - Named Entity Recognition (NER) for entity matching
    - Keyword extraction and matching
    - Generated content analysis (descriptions, metadata)
    - Multi-field weighted scoring
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the enhanced semantic search service.

        Args:
            cache_dir: Directory to cache models and embeddings
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'app', 'cache', 'search')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize NLP components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bert_model = None
        self.spacy_nlp = None
        self.lemmatizer = None
        self.stop_words = set()

        # Data storage
        self.dataset_embeddings = {}
        self.dataset_texts = {}
        self.dataset_entities = {}
        self.dataset_keywords = {}
        self.dataset_metadata = {}
        self.dataset_order = []  # Track dataset order for TF-IDF matrix
        self.is_initialized = False

        # Enhanced search weights for hybrid scoring
        self.weights = {
            'tfidf': 0.25,          # Traditional text matching
            'bert': 0.30,           # Semantic similarity
            'entities': 0.15,       # Named entity matching
            'keywords': 0.15,       # Generated keyword matching
            'metadata': 0.10,       # Metadata field matching
            'popularity': 0.05      # Recency/popularity boost
        }

        # Field-specific weights for text extraction
        self.field_weights = {
            'title': 3.0,           # Title gets highest weight
            'description': 2.0,     # Description is important
            'generated_description': 2.0,  # Auto-generated descriptions
            'keywords': 1.5,        # Generated keywords
            'tags': 1.5,           # User tags
            'category': 1.0,        # Category classification
            'data_type': 1.0,       # Data type
            'source': 0.5,          # Source information
            'entities': 1.2         # Named entities
        }
    
    def initialize(self):
        """Initialize the enhanced search models and NLP components."""
        try:
            # Initialize NLP libraries
            self._initialize_nlp_components()

            # Load/initialize search models
            self._load_tfidf_model()
            self._load_bert_model()

            # Load cached data if available
            self._load_cached_data()

            self.is_initialized = True
            logger.info("Enhanced semantic search service initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize enhanced semantic search: {e}")
            # Fall back to basic search
            self.is_initialized = False

    def _initialize_nlp_components(self):
        """Initialize NLP libraries and components."""
        # Initialize spaCy for NER - try larger models first
        if SPACY_AVAILABLE:
            try:
                # Try to load the transformer-based model first (most accurate)
                self.spacy_nlp = spacy.load("en_core_web_trf")
                logger.info("Loaded spaCy Transformer model (en_core_web_trf) for NER")
            except OSError:
                try:
                    # Fallback to large model
                    self.spacy_nlp = spacy.load("en_core_web_lg")
                    logger.info("Loaded spaCy Large model (en_core_web_lg) for NER")
                except OSError:
                    try:
                        # Fallback to medium model
                        self.spacy_nlp = spacy.load("en_core_web_md")
                        logger.info("Loaded spaCy Medium model (en_core_web_md) for NER")
                    except OSError:
                        try:
                            # Final fallback to small model
                            self.spacy_nlp = spacy.load("en_core_web_sm")
                            logger.info("Loaded spaCy Small model (en_core_web_sm) for NER")
                        except OSError:
                            logger.warning("No spaCy English model found. Install with:")
                            logger.warning("   python -m spacy download en_core_web_trf  # Best (transformer-based)")
                            logger.warning("   python -m spacy download en_core_web_lg   # Large")
                            logger.warning("   python -m spacy download en_core_web_md   # Medium")
                            logger.warning("   python -m spacy download en_core_web_sm   # Small")
                            self.spacy_nlp = None

        # Initialize NLTK components with offline fallback
        if NLTK_AVAILABLE:
            try:
                # Try to download required NLTK data (with timeout and error handling)
                import socket
                socket.setdefaulttimeout(5)  # 5 second timeout

                try:
                    # Suppress NLTK download error messages
                    import sys
                    from io import StringIO
                    old_stderr = sys.stderr
                    sys.stderr = StringIO()

                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('wordnet', quiet=True)

                    # Restore stderr
                    sys.stderr = old_stderr
                except Exception as download_error:
                    # Restore stderr in case of error
                    sys.stderr = old_stderr
                    logger.warning(f"NLTK data download failed (using offline fallbacks)")
                    # Continue with offline initialization

                # Try to initialize NLTK components
                try:
                    self.stop_words = set(stopwords.words('english'))
                    logger.info("✅ NLTK stopwords loaded successfully")
                except Exception:
                    # Fallback to basic English stopwords
                    self.stop_words = {
                        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                        'further', 'then', 'once'
                    }
                    logger.info("✅ Using fallback English stopwords")

                try:
                    self.lemmatizer = WordNetLemmatizer()
                    logger.info("✅ NLTK lemmatizer initialized")
                except Exception:
                    self.lemmatizer = None
                    logger.info("⚠️ NLTK lemmatizer not available, using basic text processing")

                logger.info("✅ NLTK components initialized (with fallbacks if needed)")
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK: {e}")
                # Set fallback values
                self.stop_words = set()
                self.lemmatizer = None

        # Add custom stop words for dataset search
        custom_stop_words = {
            'dataset', 'data', 'file', 'csv', 'json', 'xml', 'table', 'record', 'field', 'column'
        }
        self.stop_words.update(custom_stop_words)
    
    def _load_tfidf_model(self):
        """Load or initialize enhanced TF-IDF vectorizer with custom preprocessing."""
        try:
            tfidf_path = os.path.join(self.cache_dir, 'enhanced_tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded cached enhanced TF-IDF vectorizer")
            else:
                # Initialize enhanced TF-IDF vectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,          # Increased vocabulary size
                    stop_words=list(self.stop_words) if self.stop_words else 'english',
                    ngram_range=(1, 3),          # Include trigrams for better context
                    min_df=1,                    # Include rare terms (important for datasets)
                    max_df=0.90,                 # Exclude very common terms
                    sublinear_tf=True,           # Use log scaling for term frequency
                    preprocessor=self._preprocess_text,  # Custom preprocessing
                    token_pattern=r'\b[a-zA-Z][a-zA-Z0-9_-]*\b'  # Include underscores and hyphens
                )
                logger.info("Initialized enhanced TF-IDF vectorizer")
        except Exception as e:
            logger.error(f"Error loading enhanced TF-IDF model: {e}")
            # Fallback to basic vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better search results."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace common separators with spaces
        text = re.sub(r'[_-]+', ' ', text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Lemmatize if available
        if self.lemmatizer and NLTK_AVAILABLE:
            try:
                # Try NLTK tokenization
                try:
                    words = word_tokenize(text)
                except Exception:
                    # Fallback to simple tokenization
                    words = text.split()

                words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
                text = ' '.join(words)
            except Exception as e:
                pass  # Fall back to original text if lemmatization fails

        return text.strip()
    
    def _load_bert_model(self):
        """Load BERT model for semantic embeddings."""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not available, skipping BERT initialization")
                self.bert_model = None
                return

            from sentence_transformers import SentenceTransformer

            model_path = os.path.join(self.cache_dir, 'sentence_transformer')
            if os.path.exists(model_path):
                self.bert_model = SentenceTransformer(model_path)
                logger.info("Loaded cached BERT model")
            else:
                # Use a lightweight, fast model optimized for semantic search
                model_name = 'all-MiniLM-L6-v2'  # Fast and efficient
                self.bert_model = SentenceTransformer(model_name)
                # Cache the model
                self.bert_model.save(model_path)
                logger.info(f"Downloaded and cached BERT model: {model_name}")

        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.bert_model = None

    def _load_cached_data(self):
        """Load cached embeddings and processed data."""
        try:
            # Load BERT embeddings
            embeddings_path = os.path.join(self.cache_dir, 'bert_embeddings.pkl')
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'rb') as f:
                    self.dataset_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.dataset_embeddings)} cached BERT embeddings")

            # Load entity cache
            entities_path = os.path.join(self.cache_dir, 'dataset_entities.pkl')
            if os.path.exists(entities_path):
                with open(entities_path, 'rb') as f:
                    self.dataset_entities = pickle.load(f)
                logger.info(f"Loaded {len(self.dataset_entities)} cached entity extractions")

            # Load keywords cache
            keywords_path = os.path.join(self.cache_dir, 'dataset_keywords.pkl')
            if os.path.exists(keywords_path):
                with open(keywords_path, 'rb') as f:
                    self.dataset_keywords = pickle.load(f)
                logger.info(f"Loaded {len(self.dataset_keywords)} cached keyword extractions")

        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
    
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

            # Store dataset order for TF-IDF matrix alignment
            self.dataset_order = dataset_ids.copy()
            
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
        Extract comprehensive searchable text content from a dataset including generated content.

        Args:
            dataset: Dataset object

        Returns:
            Combined weighted text content for search indexing
        """
        weighted_text_parts = []

        # Add title with highest weight
        if hasattr(dataset, 'title') and dataset.title:
            weighted_text_parts.extend([dataset.title] * int(self.field_weights['title']))

        # Add description with high weight
        if hasattr(dataset, 'description') and dataset.description:
            weighted_text_parts.extend([dataset.description] * int(self.field_weights['description']))

        # Add structured/generated description if available
        if hasattr(dataset, 'structured_description') and dataset.structured_description:
            try:
                structured_desc = json.loads(dataset.structured_description)
                if isinstance(structured_desc, dict):
                    # Extract plain text from structured description
                    plain_text = structured_desc.get('plain_text', '')
                    if plain_text:
                        weighted_text_parts.extend([plain_text] * int(self.field_weights['generated_description']))

                    # Extract key insights
                    insights = structured_desc.get('key_insights', [])
                    if insights:
                        weighted_text_parts.extend(insights)

                    # Extract use cases
                    use_cases = structured_desc.get('use_cases', [])
                    if use_cases:
                        weighted_text_parts.extend(use_cases)
            except (json.JSONDecodeError, AttributeError):
                pass

        # Add user tags with medium weight
        if hasattr(dataset, 'tags') and dataset.tags:
            if isinstance(dataset.tags, str):
                tags = [tag.strip() for tag in dataset.tags.split(',') if tag.strip()]
            else:
                tags = dataset.tags if isinstance(dataset.tags, list) else []
            weighted_text_parts.extend(tags * int(self.field_weights['tags']))

        # Add category and data type
        if hasattr(dataset, 'category') and dataset.category:
            weighted_text_parts.extend([dataset.category] * int(self.field_weights['category']))

        if hasattr(dataset, 'data_type') and dataset.data_type:
            weighted_text_parts.extend([dataset.data_type] * int(self.field_weights['data_type']))

        # Add source information
        if hasattr(dataset, 'source') and dataset.source:
            weighted_text_parts.extend([dataset.source] * int(self.field_weights['source']))

        # Extract and add generated keywords from NLP processing
        keywords = self._extract_generated_keywords(dataset)
        if keywords:
            weighted_text_parts.extend(keywords * int(self.field_weights['keywords']))

        # Extract and add named entities
        entities = self._extract_named_entities(dataset)
        if entities:
            weighted_text_parts.extend(entities * int(self.field_weights['entities']))

        return ' '.join(weighted_text_parts)

    def _extract_generated_keywords(self, dataset) -> List[str]:
        """Extract generated keywords from NLP processing results."""
        keywords = []

        # Try to get keywords from NLP service if available
        try:
            from app.services.nlp_service import nlp_service

            # Extract text content for keyword generation
            text_content = ""
            if hasattr(dataset, 'title') and dataset.title:
                text_content += dataset.title + " "
            if hasattr(dataset, 'description') and dataset.description:
                text_content += dataset.description + " "

            if text_content.strip():
                extracted_keywords = nlp_service.extract_keywords(text_content, max_keywords=15)
                keywords.extend(extracted_keywords)

        except Exception as e:
            logger.debug(f"Could not extract keywords for dataset {dataset.id}: {e}")

        return keywords

    def _extract_named_entities(self, dataset) -> List[str]:
        """Extract named entities from dataset content using NER."""
        entities = []

        if not self.spacy_nlp:
            return entities

        try:
            # Combine text for entity extraction
            text_content = ""
            if hasattr(dataset, 'title') and dataset.title:
                text_content += dataset.title + " "
            if hasattr(dataset, 'description') and dataset.description:
                text_content += dataset.description + " "

            if text_content.strip():
                doc = self.spacy_nlp(text_content)

                # Extract relevant entity types for dataset search
                relevant_labels = {'ORG', 'GPE', 'PERSON', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW'}

                for ent in doc.ents:
                    if ent.label_ in relevant_labels and len(ent.text) > 2:
                        entities.append(ent.text.lower())

        except Exception as e:
            logger.debug(f"Could not extract entities for dataset {dataset.id}: {e}")

        return list(set(entities))  # Remove duplicates
    
    def search(self, query: str, datasets: List[Any], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform enhanced semantic search on datasets using multiple NLP techniques.

        Args:
            query: Search query
            datasets: List of dataset objects to search
            limit: Maximum number of results to return

        Returns:
            List of search results with detailed scores
        """
        if not query or not datasets:
            return []

        try:
            # If not initialized, fall back to basic search
            if not self.is_initialized:
                return self._enhanced_basic_search(query, datasets, limit)

            # Preprocess query for better matching
            processed_query = self._preprocess_query(query)

            # Get scores from different algorithms
            tfidf_scores = self._get_tfidf_scores(processed_query, datasets)
            bert_scores = self._get_bert_scores(query, datasets)  # Use original query for BERT
            entity_scores = self._get_entity_scores(query, datasets)
            keyword_scores = self._get_keyword_scores(query, datasets)
            metadata_scores = self._get_metadata_scores(query, datasets)
            popularity_scores = self._get_popularity_scores(datasets)

            # Combine scores using enhanced weighting
            combined_scores = self._combine_enhanced_scores(
                datasets, tfidf_scores, bert_scores, entity_scores,
                keyword_scores, metadata_scores, popularity_scores
            )

            # Sort by combined score and return top results
            results = sorted(combined_scores, key=lambda x: x['score'], reverse=True)

            # Enhanced filtering with relevance threshold and category matching
            filtered_results = self._filter_relevant_results(query, results)

            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Error in enhanced semantic search: {e}")
            return self._enhanced_basic_search(query, datasets, limit)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess search query for better matching."""
        if not query:
            return ""

        # Apply same preprocessing as dataset text
        processed = self._preprocess_text(query)

        # Extract entities from query if possible
        if self.spacy_nlp:
            try:
                doc = self.spacy_nlp(query)
                entities = [ent.text.lower() for ent in doc.ents]
                if entities:
                    # Add entities to processed query for better matching
                    processed += " " + " ".join(entities)
            except:
                pass

        return processed
    
    def _get_tfidf_scores(self, query: str, datasets: List[Any]) -> Dict[str, float]:
        """Get TF-IDF similarity scores for the query."""
        scores = {}

        try:
            if (not self.tfidf_vectorizer or
                self.tfidf_matrix is None or
                not self.dataset_order):
                return scores

            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)

            # Handle the result properly - it's a 2D array
            if similarities.shape[0] > 0:
                similarities_flat = similarities[0]  # Get first row

                # Map similarities to dataset IDs using stored order
                for i, dataset_id in enumerate(self.dataset_order):
                    if i < len(similarities_flat):
                        scores[dataset_id] = float(similarities_flat[i])

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
    
    def _get_entity_scores(self, query: str, datasets: List[Any]) -> Dict[str, float]:
        """Get Named Entity Recognition (NER) based scores."""
        scores = {}

        if not self.spacy_nlp:
            return scores

        try:
            # Extract entities from query
            doc = self.spacy_nlp(query)
            query_entities = set(ent.text.lower() for ent in doc.ents)

            if not query_entities:
                return scores

            for dataset in datasets:
                score = 0.0
                dataset_id = str(dataset.id)

                # Get cached entities or extract them
                if dataset_id in self.dataset_entities:
                    dataset_entities = set(self.dataset_entities[dataset_id])
                else:
                    dataset_entities = set(self._extract_named_entities(dataset))
                    self.dataset_entities[dataset_id] = list(dataset_entities)

                # Calculate entity overlap
                if dataset_entities and query_entities:
                    overlap = len(query_entities.intersection(dataset_entities))
                    if overlap > 0:
                        # Score based on proportion of matching entities
                        score = overlap / len(query_entities)

                scores[dataset_id] = score

        except Exception as e:
            logger.debug(f"Error calculating entity scores: {e}")

        return scores

    def _get_keyword_scores(self, query: str, datasets: List[Any]) -> Dict[str, float]:
        """Get keyword-based scores using generated keywords."""
        scores = {}

        try:
            # Extract keywords from query
            from app.services.nlp_service import nlp_service
            query_keywords = set(keyword.lower() for keyword in nlp_service.extract_keywords(query, max_keywords=10))

            if not query_keywords:
                return scores

            for dataset in datasets:
                score = 0.0
                dataset_id = str(dataset.id)

                # Get cached keywords or extract them
                if dataset_id in self.dataset_keywords:
                    dataset_keywords = set(self.dataset_keywords[dataset_id])
                else:
                    dataset_keywords = set(keyword.lower() for keyword in self._extract_generated_keywords(dataset))
                    self.dataset_keywords[dataset_id] = list(dataset_keywords)

                # Calculate keyword overlap
                if dataset_keywords and query_keywords:
                    overlap = len(query_keywords.intersection(dataset_keywords))
                    if overlap > 0:
                        # Score based on proportion of matching keywords
                        score = overlap / len(query_keywords)

                scores[dataset_id] = score

        except Exception as e:
            logger.debug(f"Error calculating keyword scores: {e}")

        return scores

    def _get_metadata_scores(self, query: str, datasets: List[Any]) -> Dict[str, float]:
        """Get enhanced metadata-based scores (exact matches in categories, tags, etc.)."""
        scores = {}
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for dataset in datasets:
            score = 0.0

            # Exact category match
            if hasattr(dataset, 'category') and dataset.category:
                category_words = set(dataset.category.lower().split())
                if query_words.intersection(category_words):
                    score += 0.5

            # Exact data type match
            if hasattr(dataset, 'data_type') and dataset.data_type:
                datatype_words = set(dataset.data_type.lower().split())
                if query_words.intersection(datatype_words):
                    score += 0.3

            # Tag matches (improved)
            if hasattr(dataset, 'tags') and dataset.tags:
                tags = dataset.tags.split(',') if isinstance(dataset.tags, str) else dataset.tags
                tag_words = set()
                for tag in tags:
                    tag_words.update(tag.lower().strip().split())

                if query_words.intersection(tag_words):
                    score += 0.4

            # Source matches
            if hasattr(dataset, 'source') and dataset.source:
                if query_lower in dataset.source.lower():
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
    
    def _combine_enhanced_scores(self, datasets: List[Any], tfidf_scores: Dict[str, float],
                               bert_scores: Dict[str, float], entity_scores: Dict[str, float],
                               keyword_scores: Dict[str, float], metadata_scores: Dict[str, float],
                               popularity_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Combine different scoring algorithms into final results with enhanced weighting."""
        results = []

        for dataset in datasets:
            dataset_id = str(dataset.id)

            # Get individual scores
            tfidf_score = tfidf_scores.get(dataset_id, 0.0)
            bert_score = bert_scores.get(dataset_id, 0.0)
            entity_score = entity_scores.get(dataset_id, 0.0)
            keyword_score = keyword_scores.get(dataset_id, 0.0)
            metadata_score = metadata_scores.get(dataset_id, 0.0)
            popularity_score = popularity_scores.get(dataset_id, 0.0)

            # Calculate weighted combined score
            combined_score = (
                tfidf_score * self.weights['tfidf'] +
                bert_score * self.weights['bert'] +
                entity_score * self.weights['entities'] +
                keyword_score * self.weights['keywords'] +
                metadata_score * self.weights['metadata'] +
                popularity_score * self.weights['popularity']
            )

            # Apply boost for high-confidence matches
            if entity_score > 0.5 or keyword_score > 0.5:
                combined_score *= 1.2  # 20% boost for strong entity/keyword matches

            if tfidf_score > 0.7 and bert_score > 0.7:
                combined_score *= 1.15  # 15% boost for strong text and semantic matches

            results.append({
                'dataset': dataset,
                'score': combined_score,
                'scores': {
                    'tfidf': tfidf_score,
                    'bert': bert_score,
                    'entities': entity_score,
                    'keywords': keyword_score,
                    'metadata': metadata_score,
                    'popularity': popularity_score,
                    'combined': combined_score
                }
            })

        return results
    
    def _enhanced_basic_search(self, query: str, datasets: List[Any], limit: int) -> List[Dict[str, Any]]:
        """Enhanced fallback search when full semantic search is not available."""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for dataset in datasets:
            score = 0.0

            # Enhanced title matching with word-level scoring
            if hasattr(dataset, 'title') and dataset.title:
                title_words = set(dataset.title.lower().split())
                title_overlap = len(query_words.intersection(title_words))
                if title_overlap > 0:
                    score += 0.6 * (title_overlap / len(query_words))

                # Exact phrase matching gets bonus
                if query_lower in dataset.title.lower():
                    score += 0.3

            # Enhanced description matching
            if hasattr(dataset, 'description') and dataset.description:
                desc_words = set(dataset.description.lower().split())
                desc_overlap = len(query_words.intersection(desc_words))
                if desc_overlap > 0:
                    score += 0.4 * (desc_overlap / len(query_words))

                if query_lower in dataset.description.lower():
                    score += 0.2

            # Enhanced tag matching
            if hasattr(dataset, 'tags') and dataset.tags:
                tags = dataset.tags.split(',') if isinstance(dataset.tags, str) else dataset.tags
                tag_words = set()
                for tag in tags:
                    tag_words.update(tag.lower().strip().split())

                tag_overlap = len(query_words.intersection(tag_words))
                if tag_overlap > 0:
                    score += 0.5 * (tag_overlap / len(query_words))

            # Category and data type matching
            if hasattr(dataset, 'category') and dataset.category:
                if query_lower in dataset.category.lower():
                    score += 0.3

            if hasattr(dataset, 'data_type') and dataset.data_type:
                if query_lower in dataset.data_type.lower():
                    score += 0.3

            # Generated content matching (if available)
            if hasattr(dataset, 'structured_description') and dataset.structured_description:
                try:
                    structured_desc = json.loads(dataset.structured_description)
                    if isinstance(structured_desc, dict):
                        plain_text = structured_desc.get('plain_text', '').lower()
                        if query_lower in plain_text:
                            score += 0.25
                except:
                    pass

            if score > 0:
                results.append({
                    'dataset': dataset,
                    'score': score,
                    'scores': {'enhanced_basic': score}
                })

        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

    def _filter_relevant_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter search results to ensure relevance and add quality scores.

        Args:
            query: Original search query
            results: List of search results

        Returns:
            Filtered and enhanced results
        """
        if not results:
            return []

        # Extract query keywords and categories for relevance checking
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Define category keywords for better filtering
        category_keywords = {
            'education': {'student', 'school', 'university', 'academic', 'education', 'learning', 'course', 'grade', 'exam', 'teacher', 'classroom'},
            'health': {'patient', 'medical', 'health', 'hospital', 'disease', 'treatment', 'medicine', 'clinical', 'diagnosis', 'therapy'},
            'finance': {'financial', 'money', 'bank', 'investment', 'stock', 'market', 'economy', 'revenue', 'profit', 'budget'},
            'technology': {'software', 'hardware', 'computer', 'tech', 'digital', 'programming', 'code', 'system', 'network', 'data'},
            'business': {'company', 'business', 'corporate', 'enterprise', 'management', 'sales', 'customer', 'product', 'service', 'marketing'},
            'science': {'research', 'experiment', 'scientific', 'study', 'analysis', 'laboratory', 'hypothesis', 'theory', 'methodology', 'publication'},
            'government': {'government', 'public', 'policy', 'administration', 'citizen', 'municipal', 'federal', 'state', 'regulation', 'law'},
            'social': {'social', 'community', 'demographic', 'population', 'survey', 'behavior', 'culture', 'society', 'people', 'human'}
        }

        # Determine query category
        query_category = None
        max_category_matches = 0

        for category, keywords in category_keywords.items():
            matches = len(query_words.intersection(keywords))
            if matches > max_category_matches:
                max_category_matches = matches
                query_category = category

        filtered_results = []

        for result in results:
            dataset = result['dataset']
            score = result['score']

            # Basic relevance threshold
            if score < 0.05:  # Increased threshold for better relevance
                continue

            # Category relevance check
            if query_category and max_category_matches >= 2:  # Strong category signal
                dataset_relevant = self._check_dataset_category_relevance(dataset, query_category, category_keywords[query_category])
                if not dataset_relevant:
                    continue  # Skip irrelevant datasets

            # Add quality score to result
            quality_score = self._get_dataset_quality_score(dataset)

            # Enhanced result with quality information
            enhanced_result = {
                'dataset': dataset,
                'score': score,
                'scores': result.get('scores', {}),
                'quality_score': quality_score,
                'relevance_category': query_category,
                'relevance_explanation': self._generate_relevance_explanation(dataset, query, score)
            }

            filtered_results.append(enhanced_result)

        # If no results after filtering and we had a strong category signal,
        # return empty results instead of irrelevant ones
        if not filtered_results and query_category and max_category_matches >= 2:
            return []

        # If we filtered too aggressively, return top results with lower threshold
        if not filtered_results and results:
            for result in results[:5]:  # Return top 5 with lower threshold
                if result['score'] > 0.01:
                    dataset = result['dataset']
                    quality_score = self._get_dataset_quality_score(dataset)

                    enhanced_result = {
                        'dataset': dataset,
                        'score': result['score'],
                        'scores': result.get('scores', {}),
                        'quality_score': quality_score,
                        'relevance_category': None,
                        'relevance_explanation': 'Low relevance match'
                    }
                    filtered_results.append(enhanced_result)

        return filtered_results

    def _check_dataset_category_relevance(self, dataset: Any, query_category: str, category_keywords: set) -> bool:
        """
        Check if a dataset is relevant to the query category.

        Args:
            dataset: Dataset object
            query_category: Detected query category
            category_keywords: Keywords for the category

        Returns:
            True if dataset is relevant to the category
        """
        try:
            # Combine all dataset text for analysis
            dataset_text = ""

            if hasattr(dataset, 'title') and dataset.title:
                dataset_text += dataset.title.lower() + " "

            if hasattr(dataset, 'description') and dataset.description:
                dataset_text += dataset.description.lower() + " "

            if hasattr(dataset, 'category') and dataset.category:
                dataset_text += dataset.category.lower() + " "

            if hasattr(dataset, 'tags') and dataset.tags:
                dataset_text += dataset.tags.lower() + " "

            if hasattr(dataset, 'keywords') and dataset.keywords:
                dataset_text += dataset.keywords.lower() + " "

            # Check for category keyword matches
            dataset_words = set(dataset_text.split())
            matches = len(dataset_words.intersection(category_keywords))

            # Consider relevant if at least 1 category keyword matches
            return matches >= 1

        except Exception as e:
            logger.debug(f"Error checking category relevance: {e}")
            return True  # Default to relevant if error occurs

    def _get_dataset_quality_score(self, dataset: Any) -> float:
        """
        Get the quality score for a dataset.

        Args:
            dataset: Dataset object

        Returns:
            Quality score (0-100)
        """
        try:
            # Try to get quality score from metadata first
            if hasattr(dataset, 'quality_score') and dataset.quality_score:
                return float(dataset.quality_score)

            # Calculate basic quality score based on available information
            score = 0.0

            # Title quality (20 points)
            if hasattr(dataset, 'title') and dataset.title:
                if len(dataset.title) > 10:
                    score += 20
                elif len(dataset.title) > 5:
                    score += 15
                else:
                    score += 10

            # Description quality (30 points)
            if hasattr(dataset, 'description') and dataset.description:
                if len(dataset.description) > 200:
                    score += 30
                elif len(dataset.description) > 100:
                    score += 25
                elif len(dataset.description) > 50:
                    score += 20
                else:
                    score += 15

            # Metadata completeness (25 points)
            metadata_fields = ['source', 'category', 'tags', 'keywords', 'data_type']
            filled_fields = sum(1 for field in metadata_fields
                              if hasattr(dataset, field) and getattr(dataset, field))
            score += (filled_fields / len(metadata_fields)) * 25

            # Data structure quality (25 points)
            if hasattr(dataset, 'record_count') and dataset.record_count:
                if dataset.record_count > 1000:
                    score += 15
                elif dataset.record_count > 100:
                    score += 12
                else:
                    score += 8

            if hasattr(dataset, 'field_count') and dataset.field_count:
                if dataset.field_count > 10:
                    score += 10
                elif dataset.field_count > 5:
                    score += 8
                else:
                    score += 5

            return min(100.0, score)

        except Exception as e:
            logger.debug(f"Error calculating quality score: {e}")
            return 75.0  # Default quality score

    def _generate_relevance_explanation(self, dataset: Any, query: str, score: float) -> str:
        """
        Generate an explanation of why this dataset is relevant to the query.

        Args:
            dataset: Dataset object
            query: Search query
            score: Relevance score

        Returns:
            Relevance explanation string
        """
        try:
            explanations = []

            # Check title match
            if hasattr(dataset, 'title') and dataset.title:
                if any(word.lower() in dataset.title.lower() for word in query.split()):
                    explanations.append("title match")

            # Check description match
            if hasattr(dataset, 'description') and dataset.description:
                if any(word.lower() in dataset.description.lower() for word in query.split()):
                    explanations.append("description match")

            # Check category match
            if hasattr(dataset, 'category') and dataset.category:
                if any(word.lower() in dataset.category.lower() for word in query.split()):
                    explanations.append("category match")

            # Check tags match
            if hasattr(dataset, 'tags') and dataset.tags:
                if any(word.lower() in dataset.tags.lower() for word in query.split()):
                    explanations.append("tags match")

            if explanations:
                return f"Relevant due to: {', '.join(explanations)}"
            elif score > 0.5:
                return "High semantic similarity"
            elif score > 0.2:
                return "Moderate semantic similarity"
            else:
                return "Low relevance match"

        except Exception as e:
            logger.debug(f"Error generating relevance explanation: {e}")
            return "Relevance determined by semantic analysis"

    @staticmethod
    def add_quality_scores_to_datasets(datasets: List[Any]) -> List[Dict[str, Any]]:
        """
        Add quality scores to a list of datasets for display.

        Args:
            datasets: List of dataset objects

        Returns:
            List of enhanced results with quality scores
        """
        enhanced_results = []

        for dataset in datasets:
            # Create a temporary search service instance to use quality scoring
            temp_service = EnhancedSemanticSearchService()
            quality_score = temp_service._get_dataset_quality_score(dataset)

            enhanced_result = {
                'dataset': dataset,
                'score': 0.0,
                'quality_score': quality_score,
                'relevance_explanation': 'Dataset listing'
            }
            enhanced_results.append(enhanced_result)

        return enhanced_results


# Global service instance
_semantic_search_service = None

def get_semantic_search_service() -> EnhancedSemanticSearchService:
    """Get or create the enhanced semantic search service instance."""
    global _semantic_search_service
    if _semantic_search_service is None:
        _semantic_search_service = EnhancedSemanticSearchService()
        _semantic_search_service.initialize()
    return _semantic_search_service
