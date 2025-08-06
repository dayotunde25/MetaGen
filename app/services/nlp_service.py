"""
NLP Service for dataset processing and analysis.
"""

import re
import json
import os
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
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, pipeline, T5Tokenizer, T5ForConditionalGeneration
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import free AI service for enhanced description generation
try:
    from app.services.free_ai_service import free_ai_service
    FREE_AI_AVAILABLE = True
except ImportError:
    FREE_AI_AVAILABLE = False

# Focus on free AI models with FLAN-T5 Base as fallback


class NLPService:
    """Service for NLP processing of dataset content and metadata"""
    
    def __init__(self):
        self.nlp = None
        self.stemmer = None
        self.stop_words = set()
        self.tfidf_vectorizer = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.ner_pipeline = None

        # T5 model for offline description generation
        self.t5_tokenizer = None
        self.t5_model = None

        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize advanced NLP libraries if available"""
        print("🔧 Initializing Advanced NLP Service with BERT, TF-IDF, and NER...")

        # Initialize spaCy if available - try larger models first
        if SPACY_AVAILABLE:
            try:
                # Try to load the transformer-based model first (most accurate)
                self.nlp = spacy.load("en_core_web_trf")
                print("✅ spaCy Transformer model (en_core_web_trf) loaded successfully")
            except OSError:
                try:
                    # Fallback to large model
                    self.nlp = spacy.load("en_core_web_lg")
                    print("✅ spaCy Large model (en_core_web_lg) loaded successfully")
                except OSError:
                    try:
                        # Fallback to medium model
                        self.nlp = spacy.load("en_core_web_md")
                        print("✅ spaCy Medium model (en_core_web_md) loaded successfully")
                    except OSError:
                        try:
                            # Final fallback to small model
                            self.nlp = spacy.load("en_core_web_sm")
                            print("✅ spaCy Small model (en_core_web_sm) loaded successfully")
                        except OSError:
                            print("⚠️ No spaCy English model found. Install with:")
                            print("   python -m spacy download en_core_web_trf  # Best (transformer-based)")
                            print("   python -m spacy download en_core_web_lg   # Large")
                            print("   python -m spacy download en_core_web_md   # Medium")
                            print("   python -m spacy download en_core_web_sm   # Small")
                            self.nlp = None

        # Initialize NLTK if available with robust offline fallback
        if NLTK_AVAILABLE:
            try:
                # Set shorter timeout for NLTK downloads
                import socket
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(3)  # 3 second timeout

                # Try to download NLTK data with better error handling
                nltk_data_downloaded = False
                try:
                    # Check if data already exists before downloading
                    try:
                        nltk.data.find('tokenizers/punkt')
                        nltk.data.find('corpora/stopwords')
                        nltk.data.find('corpora/wordnet')
                        nltk_data_downloaded = True
                        print("✅ NLTK data already available")
                    except LookupError:
                        # Data not found, try to download (suppress error output)
                        print("🔄 Downloading NLTK data...")
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

                            nltk_data_downloaded = True
                            print("✅ NLTK data downloaded successfully")
                        except Exception as download_error:
                            # Restore stderr in case of error
                            sys.stderr = old_stderr
                            print(f"⚠️ NLTK data download failed (using offline fallbacks)")
                            nltk_data_downloaded = False
                except Exception as download_error:
                    print(f"⚠️ NLTK data download failed (using offline fallbacks)")
                    nltk_data_downloaded = False
                finally:
                    # Restore original timeout
                    socket.setdefaulttimeout(original_timeout)

                # Initialize NLTK components with fallbacks
                try:
                    self.stemmer = PorterStemmer()
                    print("✅ NLTK Porter Stemmer initialized")
                except Exception:
                    self.stemmer = None
                    print("⚠️ NLTK Porter Stemmer not available")

                # Initialize stopwords with robust fallback
                if nltk_data_downloaded:
                    try:
                        self.stop_words = set(stopwords.words('english'))
                        print("✅ NLTK stopwords loaded successfully")
                    except Exception as e:
                        print(f"⚠️ NLTK stopwords failed to load: {e}")
                        self.stop_words = self._get_fallback_stopwords()
                        print("✅ Using fallback English stopwords")
                else:
                    self.stop_words = self._get_fallback_stopwords()
                    print("✅ Using fallback English stopwords (offline mode)")

                print("✅ NLTK initialized successfully (with fallbacks if needed)")
            except Exception as e:
                print(f"⚠️ Failed to initialize NLTK: {e}")
                self.stemmer = None
                self.stop_words = set()

        # Initialize TF-IDF Vectorizer
        if SKLEARN_AVAILABLE:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=500,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,  # Minimum document frequency
                    max_df=0.99,  # Maximum document frequency
                    lowercase=True,
                    token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens with 2+ chars
                )
                print("✅ TF-IDF Vectorizer initialized successfully")
            except Exception as e:
                print(f"⚠️ TF-IDF initialization failed: {e}")
                self.tfidf_vectorizer = None

        # Initialize BERT model for embeddings - try offline first, then online
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to load cached models first for offline capability
                models_to_try = [
                    ("roberta-large", "RoBERTa-Large"),
                    ("bert-large-uncased", "BERT-Large"),
                    ("bert-base-uncased", "BERT-Base"),
                    ("distilbert-base-uncased", "DistilBERT")
                ]

                model_loaded = False
                for model_name, display_name in models_to_try:
                    try:
                        # Try offline first
                        try:
                            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                local_files_only=True,
                                cache_dir=None  # Use default cache
                            )
                            self.bert_model = AutoModel.from_pretrained(
                                model_name,
                                local_files_only=True,
                                cache_dir=None  # Use default cache
                            )
                            print(f"✅ {display_name} model loaded from cache (offline)")
                            model_loaded = True
                            break
                        except Exception as offline_e:
                            # Try online if offline fails
                            print(f"⚠️ {display_name} offline failed, trying online...")
                            try:
                                self.bert_tokenizer = AutoTokenizer.from_pretrained(
                                    model_name,
                                    local_files_only=False
                                )
                                self.bert_model = AutoModel.from_pretrained(
                                    model_name,
                                    local_files_only=False
                                )
                                print(f"✅ {display_name} model loaded online and cached")
                                model_loaded = True
                                break
                            except Exception as online_e:
                                print(f"⚠️ {display_name} failed both offline and online: {online_e}")
                                continue
                    except Exception as e:
                        print(f"⚠️ {display_name} completely failed: {e}")
                        continue

                if not model_loaded:
                    print("⚠️ All BERT models failed, running without BERT embeddings")
                    self.bert_tokenizer = None
                    self.bert_model = None

                # Initialize advanced NER pipeline with offline-first approach
                self.ner_pipeline = self._initialize_ner_pipeline()

            except Exception as e:
                print(f"⚠️ BERT/NER initialization failed: {e}")
                self.bert_tokenizer = None
                self.bert_model = None
                self.ner_pipeline = None

        # Initialize T5 model for offline description generation
        self._initialize_t5_model()

        print("🚀 Advanced NLP Service initialization complete!")



    def _try_load_cached_models(self):
        """Try to load cached/offline models"""
        try:
            # Try to load any cached models from local directories
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
            if os.path.exists(cache_dir):
                print(f"🔍 Looking for cached models in {cache_dir}")
                # Try to load the smallest available model
                try:
                    model_name = "distilbert-base-uncased"
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    self.bert_model = AutoModel.from_pretrained(model_name, local_files_only=True)
                    print("✅ Loaded cached DistilBERT model")
                except:
                    print("⚠️ No cached models available - running without BERT embeddings")
            else:
                print("⚠️ No cache directory found - running without BERT embeddings")
        except Exception as e:
            print(f"⚠️ Failed to load cached models: {e}")

    def _initialize_t5_model(self):
        """Initialize FLAN-T5 Base model with proper offline support"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available, skipping T5 initialization")
            return

        # Models to try in order of preference
        models_to_try = [
            ("google/flan-t5-base", "FLAN-T5 Base", True),  # Use legacy=False for FLAN-T5
            ("t5-small", "T5-small", False)  # Use legacy=True for T5-small (default)
        ]

        for model_name, display_name, use_new_tokenizer in models_to_try:
            try:
                print(f"🔄 Trying {display_name} model: {model_name}")

                # Try offline first
                try:
                    print(f"   Attempting offline load for {display_name}...")
                    if use_new_tokenizer:
                        self.t5_tokenizer = T5Tokenizer.from_pretrained(
                            model_name,
                            local_files_only=True,
                            legacy=False,
                            cache_dir=None
                        )
                    else:
                        self.t5_tokenizer = T5Tokenizer.from_pretrained(
                            model_name,
                            local_files_only=True,
                            cache_dir=None
                        )

                    self.t5_model = T5ForConditionalGeneration.from_pretrained(
                        model_name,
                        local_files_only=True,
                        cache_dir=None
                    )
                    print(f"✅ {display_name} loaded from cache (offline)")
                    return  # Success, exit function

                except Exception as offline_e:
                    print(f"   Offline load failed: {offline_e}")
                    print(f"   Attempting online load for {display_name}...")

                    # Try online
                    try:
                        if use_new_tokenizer:
                            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                                model_name,
                                local_files_only=False,
                                legacy=False
                            )
                        else:
                            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                                model_name,
                                local_files_only=False
                            )

                        self.t5_model = T5ForConditionalGeneration.from_pretrained(
                            model_name,
                            local_files_only=False
                        )
                        print(f"✅ {display_name} loaded online and cached")
                        return  # Success, exit function

                    except Exception as online_e:
                        print(f"   Online load failed: {online_e}")
                        continue  # Try next model

            except Exception as e:
                print(f"⚠️ {display_name} completely failed: {e}")
                continue  # Try next model

        # If we get here, all models failed
        print("❌ All T5 models failed to load")
        self.t5_tokenizer = None
        self.t5_model = None

    def _initialize_ner_pipeline(self):
        """Initialize NER pipeline with offline-first approach"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available, skipping NER initialization")
            return None

        # NER models to try in order of preference (offline first, then online)
        ner_models = [
            ("dbmdz/bert-large-cased-finetuned-conll03-english", "BERT-Large NER"),
            ("dbmdz/bert-base-cased-finetuned-conll03-english", "BERT-Base NER"),
            ("distilbert-base-cased", "DistilBERT NER")
        ]

        for model_name, display_name in ner_models:
            try:
                print(f"🔄 Trying {display_name}: {model_name}")

                # Try offline first
                try:
                    print(f"   Attempting offline load for {display_name}...")
                    # Load tokenizer and model separately for offline use
                    from transformers import AutoTokenizer, AutoModelForTokenClassification
                    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)

                    ner_pipeline = pipeline(
                        "ner",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="simple"
                    )
                    print(f"✅ {display_name} loaded from cache (offline)")
                    return ner_pipeline

                except Exception as offline_e:
                    print(f"   Offline load failed: {offline_e}")
                    print(f"   Attempting online load for {display_name}...")

                    # Try online
                    try:
                        ner_pipeline = pipeline(
                            "ner",
                            model=model_name,
                            aggregation_strategy="simple"
                        )
                        print(f"✅ {display_name} loaded online and cached")
                        return ner_pipeline

                    except Exception as online_e:
                        print(f"   Online load failed: {online_e}")
                        continue  # Try next model

            except Exception as e:
                print(f"⚠️ {display_name} completely failed: {e}")
                continue  # Try next model

        # If all NER models fail, try spaCy NER as fallback
        print("🔄 All transformer NER models failed, trying spaCy NER...")
        try:
            if self.nlp:  # spaCy model is available
                print("✅ Using spaCy NER as fallback")
                return "spacy"  # Special marker for spaCy NER
        except Exception as e:
            print(f"⚠️ spaCy NER fallback failed: {e}")

        print("❌ All NER models failed to load")
        return None

    def generate_description_with_t5(self, dataset_info: str) -> str:
        """Generate description using T5 model (offline alternative)"""
        if not self.t5_model or not self.t5_tokenizer:
            return ""

        try:
            # Prepare input for T5
            input_text = f"summarize: {dataset_info}"

            # Tokenize input
            inputs = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

            # Generate description
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs,
                    max_length=200,
                    min_length=50,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode output
            description = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("✅ T5 description generated successfully")
            return description

        except Exception as e:
            print(f"⚠️ T5 description generation failed: {e}")
            return ""

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using advanced NLP techniques (TF-IDF, BERT, spaCy)"""
        if not text:
            return []

        # Handle non-string inputs
        if not isinstance(text, str):
            if text is None:
                return []
            text = str(text)

        # Clean text
        text = self._clean_text(text)

        print(f"🔍 Extracting keywords using advanced NLP techniques...")

        # Try TF-IDF first for best results
        if self.tfidf_vectorizer is not None:
            keywords = self._extract_keywords_tfidf(text, max_keywords)
            if keywords:
                print(f"✅ TF-IDF extracted {len(keywords)} keywords")
                return keywords

        # Fallback to spaCy
        if self.nlp:
            keywords = self._extract_keywords_spacy(text, max_keywords)
            print(f"✅ spaCy extracted {len(keywords)} keywords")
            return keywords
        elif NLTK_AVAILABLE:
            keywords = self._extract_keywords_nltk(text, max_keywords)
            print(f"✅ NLTK extracted {len(keywords)} keywords")
            return keywords
        else:
            keywords = self._extract_keywords_simple(text, max_keywords)
            print(f"✅ Simple extraction found {len(keywords)} keywords")
            return keywords
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Handle non-string inputs
        if not isinstance(text, str):
            if text is None:
                return ""
            # Convert to string if it's not already
            text = str(text)

        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def _extract_keywords_tfidf(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using TF-IDF vectorization"""
        try:
            # Clean and prepare text
            if len(text.strip()) < 10:
                print("⚠️ Text too short for TF-IDF analysis")
                return []

            # Create a more flexible TF-IDF vectorizer for this specific text
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Split text into sentences for better TF-IDF analysis
            print(f"🔍 TF-IDF: Processing {len(text)} characters")

            if NLTK_AVAILABLE:
                try:
                    sentences = sent_tokenize(text)
                except Exception:
                    # Fallback to simple sentence splitting
                    sentences = [s.strip() for s in text.split('.') if s.strip()]
            else:
                sentences = [s.strip() for s in text.split('.') if s.strip()]

            # Filter out very short sentences
            sentences = [s for s in sentences if len(s.strip()) > 5]
            print(f"🔍 TF-IDF: Initial sentences: {len(sentences)}")

            # For large texts, create more documents by intelligent chunking
            if len(text) > 10000:
                print("🔍 TF-IDF: Large text detected, creating intelligent chunks")
                # Create chunks of reasonable size for TF-IDF
                chunk_size = min(1000, max(200, len(text) // 100))  # Aim for ~100 chunks
                text_chunks = []

                # Split by paragraphs first if available
                paragraphs = text.split('\n\n')
                if len(paragraphs) > 10:
                    # Use paragraphs as documents
                    text_chunks = [p.strip() for p in paragraphs if len(p.strip()) > 20]
                else:
                    # Create fixed-size chunks
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i+chunk_size]
                        if len(chunk.strip()) > 50:
                            text_chunks.append(chunk.strip())

                if len(text_chunks) > len(sentences):
                    sentences = text_chunks
                    print(f"🔍 TF-IDF: Created {len(sentences)} intelligent chunks")

            # If we still have very few sentences, split by other delimiters
            elif len(sentences) < 5:
                print("🔍 TF-IDF: Few sentences, trying additional delimiters")
                additional_sentences = []
                for delimiter in ['\n', ';', '!', '?', ':', ',']:
                    for sentence in sentences:
                        parts = [s.strip() for s in sentence.split(delimiter) if len(s.strip()) > 8]
                        additional_sentences.extend(parts)
                if len(additional_sentences) > len(sentences):
                    sentences = additional_sentences
                    print(f"🔍 TF-IDF: After delimiters: {len(sentences)} sentences")

            # Need at least 2 documents for TF-IDF
            if len(sentences) < 2:
                print("🔍 TF-IDF: Still too few sentences, creating fallback chunks")
                # Split text into chunks if it's long enough
                if len(text) > 100:
                    chunk_size = max(50, len(text) // 5)  # Create 5 chunks minimum
                    sentences = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                else:
                    # For very short text, create artificial documents
                    words = text.split()
                    if len(words) > 4:
                        mid = len(words) // 2
                        sentences = [' '.join(words[:mid]), ' '.join(words[mid:])]
                    else:
                        sentences = [text, text]  # Last resort

            # Ensure we have valid sentences
            sentences = [s for s in sentences if len(s.strip()) > 3]
            if len(sentences) < 2:
                print("⚠️ Unable to create sufficient documents for TF-IDF, using frequency method")
                return self._extract_keywords_frequency(text, max_keywords)

            # Create a more robust TF-IDF vectorizer with very lenient parameters
            total_words = len(' '.join(sentences).split())
            print(f"🔍 TF-IDF setup: {len(sentences)} sentences, {total_words} total words")

            custom_vectorizer = TfidfVectorizer(
                max_features=min(2000, max(50, total_words // 2)),  # Even more generous max features
                stop_words=None,  # Don't filter stop words initially
                ngram_range=(1, 1),  # Start with unigrams only for better vocabulary
                min_df=1,  # Very lenient minimum document frequency
                max_df=0.98,  # Allow even more common terms
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b',  # Simpler pattern, min 2 chars
                sublinear_tf=True,  # Use sublinear TF scaling
                smooth_idf=True,  # Smooth IDF weights
                use_idf=True,  # Use inverse document frequency
                strip_accents='unicode',  # Handle accented characters
                analyzer='word'  # Explicit word analysis
            )

            # Fit TF-IDF on sentences with error handling
            try:
                tfidf_matrix = custom_vectorizer.fit_transform(sentences)
                feature_names = custom_vectorizer.get_feature_names_out()

                if len(feature_names) == 0:
                    print("⚠️ TF-IDF produced empty vocabulary, using frequency method")
                    return self._extract_keywords_frequency(text, max_keywords)

            except ValueError as ve:
                if any(phrase in str(ve).lower() for phrase in ["empty vocabulary", "no terms remain", "after pruning"]):
                    print("⚠️ TF-IDF vocabulary empty, using frequency method")
                    return self._extract_keywords_frequency(text, max_keywords)
                else:
                    raise ve

            # Get average TF-IDF scores across all sentences
            if tfidf_matrix.shape[0] > 1:
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            else:
                mean_scores = tfidf_matrix.toarray()[0]

            # Get top keywords by TF-IDF score with very low threshold
            top_indices = mean_scores.argsort()[-max_keywords*3:][::-1]  # Get more candidates
            keywords = []

            # Basic English stop words to filter out
            basic_stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}

            for i in top_indices:
                if i < len(feature_names) and mean_scores[i] > 0:  # Any positive score
                    keyword = feature_names[i].lower()
                    # Filter out stop words and very short words
                    if (len(keyword) > 2 and
                        not keyword.isdigit() and
                        keyword not in basic_stop_words and
                        keyword.isalpha()):
                        keywords.append(keyword)

            # If we still don't have enough keywords, be even more lenient
            if len(keywords) < max_keywords // 2:
                keywords = []
                for i in top_indices:
                    if i < len(feature_names) and mean_scores[i] > 0:
                        keyword = feature_names[i].lower()
                        if len(keyword) > 1 and not keyword.isdigit():
                            keywords.append(keyword)

            print(f"✅ TF-IDF analysis: processed {len(sentences)} documents, found {len(keywords)} keywords")
            return keywords[:max_keywords] if keywords else self._extract_keywords_frequency(text, max_keywords)

        except ValueError as ve:
            if any(phrase in str(ve).lower() for phrase in ["after pruning", "no terms remain", "empty vocabulary"]):
                print("⚠️ TF-IDF failed: No valid terms after filtering. Falling back to simple word frequency.")
                return self._extract_keywords_frequency(text, max_keywords)
            else:
                print(f"⚠️ TF-IDF extraction failed: {ve}")
                return self._extract_keywords_frequency(text, max_keywords)
        except Exception as e:
            print(f"⚠️ TF-IDF extraction failed: {e}")
            return self._extract_keywords_frequency(text, max_keywords)

    def _extract_keywords_frequency(self, text: str, max_keywords: int) -> List[str]:
        """Fallback keyword extraction using simple word frequency"""
        try:
            import re
            from collections import Counter

            # Clean text and extract words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

            # Remove common stop words
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
            words = [w for w in words if w not in stop_words and len(w) > 2]

            # Count word frequencies
            word_counts = Counter(words)

            # Get most common words
            keywords = [word for word, count in word_counts.most_common(max_keywords)]

            print(f"🔍 Frequency analysis: found {len(keywords)} keywords from {len(words)} words")
            return keywords

        except Exception as e:
            print(f"⚠️ Frequency analysis failed: {e}")
            return []

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

    def _count_sentences(self, text: str) -> int:
        """Count sentences with NLTK fallback"""
        if NLTK_AVAILABLE:
            try:
                return len(sent_tokenize(text))
            except Exception:
                pass

        # Fallback to simple sentence counting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return len(sentences)
    
    def _extract_keywords_nltk(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using NLTK with fallback"""
        try:
            # Try NLTK tokenization first
            try:
                tokens = word_tokenize(text)
            except Exception:
                # Fallback to simple tokenization
                tokens = text.lower().split()
            
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
        if not isinstance(text, str):
            if text is None:
                return []
            text = str(text)

        keywords = self.extract_keywords(text, num_tags * 2)
        
        # Filter and clean keywords for tags
        tags = []
        for keyword in keywords:
            if len(keyword) > 2 and keyword.isalpha():
                tags.append(keyword.replace('_', ' ').title())
        
        return tags[:num_tags]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis"""
        if not isinstance(text, str):
            if text is None:
                return {'sentiment': 'neutral', 'confidence': 0.0}
            text = str(text)

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
        """Extract named entities using advanced NER (BERT-based + spaCy)"""
        if not isinstance(text, str):
            if text is None:
                return []
            text = str(text)

        if not text:
            return []

        # Clean and prepare text for better NER performance
        text = self._prepare_text_for_ner(text)
        print(f"🔍 Extracting entities using advanced NER techniques from {len(text)} characters...")
        entities = []

        # Try BERT-based NER first with text chunking for better performance
        if self.ner_pipeline and self.ner_pipeline != "spacy":
            try:
                # Split text into chunks for better BERT-NER performance
                text_chunks = self._split_text_for_ner(text)
                bert_entities = []

                for chunk in text_chunks:
                    if len(chunk.strip()) > 10:  # Only process meaningful chunks
                        chunk_entities = self.ner_pipeline(chunk)
                        bert_entities.extend(chunk_entities)

                # Process BERT entities
                for ent in bert_entities:
                    # Clean entity text
                    entity_text = ent['word'].replace('##', '').strip()
                    if len(entity_text) > 1 and entity_text.isalpha():
                        entities.append({
                            'text': entity_text,
                            'label': ent['entity_group'],
                            'confidence': round(ent['score'], 3),
                            'description': self._get_entity_description(ent['entity_group']),
                            'method': 'BERT-NER'
                        })

                print(f"✅ BERT-NER extracted {len(bert_entities)} raw entities, {len([e for e in entities if e['method'] == 'BERT-NER'])} processed")
            except Exception as e:
                print(f"⚠️ BERT-NER failed: {e}")
        elif self.ner_pipeline == "spacy":
            print("🔄 Using spaCy NER as primary method (BERT models unavailable)")
            # spaCy NER will be handled below

        # Also use spaCy NER for comparison
        if self.nlp:
            try:
                doc = self.nlp(text)
                spacy_entities = []
                for ent in doc.ents:
                    spacy_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'confidence': 1.0,  # spaCy doesn't provide confidence scores
                        'description': spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_,
                        'method': 'spaCy-NER'
                    })
                entities.extend(spacy_entities)
                print(f"✅ spaCy-NER extracted {len(spacy_entities)} entities")
            except Exception as e:
                print(f"⚠️ spaCy-NER failed: {e}")

        # Remove duplicates and sort by confidence
        unique_entities = self._deduplicate_entities(entities)
        print(f"🎯 Total unique entities extracted: {len(unique_entities)}")

        return unique_entities

    def _prepare_text_for_ner(self, text: str) -> str:
        """Prepare text for better NER performance"""
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Ensure proper sentence structure for NER
        text = text.replace('\n', '. ')
        text = text.replace('\t', ' ')

        # Fix common formatting issues
        text = text.replace('_', ' ')
        text = text.replace('-', ' ')

        # Ensure sentences end properly
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def _split_text_for_ner(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks for better NER processing"""
        if len(text) <= max_length:
            return [text]

        chunks = []
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_entity_description(self, entity_label: str) -> str:
        """Get description for entity labels"""
        descriptions = {
            'PER': 'Person',
            'PERSON': 'Person',
            'ORG': 'Organization',
            'ORGANIZATION': 'Organization',
            'LOC': 'Location',
            'LOCATION': 'Location',
            'MISC': 'Miscellaneous',
            'GPE': 'Geopolitical Entity',
            'DATE': 'Date',
            'TIME': 'Time',
            'MONEY': 'Money',
            'PERCENT': 'Percentage',
            'FACILITY': 'Facility',
            'PRODUCT': 'Product'
        }
        return descriptions.get(entity_label.upper(), entity_label)

    def _deduplicate_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate entities and keep highest confidence ones"""
        entity_map = {}

        for entity in entities:
            text = entity['text'].lower().strip()
            if text not in entity_map or entity['confidence'] > entity_map[text]['confidence']:
                entity_map[text] = entity

        # Sort by confidence descending
        return sorted(entity_map.values(), key=lambda x: x['confidence'], reverse=True)
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary"""
        if not isinstance(text, str):
            if text is None:
                return ""
            text = str(text)

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

    def get_bert_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Get BERT embeddings for text"""
        if not self.bert_model or not self.bert_tokenizer:
            return None

        try:
            # Tokenize and encode
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            print(f"✅ BERT embeddings generated: {embeddings.shape}")
            return embeddings

        except Exception as e:
            print(f"⚠️ BERT embedding generation failed: {e}")
            return None

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using BERT embeddings"""
        if not self.bert_model:
            return 0.0

        try:
            emb1 = self.get_bert_embeddings(text1)
            emb2 = self.get_bert_embeddings(text2)

            if emb1 is None or emb2 is None:
                return 0.0

            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            print(f"🔍 Semantic similarity calculated: {similarity:.3f}")
            return float(similarity)

        except Exception as e:
            print(f"⚠️ Semantic similarity calculation failed: {e}")
            return 0.0

    def advanced_content_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive content analysis using all available NLP techniques"""
        if not isinstance(text, str):
            if text is None:
                return {}
            text = str(text)

        print(f"🚀 Starting advanced content analysis...")

        analysis = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': self._count_sentences(text),
            'processing_methods': []
        }

        # Extract keywords with multiple methods
        analysis['keywords'] = self.extract_keywords(text, 15)
        analysis['processing_methods'].append('TF-IDF/spaCy/NLTK keyword extraction')

        # Extract entities with advanced NER
        analysis['entities'] = self.extract_entities(text)
        analysis['processing_methods'].append('BERT-NER + spaCy entity recognition')

        # Sentiment analysis
        analysis['sentiment'] = self.analyze_sentiment(text)
        analysis['processing_methods'].append('Lexicon-based sentiment analysis')

        # Generate summary
        analysis['summary'] = self.generate_summary(text, 3)
        analysis['processing_methods'].append('Extractive summarization')

        # BERT embeddings if available
        if self.bert_model:
            embeddings = self.get_bert_embeddings(text)
            if embeddings is not None:
                analysis['bert_embedding_dim'] = embeddings.shape[0]
                analysis['processing_methods'].append('BERT embeddings')

        # TF-IDF analysis if available
        if self.tfidf_vectorizer:
            analysis['processing_methods'].append('TF-IDF vectorization')

        print(f"✅ Advanced content analysis complete: {len(analysis['processing_methods'])} methods used")
        return analysis

    def generate_enhanced_description(self, dataset_info: Dict[str, Any]) -> str:
        """Generate enhanced dataset description using Free AI models with FLAN-T5 fallback"""
        print("🔍 Generating enhanced description using Free AI models...")
        print(f"🔍 Free AI available: {FREE_AI_AVAILABLE}")
        print(f"🔍 FLAN-T5 Model available: {self.t5_model is not None}")
        print(f"🔍 Dataset info keys: {list(dataset_info.keys())}")

        # Try Free AI models first (Mistral, Groq)
        if FREE_AI_AVAILABLE:
            try:
                print("🔄 Attempting Free AI description generation...")
                description = free_ai_service.generate_enhanced_description(dataset_info)
                if description and len(description.strip()) > 100:
                    # Clean the description to remove any remaining special characters
                    cleaned_description = self._clean_ai_description(description)
                    print(f"✅ Free AI description generated successfully: {len(cleaned_description)} chars")
                    return cleaned_description
                else:
                    print(f"⚠️ Free AI returned short description: '{description}'")
            except Exception as e:
                print(f"⚠️ Free AI models failed: {e}")

        # Fallback to FLAN-T5 Base model (offline capability)
        if self.t5_model is not None and self.t5_tokenizer is not None:
            try:
                print("🔄 Attempting FLAN-T5 Base description generation...")
                description = self._generate_description_t5(dataset_info)
                if description and len(description.strip()) > 50:
                    # Clean the description to remove any special characters
                    cleaned_description = self._clean_ai_description(description)
                    print(f"✅ FLAN-T5 Base description generated successfully: {len(cleaned_description)} chars")
                    return cleaned_description
                else:
                    print(f"⚠️ FLAN-T5 generated short description: '{description}'")
            except Exception as e:
                print(f"⚠️ FLAN-T5 model failed: {e}")
                import traceback
                traceback.print_exc()

        print("⚠️ All AI models unavailable or failed, using advanced local NLP description generation")
        return self._generate_comprehensive_local_description(dataset_info)

    def _clean_ai_description(self, description: str) -> str:
        """
        Clean AI-generated description by removing unnecessary characters and formatting

        Args:
            description: Raw AI-generated description

        Returns:
            Cleaned description without special characters
        """
        if not description or not isinstance(description, str):
            return ""

        # Remove markdown formatting
        description = re.sub(r'\*\*([^*]+)\*\*', r'\1', description)  # Bold
        description = re.sub(r'\*([^*]+)\*', r'\1', description)      # Italic
        description = re.sub(r'`([^`]+)`', r'\1', description)        # Code
        description = re.sub(r'#{1,6}\s*([^\n]+)', r'\1', description)  # Headers

        # Remove special characters but keep basic punctuation
        description = re.sub(r'[#@$%^&*+=\[\]{}|\\<>~`]', '', description)

        # Clean up multiple spaces and newlines
        description = re.sub(r'\n+', ' ', description)
        description = re.sub(r'\s+', ' ', description)

        # Remove leading/trailing whitespace
        description = description.strip()

        # Ensure proper sentence structure
        if description and not description.endswith('.'):
            description += '.'

        return description

    def _generate_description_t5(self, dataset_info: Dict[str, Any]) -> str:
        """Generate comprehensive description using T5 model with advanced prompting"""
        try:
            print(f"🔄 T5 generation starting with dataset_info: {dataset_info}")

            # Extract comprehensive dataset information with better defaults
            title = dataset_info.get('title', 'Unknown Dataset')
            field_names = dataset_info.get('field_names', [])
            record_count = dataset_info.get('record_count', 0)
            data_types = dataset_info.get('data_types', [])
            keywords = dataset_info.get('keywords', [])
            entities = dataset_info.get('entities', [])
            sample_data = dataset_info.get('sample_data', [])
            format_type = dataset_info.get('format', dataset_info.get('data_type', 'data'))
            use_cases = dataset_info.get('use_cases', [])
            category = dataset_info.get('category', 'General')
            summary = dataset_info.get('summary', '')

            print(f"🔍 Extracted info - Title: {title}, Records: {record_count}, Fields: {len(field_names)}")
            print(f"🔍 Keywords: {keywords[:3]}, Format: {format_type}, Category: {category}")

            # Create dynamic, dataset-specific prompts
            descriptions = []

            # 1. Generate dataset-specific overview
            overview_prompt = self._create_dynamic_overview_prompt(title, record_count, field_names, format_type, keywords, category)
            print(f"🔄 Overview prompt: {overview_prompt[:100]}...")
            overview_desc = self._generate_t5_text(overview_prompt, max_length=180, min_length=50)
            if overview_desc and len(overview_desc.strip()) > 20:
                descriptions.append(overview_desc)
                print(f"✅ Overview generated: {len(overview_desc)} chars")

            # 2. Generate field-specific structure description
            if field_names and len(field_names) > 0:
                structure_prompt = self._create_dynamic_structure_prompt(field_names, data_types, sample_data, title)
                print(f"🔄 Structure prompt: {structure_prompt[:100]}...")
                structure_desc = self._generate_t5_text(structure_prompt, max_length=150, min_length=40)
                if structure_desc and len(structure_desc.strip()) > 20:
                    descriptions.append(structure_desc)
                    print(f"✅ Structure generated: {len(structure_desc)} chars")

            # 3. Generate content-specific use case description
            if keywords or entities or summary:
                usecase_prompt = self._create_dynamic_usecase_prompt(title, keywords, entities, summary, category)
                print(f"🔄 Use case prompt: {usecase_prompt[:100]}...")
                usecase_desc = self._generate_t5_text(usecase_prompt, max_length=120, min_length=30)
                if usecase_desc and len(usecase_desc.strip()) > 20:
                    descriptions.append(usecase_desc)
                    print(f"✅ Use case generated: {len(usecase_desc)} chars")

            # Combine all descriptions into a comprehensive description
            if descriptions:
                # Try the new method first, fallback to existing method
                try:
                    combined_description = self._combine_dynamic_descriptions(descriptions, dataset_info)
                    print(f"✅ Comprehensive T5 description generated: {len(combined_description)} chars")
                    return combined_description
                except AttributeError:
                    # Fallback to existing method if new method not available (cache issue)
                    print("⚠️ Using fallback combination method due to cache issue")
                    combined_description = self._combine_t5_descriptions(descriptions)
                    print(f"✅ Fallback T5 description generated: {len(combined_description)} chars")
                    return combined_description
            else:
                # Fallback to enhanced simple generation
                print("⚠️ No T5 descriptions generated, using enhanced fallback")
                return self._generate_enhanced_simple_description(dataset_info)

        except Exception as e:
            print(f"⚠️ T5 description generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _create_overview_prompt(self, title: str, record_count: int, field_count: int, format_type: str, keywords: List[str]) -> str:
        """Create overview prompt for T5"""
        keyword_text = f" with topics including {', '.join(keywords[:5])}" if keywords else ""
        return f"You are a data science expert. Generate clear, professional dataset descriptions without markdown formatting or special characters. Give detailed explanation with focus on content, structure, and potential use cases. describe dataset: {title} contains {record_count:,} records in {field_count} fields, {format_type} format{keyword_text}. comprehensive research dataset for analysis."

    def _create_dynamic_overview_prompt(self, title: str, record_count: int, field_names: List[str], format_type: str, keywords: List[str], category: str = "General") -> str:
        """Create dynamic, dataset-specific overview prompt for T5"""
        field_count = len(field_names) if field_names else 0

        # Create contextual description based on category and content
        if category.lower() in ['business', 'commerce', 'sales', 'retail']:
            context = "business analytics and commercial insights"
        elif category.lower() in ['healthcare', 'medical', 'clinical']:
            context = "medical research and healthcare analysis"
        elif category.lower() in ['research', 'academic', 'scientific']:
            context = "academic research and scientific investigation"
        elif category.lower() in ['finance', 'financial', 'economic']:
            context = "financial analysis and economic modeling"
        else:
            context = "data analysis and research applications"

        # Include key fields for context
        field_context = ""
        if field_names:
            key_fields = field_names[:4]  # First 4 fields for context
            field_context = f" including {', '.join(key_fields)}"

        # Include keywords for domain context
        keyword_context = ""
        if keywords:
            keyword_context = f" focusing on {', '.join(keywords[:3])}"

        return f"summarize: {title} is a comprehensive {format_type} dataset containing {record_count:,} records across {field_count} fields{field_context}. This dataset is designed for {context}{keyword_context} and provides structured data for advanced analytical applications."

    def _create_dynamic_usecase_prompt(self, title: str, keywords: List[str], entities: List[str], summary: str, category: str = "General") -> str:
        """Create dynamic use case prompt for T5"""
        # Extract domain context from keywords and entities
        domain_indicators = {
            'business': ['sales', 'customer', 'revenue', 'profit', 'marketing', 'commerce', 'transaction'],
            'healthcare': ['patient', 'medical', 'clinical', 'diagnosis', 'treatment', 'health', 'hospital'],
            'finance': ['financial', 'investment', 'banking', 'credit', 'loan', 'portfolio', 'trading'],
            'research': ['research', 'study', 'experiment', 'analysis', 'scientific', 'academic'],
            'technology': ['software', 'system', 'network', 'database', 'algorithm', 'programming'],
            'education': ['student', 'course', 'learning', 'education', 'academic', 'university']
        }

        # Determine domain from keywords
        detected_domains = []
        all_terms = keywords + entities if entities else keywords
        for domain, indicators in domain_indicators.items():
            if any(indicator.lower() in [term.lower() for term in all_terms] for indicator in indicators):
                detected_domains.append(domain)

        # Create domain-specific use cases
        if 'business' in detected_domains:
            use_cases = "business intelligence, customer analytics, sales forecasting, market research, and performance optimization"
        elif 'healthcare' in detected_domains:
            use_cases = "medical research, patient outcome analysis, clinical decision support, epidemiological studies, and healthcare quality improvement"
        elif 'finance' in detected_domains:
            use_cases = "risk assessment, fraud detection, investment analysis, credit scoring, and financial modeling"
        elif 'research' in detected_domains:
            use_cases = "academic research, statistical analysis, hypothesis testing, data mining, and scientific discovery"
        elif 'technology' in detected_domains:
            use_cases = "system optimization, performance monitoring, software analytics, network analysis, and technical research"
        else:
            use_cases = "data analysis, statistical modeling, research applications, decision support, and analytical insights"

        # Include key entities for context
        entity_context = ""
        if entities and len(entities) > 0:
            key_entities = entities[:3] if isinstance(entities, list) else [str(entities)]
            entity_context = f" involving {', '.join(key_entities)}"

        return f"explain applications: {title} dataset{entity_context} can be used for {use_cases}. Describe specific analytical methods and research applications."

    def _create_dynamic_structure_prompt(self, field_names: List[str], data_types: List[str], sample_data: List[Dict], title: str = "dataset") -> str:
        """Create dynamic structure prompt for T5"""
        # Analyze field names for domain context
        field_categories = {
            'personal': ['name', 'id', 'age', 'gender', 'email', 'phone', 'address'],
            'business': ['revenue', 'profit', 'sales', 'customer', 'product', 'price', 'order'],
            'temporal': ['date', 'time', 'year', 'month', 'day', 'timestamp', 'created'],
            'geographic': ['location', 'city', 'country', 'region', 'latitude', 'longitude', 'address'],
            'cultural': ['culture', 'tradition', 'language', 'religion', 'custom', 'art', 'music', 'dance'],
            'medical': ['patient', 'diagnosis', 'treatment', 'symptom', 'medication', 'doctor'],
            'financial': ['amount', 'payment', 'transaction', 'account', 'balance', 'credit']
        }

        # Categorize fields
        detected_categories = []
        field_names_lower = [field.lower() for field in field_names]

        for category, indicators in field_categories.items():
            if any(indicator in ' '.join(field_names_lower) for indicator in indicators):
                detected_categories.append(category)

        # Create field context
        field_count = len(field_names)
        key_fields = field_names[:6] if field_names else []
        field_context = f"with {field_count} fields including {', '.join(key_fields)}" if key_fields else f"with {field_count} structured fields"

        # Create data type context
        unique_types = list(set(data_types)) if data_types else ['mixed types']
        type_context = f"containing {', '.join(unique_types[:4])} data"

        # Create sample data context
        sample_context = ""
        if sample_data and len(sample_data) > 0:
            sample_record = sample_data[0] if isinstance(sample_data, list) else sample_data
            if isinstance(sample_record, dict):
                sample_values = [str(v)[:20] for v in list(sample_record.values())[:3] if v is not None]
                if sample_values:
                    sample_context = f" with examples like {', '.join(sample_values)}"

        # Create domain-specific structure description
        if 'cultural' in detected_categories:
            structure_desc = "organized to capture cultural heritage, traditions, and social practices"
        elif 'business' in detected_categories:
            structure_desc = "structured for business analytics and commercial insights"
        elif 'medical' in detected_categories:
            structure_desc = "designed for healthcare and medical research applications"
        elif 'financial' in detected_categories:
            structure_desc = "organized for financial analysis and economic modeling"
        elif 'geographic' in detected_categories:
            structure_desc = "structured with geographic and location-based information"
        else:
            structure_desc = "organized for comprehensive data analysis and research"

        return f"describe structure: {title} is {structure_desc} {field_context} {type_context}{sample_context}. The dataset provides systematic organization for analytical processing and research applications."

    def _combine_dynamic_descriptions(self, descriptions: List[str], dataset_info: Dict) -> str:
        """Combine multiple T5-generated descriptions into a comprehensive description"""
        if not descriptions:
            return ""

        # Get dataset context
        title = dataset_info.get('title', 'dataset')
        record_count = dataset_info.get('record_count', 0)
        field_count = len(dataset_info.get('field_names', []))
        category = dataset_info.get('category', 'General')

        # Create introduction based on dataset characteristics
        if category.lower() in ['cultural', 'heritage', 'anthropology']:
            intro = f"The {title} represents a comprehensive cultural heritage collection"
        elif category.lower() in ['business', 'commerce', 'sales']:
            intro = f"The {title} provides extensive business intelligence data"
        elif category.lower() in ['medical', 'healthcare', 'clinical']:
            intro = f"The {title} contains valuable medical research information"
        elif category.lower() in ['research', 'academic', 'scientific']:
            intro = f"The {title} offers rich academic research data"
        else:
            intro = f"The {title} presents a comprehensive dataset"

        # Add scale information
        scale_info = f"encompassing {record_count:,} records across {field_count} structured fields" if record_count > 0 else "with structured data organization"

        # Combine descriptions intelligently
        combined_parts = [f"{intro} {scale_info}."]

        # Add each description with appropriate transitions
        for i, desc in enumerate(descriptions):
            if desc and len(desc.strip()) > 10:
                # Clean the description
                clean_desc = desc.strip()
                if not clean_desc.endswith('.'):
                    clean_desc += '.'

                # Add transition words for flow
                if i == 0:
                    combined_parts.append(clean_desc)
                elif i == 1:
                    combined_parts.append(f"Furthermore, {clean_desc.lower()}")
                else:
                    combined_parts.append(f"Additionally, {clean_desc.lower()}")

        # Add concluding statement
        if category.lower() in ['cultural', 'heritage']:
            conclusion = "This dataset serves as a valuable resource for cultural preservation, anthropological research, and heritage documentation initiatives."
        elif category.lower() in ['business', 'commerce']:
            conclusion = "This dataset enables comprehensive business analytics, market research, and strategic decision-making processes."
        elif category.lower() in ['medical', 'healthcare']:
            conclusion = "This dataset supports medical research, clinical analysis, and healthcare quality improvement initiatives."
        else:
            conclusion = "This dataset facilitates advanced analytical research, statistical modeling, and data-driven insights across multiple domains."

        combined_parts.append(conclusion)

        # Join all parts
        final_description = ' '.join(combined_parts)

        # Clean up any formatting issues
        final_description = ' '.join(final_description.split())  # Remove extra whitespace
        final_description = final_description.replace(' .', '.')  # Fix spacing before periods
        final_description = final_description.replace('..', '.')  # Remove double periods

        return final_description

    def _generate_python_analysis_code(self, dataset_info: Dict) -> str:
        """Generate dynamic Python code for dataset analysis using AI models"""

        # Try enhanced free AI models first
        try:
            from app.services.free_ai_service import free_ai_service

            if free_ai_service.is_available():
                print("🔄 Attempting enhanced AI Python code generation...")
                python_code = free_ai_service.generate_enhanced_python_code(dataset_info)
                if python_code and len(python_code) > 200:
                    print(f"✅ Enhanced AI Python code generated successfully: {len(python_code)} chars")
                    return python_code
                else:
                    print(f"⚠️ Enhanced AI returned short code: '{python_code[:100] if python_code else 'None'}'")
        except Exception as e:
            print(f"⚠️ Enhanced AI Python code generation failed: {e}")

        # Fallback to T5 model if available
        if not self.t5_model or not self.t5_tokenizer:
            print("⚠️ T5 model not available, using fallback")
            return self._generate_fallback_python_code(dataset_info)

        try:
            print("🔄 Attempting T5 Python code generation...")
            # Create Python code generation prompt
            title = dataset_info.get('title', 'dataset')
            field_names = dataset_info.get('field_names', [])
            data_types = dataset_info.get('data_types', [])
            record_count = dataset_info.get('record_count', 0)
            category = dataset_info.get('category', 'General')

            # Create context-aware prompt for Python code generation
            code_prompt = self._create_python_code_prompt(title, field_names, data_types, record_count, category)

            # Generate Python code using T5
            python_code = self._generate_t5_text(code_prompt, max_length=300, min_length=100)

            if python_code and len(python_code) > 50:
                # Clean and format the generated code
                formatted_code = self._format_python_code(python_code, dataset_info)
                print(f"✅ Generated Python analysis code: {len(formatted_code)} characters")
                return formatted_code
            else:
                print("⚠️ T5 Python code generation too short, using fallback")
                return self._generate_fallback_python_code(dataset_info)

        except Exception as e:
            print(f"⚠️ T5 Python code generation failed: {e}")
            return self._generate_fallback_python_code(dataset_info)

    def _create_python_code_prompt(self, title: str, field_names: List[str], data_types: List[str], record_count: int, category: str) -> str:
        """Create prompt for Python code generation"""
        # Analyze field types for appropriate analysis
        has_numeric = any('int' in str(dt).lower() or 'float' in str(dt).lower() for dt in data_types)
        has_text = any('object' in str(dt).lower() or 'string' in str(dt).lower() for dt in data_types)
        has_datetime = any('datetime' in str(dt).lower() or 'date' in str(dt).lower() for dt in data_types)

        # Create field context
        field_context = ""
        if field_names:
            key_fields = field_names[:4]
            field_context = f" with fields {', '.join(key_fields)}"

        # Create analysis context based on data types
        analysis_types = []
        if has_numeric:
            analysis_types.append("statistical analysis")
        if has_text:
            analysis_types.append("text analysis")
        if has_datetime:
            analysis_types.append("temporal analysis")
        if not analysis_types:
            analysis_types.append("exploratory data analysis")

        analysis_context = " and ".join(analysis_types)

        return f"generate python code: analyze {title} dataset{field_context} using pandas for {analysis_context}. include data loading, exploration, visualization with matplotlib and seaborn."

    def _format_python_code(self, raw_code: str, dataset_info: Dict) -> str:
        """Format and enhance the generated Python code"""
        title = dataset_info.get('title', 'dataset')
        field_names = dataset_info.get('field_names', [])

        # Clean the raw code
        code_lines = []

        # Add header comment
        code_lines.append(f"# Python Analysis Code for {title}")
        code_lines.append("# Auto-generated by FLAN-T5 Base Model")
        code_lines.append("")

        # Add imports
        code_lines.append("import pandas as pd")
        code_lines.append("import numpy as np")
        code_lines.append("import matplotlib.pyplot as plt")
        code_lines.append("import seaborn as sns")
        code_lines.append("from scipy import stats")
        code_lines.append("")

        # Add data loading
        code_lines.append("# Load the dataset")
        code_lines.append("df = pd.read_csv('your_dataset.csv')  # Replace with actual file path")
        code_lines.append("")

        # Process the T5 generated code
        raw_lines = raw_code.split('\n')
        for line in raw_lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Clean and add the line
                if 'import' not in line:  # Skip duplicate imports
                    code_lines.append(line)

        # Add standard analysis if T5 output is minimal
        if len(code_lines) < 15:
            code_lines.extend([
                "",
                "# Basic dataset exploration",
                "print('Dataset Shape:', df.shape)",
                "print('\\nDataset Info:')",
                "df.info()",
                "print('\\nBasic Statistics:')",
                "df.describe()",
                "",
                "# Check for missing values",
                "print('\\nMissing Values:')",
                "print(df.isnull().sum())",
                "",
                "# Visualizations",
                "plt.figure(figsize=(12, 8))",
                "df.hist(bins=20, figsize=(12, 8))",
                "plt.suptitle(f'Distribution of Numeric Variables - {title}')",
                "plt.tight_layout()",
                "plt.show()",
                "",
                "# Correlation matrix for numeric columns",
                "numeric_cols = df.select_dtypes(include=[np.number]).columns",
                "if len(numeric_cols) > 1:",
                "    plt.figure(figsize=(10, 8))",
                "    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)",
                "    plt.title('Correlation Matrix')",
                "    plt.show()"
            ])

        return '\n'.join(code_lines)

    def _generate_fallback_python_code(self, dataset_info: Dict) -> str:
        """Generate fallback Python code when T5 is unavailable"""
        title = dataset_info.get('title', 'Dataset')
        field_names = dataset_info.get('field_names', [])
        data_types = dataset_info.get('data_types', [])

        code_lines = [
            f"# Python Analysis Code for {title}",
            "# Generated by AI Meta Harvest",
            "",
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from scipy import stats",
            "",
            "# Load the dataset",
            "df = pd.read_csv('your_dataset.csv')  # Replace with actual file path",
            "",
            "# Basic dataset exploration",
            "print('Dataset Shape:', df.shape)",
            "print('\\nColumn Names:')",
            "print(df.columns.tolist())",
            "print('\\nDataset Info:')",
            "df.info()",
            "print('\\nBasic Statistics:')",
            "df.describe()",
            "",
            "# Check for missing values",
            "print('\\nMissing Values:')",
            "missing_data = df.isnull().sum()",
            "print(missing_data[missing_data > 0])",
            "",
            "# Data type analysis",
            "print('\\nData Types:')",
            "print(df.dtypes)",
            ""
        ]

        # Add field-specific analysis
        if field_names:
            code_lines.extend([
                "# Field-specific analysis",
                f"# Key fields: {', '.join(field_names[:5])}"
            ])

            # Add analysis for specific field types
            has_numeric = any('int' in str(dt).lower() or 'float' in str(dt).lower() for dt in data_types)
            has_text = any('object' in str(dt).lower() for dt in data_types)

            if has_numeric:
                code_lines.extend([
                    "",
                    "# Numeric data analysis",
                    "numeric_cols = df.select_dtypes(include=[np.number]).columns",
                    "print('\\nNumeric Columns:', numeric_cols.tolist())",
                    "",
                    "# Distribution plots",
                    "plt.figure(figsize=(15, 10))",
                    "df[numeric_cols].hist(bins=20, figsize=(15, 10))",
                    f"plt.suptitle('Distribution of Numeric Variables - {title}')",
                    "plt.tight_layout()",
                    "plt.show()",
                    "",
                    "# Correlation analysis",
                    "if len(numeric_cols) > 1:",
                    "    plt.figure(figsize=(10, 8))",
                    "    correlation_matrix = df[numeric_cols].corr()",
                    "    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)",
                    "    plt.title('Correlation Matrix')",
                    "    plt.show()"
                ])

            if has_text:
                code_lines.extend([
                    "",
                    "# Text data analysis",
                    "text_cols = df.select_dtypes(include=['object']).columns",
                    "print('\\nText Columns:', text_cols.tolist())",
                    "",
                    "# Unique values in text columns",
                    "for col in text_cols[:3]:  # Analyze first 3 text columns",
                    "    print(f'\\nUnique values in {col}: {df[col].nunique()}')",
                    "    print(f'Top 5 values in {col}:')",
                    "    print(df[col].value_counts().head())"
                ])

        # Add visualization section
        code_lines.extend([
            "",
            "# Advanced visualizations",
            "plt.style.use('seaborn-v0_8')",
            "",
            "# Pairplot for numeric variables (if not too many)",
            "numeric_cols = df.select_dtypes(include=[np.number]).columns",
            "if len(numeric_cols) <= 5 and len(numeric_cols) > 1:",
            "    sns.pairplot(df[numeric_cols])",
            f"    plt.suptitle('Pairplot - {title}', y=1.02)",
            "    plt.show()",
            "",
            "# Box plots for outlier detection",
            "if len(numeric_cols) > 0:",
            "    plt.figure(figsize=(12, 6))",
            "    df[numeric_cols].boxplot()",
            "    plt.title('Box Plots for Outlier Detection')",
            "    plt.xticks(rotation=45)",
            "    plt.tight_layout()",
            "    plt.show()",
            "",
            "print('\\nAnalysis completed successfully!')"
        ])

        return '\n'.join(code_lines)

    def _create_structure_prompt(self, field_names: List[str], data_types: List[str], sample_data: List[Dict]) -> str:
        """Create structure prompt for T5"""
        fields_text = ', '.join(field_names[:8])
        types_text = ', '.join(list(set(data_types))[:5]) if data_types else "mixed types"
        return f"explain data structure: fields {fields_text} with data types {types_text}. organized tabular data for statistical analysis."

    def _create_usecase_prompt(self, title: str, keywords: List[str], entities: List[Dict], use_cases: List[str]) -> str:
        """Create use case prompt for T5"""
        keyword_text = ', '.join(keywords[:5]) if keywords else "general analysis"
        entity_text = ', '.join([e.get('text', '') for e in entities[:3]]) if entities else ""
        context = f"{keyword_text} {entity_text}".strip()
        return f"applications for {title}: {context}. suitable for research, machine learning, statistical modeling, data science projects."

    def _generate_t5_text(self, prompt: str, max_length: int = 150, min_length: int = 30) -> str:
        """Generate text using T5 with specific parameters"""
        try:
            if not self.t5_model or not self.t5_tokenizer:
                return ""

            # Prepare input
            t5_input = f"summarize: {prompt}"
            inputs = self.t5_tokenizer.encode(t5_input, return_tensors="pt", max_length=512, truncation=True)

            # Generate
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=1.5,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.2
                )

            return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"⚠️ T5 text generation failed: {e}")
            return ""

    def _combine_t5_descriptions(self, descriptions: List[str], dataset_info: Dict[str, Any]) -> str:
        """Combine multiple T5 descriptions into a comprehensive description"""
        try:
            title = dataset_info.get('title', 'Dataset')
            record_count = dataset_info.get('record_count', 0)
            field_names = dataset_info.get('field_names', [])

            # Start with overview
            combined_parts = []

            if descriptions:
                # Add main description
                combined_parts.append(descriptions[0])

                # Add structural information
                if len(descriptions) > 1:
                    combined_parts.append(descriptions[1])

                # Add use case information
                if len(descriptions) > 2:
                    combined_parts.append(descriptions[2])

            # Add specific details
            if record_count and len(field_names):
                combined_parts.append(f"The dataset comprises {record_count:,} records organized across {len(field_names)} distinct fields, providing a robust foundation for comprehensive data analysis and research applications.")

            # Add field details for smaller datasets
            if field_names and len(field_names) <= 10:
                combined_parts.append(f"Key data fields include: {', '.join(field_names)}.")
            elif field_names:
                combined_parts.append(f"Primary data fields include: {', '.join(field_names[:6])}, among {len(field_names)} total variables.")

            # Add methodological suggestions
            combined_parts.append("This dataset supports various analytical approaches including statistical modeling, machine learning applications, and exploratory data analysis.")

            return " ".join(combined_parts)

        except Exception as e:
            print(f"⚠️ T5 description combination failed: {e}")
            return descriptions[0] if descriptions else ""

    def _generate_simple_t5_description(self, dataset_info: Dict[str, Any]) -> str:
        """Generate simple T5 description as fallback"""
        try:
            title = dataset_info.get('title', 'Dataset')
            record_count = dataset_info.get('record_count', 0)
            field_count = len(dataset_info.get('field_names', []))

            simple_prompt = f"describe: {title} dataset with {record_count:,} records and {field_count} fields for research analysis"
            return self._generate_t5_text(simple_prompt, max_length=200, min_length=50)
        except Exception as e:
            print(f"⚠️ Simple T5 description failed: {e}")
            return ""

    def _enhance_t5_description(self, base_description: str, dataset_info: Dict[str, Any]) -> str:
        """Enhance T5 generated description with additional context"""
        try:
            title = dataset_info.get('title', 'Dataset')
            record_count = dataset_info.get('record_count', 0)
            field_names = dataset_info.get('field_names', [])

            # Start with T5 description
            enhanced_parts = [base_description.strip()]

            # Add specific details
            if record_count and len(field_names):
                enhanced_parts.append(f"The dataset contains {record_count:,} records organized across {len(field_names)} fields, providing a comprehensive data structure for analysis.")

            # Add field information
            if field_names and len(field_names) <= 8:
                enhanced_parts.append(f"Key fields include: {', '.join(field_names)}.")
            elif field_names:
                enhanced_parts.append(f"Primary fields include: {', '.join(field_names[:5])}, among {len(field_names)} total variables.")

            # Add use case suggestions
            enhanced_parts.append("This dataset is well-suited for statistical analysis, data mining, and research applications.")

            return " ".join(enhanced_parts)

        except Exception as e:
            print(f"⚠️ T5 description enhancement failed: {e}")
            return base_description

# Removed free AI model methods - focusing on offline T5 model

    def _generate_comprehensive_local_description(self, dataset_info: Dict[str, Any]) -> str:
        """Generate comprehensive description using advanced local NLP techniques"""
        try:
            title = dataset_info.get('title', 'Dataset')
            field_names = dataset_info.get('field_names', [])
            record_count = dataset_info.get('record_count', 0)
            data_types = dataset_info.get('data_types', [])
            keywords = dataset_info.get('keywords', [])
            category = dataset_info.get('category', '')

            description_parts = []

            # Enhanced introduction with category context
            if category:
                description_parts.append(f"This {category.lower()} dataset, titled '{title}', represents a comprehensive collection of structured data designed for advanced analytical and research applications.")
            else:
                description_parts.append(f"This dataset, titled '{title}', represents a comprehensive collection of structured data designed for advanced analytical and research applications.")

            # Detailed data structure analysis
            if record_count and field_names:
                description_parts.append(f"The dataset encompasses {record_count:,} records systematically organized across {len(field_names)} distinct fields, providing a robust foundation for statistical analysis and data mining operations.")
            elif record_count:
                description_parts.append(f"The dataset contains {record_count:,} records, offering substantial data volume for comprehensive analysis.")

            # Advanced field analysis
            if field_names:
                if len(field_names) <= 8:
                    field_list = ", ".join(field_names)
                    description_parts.append(f"The dataset structure includes the following fields: {field_list}.")
                else:
                    primary_fields = field_names[:6]
                    field_list = ", ".join(primary_fields)
                    description_parts.append(f"Primary fields include: {field_list}, along with {len(field_names) - 6} additional variables providing comprehensive data coverage.")

            # Enhanced data type analysis
            if data_types:
                unique_types = list(set(data_types))
                if 'object' in unique_types or 'string' in unique_types:
                    description_parts.append("The dataset incorporates textual data elements enabling qualitative analysis and natural language processing applications.")

                if any(dtype in ['int64', 'float64', 'numeric'] for dtype in unique_types):
                    description_parts.append("Numerical data components support statistical modeling, mathematical analysis, and quantitative research methodologies.")

            # Advanced keyword and topic analysis
            if keywords:
                if len(keywords) <= 8:
                    keyword_list = ", ".join(keywords)
                    description_parts.append(f"The dataset encompasses key concepts and topics including: {keyword_list}, indicating its relevance for domain-specific research and analysis.")
                else:
                    primary_keywords = keywords[:6]
                    keyword_list = ", ".join(primary_keywords)
                    description_parts.append(f"Primary thematic elements include: {keyword_list}, among {len(keywords) - 6} additional conceptual dimensions.")

            # Comprehensive use case analysis
            use_cases = []

            if record_count and record_count > 1000:
                use_cases.append("large-scale statistical analysis and machine learning model development")
            elif record_count and record_count > 100:
                use_cases.append("statistical analysis and predictive modeling")

            if len(field_names) > 10:
                use_cases.append("multivariate analysis and feature engineering")

            # Default use cases
            if not use_cases:
                use_cases = ["exploratory data analysis", "statistical modeling", "data visualization", "research applications"]

            description_parts.append(f"This dataset is particularly well-suited for {', '.join(use_cases[:-1])}, and {use_cases[-1]}.")

            # Quality and methodology note
            description_parts.append("The structured nature of this dataset, combined with its comprehensive field coverage, makes it a valuable resource for researchers, analysts, and data scientists seeking to derive meaningful insights through rigorous analytical methodologies.")

            return " ".join(description_parts)

        except Exception as e:
            print(f"⚠️ Comprehensive local description generation failed: {e}")
            return self._generate_basic_description(dataset_info)

    def _generate_basic_description(self, dataset_info: Dict[str, Any]) -> str:
        """Generate basic description as fallback"""
        title = dataset_info.get('title', 'Dataset')
        record_count = dataset_info.get('record_count', 0)
        field_count = len(dataset_info.get('field_names', []))
        keywords = dataset_info.get('keywords', [])

        description_parts = [
            f"This dataset, titled '{title}', contains {record_count:,} records with {field_count} fields."
        ]

        if keywords:
            description_parts.append(f"Key topics include: {', '.join(keywords[:5])}.")

        if record_count > 1000:
            description_parts.append("This is a substantial dataset suitable for comprehensive analysis.")
        elif record_count > 100:
            description_parts.append("This dataset provides a good sample size for analysis.")

        description_parts.append("The dataset can be used for research, analysis, and data science applications.")

        return ' '.join(description_parts)

    def analyze_content_basic(self, text_content: str, field_names: List[str] = None) -> Dict[str, Any]:
        """Basic content analysis using existing methods"""
        if not isinstance(text_content, str):
            text_content = str(text_content) if text_content else ""

        return {
            'keywords': self.extract_keywords(text_content, 15),
            'entities': self.extract_entities(text_content),
            'summary': self.generate_summary(text_content),
            'sentiment': self.analyze_sentiment(text_content),
            'tags': self.suggest_tags(text_content, 8)
        }

    def _get_fallback_stopwords(self) -> set:
        """Get fallback English stopwords for offline use"""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now'
        }


# Global service instance
nlp_service = NLPService()


def get_nlp_service() -> NLPService:
    """Get the NLP service instance"""
    return nlp_service
