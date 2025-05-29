# Semantic Search Setup Guide

This guide explains how to set up advanced semantic search capabilities for AIMetaHarvest using NLP techniques like BERT and TF-IDF.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Install semantic search dependencies
pip install -r semantic_search_requirements.txt
```

### 2. Initialize Search Index

Visit the admin reindex route (requires admin user):
```
http://localhost:5000/admin/reindex-search
```

Or run programmatically:
```python
from app.services.dataset_service import get_dataset_service
service = get_dataset_service('uploads')
service.reindex_all_datasets()
```

## 🧠 How It Works

### Search Algorithms

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Traditional text matching
   - Fast and reliable
   - Weight: 30%

2. **BERT Embeddings (Semantic Similarity)**
   - Uses sentence-transformers with 'all-MiniLM-L6-v2' model
   - Understands semantic meaning
   - Weight: 40%

3. **Metadata Matching**
   - Exact matches in categories, tags, data types
   - Weight: 20%

4. **Popularity Scoring**
   - Based on recency and other factors
   - Weight: 10%

### Search Features

- **Semantic Understanding**: "climate data" matches "weather information"
- **Multi-field Search**: Searches title, description, tags, source
- **Hybrid Scoring**: Combines multiple algorithms for best results
- **Graceful Degradation**: Falls back to basic search if ML models unavailable
- **Caching**: Models and embeddings are cached for performance

## 📦 Dependencies

### Required
- `scikit-learn>=1.3.0` - For TF-IDF vectorization
- `sentence-transformers>=2.2.2` - For BERT embeddings
- `transformers>=4.21.0` - Transformer models
- `torch>=1.12.0` - PyTorch backend

### Optional (for better performance)
- `faiss-cpu>=1.7.4` - Fast similarity search
- `accelerate>=0.20.0` - Faster BERT inference

## 🔧 Configuration

### Search Weights
You can adjust the algorithm weights in `semantic_search_service.py`:

```python
self.weights = {
    'tfidf': 0.3,      # Traditional text matching
    'bert': 0.4,       # Semantic similarity
    'metadata': 0.2,   # Exact metadata matches
    'popularity': 0.1  # Recency/popularity
}
```

### Model Selection
The service uses `all-MiniLM-L6-v2` by default (fast, lightweight). You can change this in the `_load_bert_model` method:

```python
# For better quality (slower):
self.bert_model = SentenceTransformer('all-mpnet-base-v2')

# For faster inference (lower quality):
self.bert_model = SentenceTransformer('all-MiniLM-L12-v2')
```

## 🎯 Usage Examples

### Basic Search
```
Query: "climate"
Results: Datasets about climate, weather, temperature, etc.
```

### Semantic Search
```
Query: "machine learning models"
Results: Datasets about AI, neural networks, algorithms, etc.
```

### Category + Text Search
```
Query: "temperature" + Category: "Environment"
Results: Environmental datasets containing temperature data
```

## 🚨 Troubleshooting

### Models Not Loading
- Check internet connection (models download on first use)
- Ensure sufficient disk space (~500MB for models)
- Check write permissions in cache directory

### Slow Search
- Install optional dependencies (`faiss-cpu`, `accelerate`)
- Reduce dataset count for indexing
- Use smaller BERT model

### Memory Issues
- Use CPU-only versions of dependencies
- Reduce batch size in model configuration
- Clear cache periodically

## 📊 Performance

### Indexing Time
- ~1-2 seconds per 100 datasets
- One-time operation (cached)
- Incremental updates supported

### Search Time
- ~50-200ms per query
- Depends on dataset count
- Cached embeddings improve speed

### Memory Usage
- ~200-500MB for models
- ~1-5MB per 1000 datasets (embeddings)
- Scales linearly with dataset count

## 🔄 Maintenance

### Reindexing
Reindex when:
- New datasets are added in bulk
- Search quality degrades
- Model updates are available

### Cache Management
Cache files are stored in `app/cache/search/`:
- `tfidf_vectorizer.pkl` - TF-IDF model
- `sentence_transformer/` - BERT model
- `bert_embeddings.pkl` - Dataset embeddings

Clear cache to force model redownload or reset embeddings.

## 🎉 Benefits

- **Better Results**: Semantic understanding improves relevance
- **User Friendly**: Natural language queries work better
- **Scalable**: Efficient caching and indexing
- **Robust**: Graceful fallback to basic search
- **Configurable**: Adjustable weights and models
