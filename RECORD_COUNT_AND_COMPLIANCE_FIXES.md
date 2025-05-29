# Record Count and Schema.org/FAIR Compliance Fixes

## Overview

This document outlines the fixes and improvements made to address two critical issues:

1. **Record Count Problem**: Datasets with more than 10,000 rows showing incorrect record counts
2. **Schema.org and FAIR Compliance**: Enhanced metadata structure for better compliance

## 🔧 Record Count Fix

### Problem
The dataset processing was using `pd.read_csv(nrows=100)` to analyze CSV files, which limited the analysis to only the first 100 rows. The record count was then calculated as `len(df)`, resulting in a maximum count of 100 instead of the actual file size.

### Solution
Implemented efficient row counting that separates schema analysis from record counting:

#### Changes Made:

**1. Enhanced CSV Processing (`app/services/dataset_service.py`)**
```python
# Before: Limited to 100 rows
df = pd.read_csv(file_path, sep=delimiter, nrows=100, encoding='utf-8')
record_count = len(df)  # Always ≤ 100

# After: Separate sample analysis and full counting
df_sample = pd.read_csv(file_path, sep=delimiter, nrows=100, encoding='utf-8')
total_rows = self._count_csv_rows(file_path, delimiter)  # Count all rows
```

**2. New Efficient Row Counting Method**
```python
def _count_csv_rows(self, file_path, delimiter=','):
    """Efficiently count total rows in CSV file without loading all data"""
    try:
        row_count = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file, None)  # Skip header
            for line in file:
                if line.strip():  # Skip empty lines
                    row_count += 1
        return row_count
    except Exception as e:
        # Fallback with different encodings
        # ... (handles encoding issues)
```

**3. Fallback Processing Enhancement**
- Updated fallback CSV processing to also use correct record counting
- Added `_count_csv_rows_with_encoding()` for encoding-specific counting

### Benefits:
- ✅ Accurate record counts for datasets of any size
- ✅ Maintains fast processing by only sampling data for schema analysis
- ✅ Robust encoding support with fallback mechanisms
- ✅ Memory efficient - doesn't load entire dataset into memory

## 🏛️ Schema.org and FAIR Compliance Enhancements

### Enhanced Dataset Model

**New Fields Added (`app/models/dataset.py`)**:
```python
# Enhanced Schema.org and FAIR compliance fields
persistent_id = StringField(max_length=100, unique=True)  # DOI-like identifier
version = StringField(max_length=20, default='1.0')
access_url = StringField(max_length=512)
access_protocol = StringField(max_length=50, default='HTTP')
distribution_format = StringField(max_length=50)
content_size = StringField(max_length=50)
encoding_format = StringField(max_length=50)
date_published = DateTimeField()
date_modified = DateTimeField()
creator_type = StringField(max_length=50, default='Person')
publisher_type = StringField(max_length=50, default='Organization')
variable_measured = StringField()  # JSON string of measured variables
measurement_technique = StringField()
funding = StringField()
citation = StringField()
is_based_on = StringField()  # Related work/datasets
same_as = StringField()  # Equivalent datasets
schema_org_json = StringField()  # Full Schema.org JSON-LD
fair_score = FloatField(default=0.0)
schema_org_score = FloatField(default=0.0)
```

### Enhanced Schema.org Metadata Generation

**Comprehensive Schema.org JSON-LD (`app/services/metadata_completion_service.py`)**:
```python
def _generate_schema_org_metadata(self, dataset, processed_data=None):
    """Generate comprehensive Schema.org compliant metadata."""
    schema_org = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": dataset.title,
        "description": dataset.description,
        "identifier": persistent_id,
        "dateCreated": dataset.created_at.isoformat(),
        "dateModified": dataset.updated_at.isoformat(),
        "datePublished": dataset.date_published.isoformat(),
        "version": dataset.version,
        "creator": {
            "@type": dataset.creator_type,
            "name": dataset.author
        },
        "publisher": {
            "@type": dataset.publisher_type,
            "name": dataset.publisher
        },
        "license": dataset.license,
        "keywords": self._parse_keywords(dataset.keywords),
        "inLanguage": "en",
        "isAccessibleForFree": True,
        "distribution": {
            "@type": "DataDownload",
            "encodingFormat": processed_data.get('format'),
            "contentSize": dataset.size,
            "contentUrl": f"/datasets/{dataset.id}/download",
            "numberOfRecords": processed_data['record_count']  # Now accurate!
        },
        "variableMeasured": [
            {
                "@type": "PropertyValue",
                "name": var,
                "description": f"Data variable: {var}"
            } for var in variables[:10]
        ]
        # ... additional fields for temporal/spatial coverage, funding, etc.
    }
```

### Enhanced FAIR Compliance Scoring

**Improved Findability Assessment (`app/services/quality_scoring/dimension_scorers.py`)**:
```python
def _assess_findability(self, dataset):
    """Enhanced findability assessment with persistent identifiers."""
    score = 0.0
    
    # F1: Persistent identifiers (30 points)
    if hasattr(dataset, 'persistent_id') and dataset.persistent_id:
        score += 30.0  # Higher score for persistent ID
    elif hasattr(dataset, 'identifier') and dataset.identifier:
        score += 15.0  # Partial score for basic identifier
    
    # F2: Rich metadata (25 points)
    metadata_richness = 0
    if dataset.description and len(dataset.description) > 100:
        metadata_richness += 10
    if dataset.keywords: metadata_richness += 5
    if dataset.tags: metadata_richness += 5
    if dataset.category: metadata_richness += 5
    score += metadata_richness
    
    # F3: Metadata includes identifier (20 points)
    if dataset.persistent_id and dataset.description:
        score += 20
    
    # F4: Indexed and searchable (25 points)
    if dataset.indexed: score += 15
    if dataset.schema_org_json: score += 10  # Schema.org improves findability
    
    return min(score, 100.0)
```

**Enhanced Interoperability Assessment**:
```python
def _assess_interoperability(self, dataset):
    """Enhanced interoperability with Schema.org support."""
    score = 0.0
    
    # I1: Knowledge representation language
    if hasattr(dataset, 'schema_org_json') and dataset.schema_org_json:
        score += 35.0  # Higher score for full Schema.org JSON-LD
    elif hasattr(dataset, 'schema_org') and dataset.schema_org:
        score += 25.0
    elif dataset.format in ['json', 'xml', 'rdf', 'csv']:
        score += 15.0
    
    # ... additional assessments
```

## 🔄 Processing Pipeline Updates

**Enhanced Metadata Storage (`app/services/processing_service.py`)**:
```python
# Handle Schema.org metadata specially
if key == 'schema_org_metadata':
    import json
    filtered_enhancements['schema_org_json'] = json.dumps(value)
    continue
```

## 📊 Benefits of the Enhancements

### Record Count Accuracy:
- ✅ **Accurate counting** for datasets of any size (tested up to 15,000+ rows)
- ✅ **Memory efficient** - doesn't load entire dataset
- ✅ **Fast processing** - maintains quick schema analysis
- ✅ **Robust encoding support** - handles various file encodings

### Schema.org Compliance:
- ✅ **Complete JSON-LD structure** following Schema.org Dataset specification
- ✅ **Rich metadata** including distribution, variables, provenance
- ✅ **Persistent identifiers** for better findability
- ✅ **Structured data** for search engine optimization

### FAIR Compliance:
- ✅ **Enhanced Findability** with persistent IDs and rich metadata
- ✅ **Improved Accessibility** with proper access protocols
- ✅ **Better Interoperability** through Schema.org and standard formats
- ✅ **Increased Reusability** with clear licenses and provenance

## 🧪 Testing

Run the test script to verify the record count fix:

```bash
python test_record_count_fix.py
```

This test:
- Creates CSV files with 15,000 rows
- Creates JSON files with 12,000 records
- Verifies accurate record counting
- Tests both normal and fallback processing

## 🚀 Next Steps

1. **Deploy the changes** to the production environment
2. **Re-process existing datasets** to update record counts and metadata
3. **Monitor FAIR compliance scores** for improved ratings
4. **Add Schema.org structured data** to dataset pages for SEO
5. **Implement persistent identifier generation** for new datasets

## 📝 Files Modified

1. `app/services/dataset_service.py` - Fixed record counting
2. `app/models/dataset.py` - Added new compliance fields
3. `app/services/metadata_completion_service.py` - Enhanced Schema.org generation
4. `app/services/quality_scoring/dimension_scorers.py` - Improved FAIR scoring
5. `app/services/processing_service.py` - Updated metadata storage
6. `test_record_count_fix.py` - Test script for verification
