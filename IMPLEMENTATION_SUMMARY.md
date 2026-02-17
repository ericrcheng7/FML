# NYC Airbnb Data Analysis - Implementation Summary

## Overview

This implementation fulfills all requirements from the problem statement:

1. ✅ **Import NYC Airbnb listings data** from Inside Airbnb (https://insideairbnb.com/get-the-data/)
2. ✅ **Clean the dataset** including price column validation, duplicate removal, and feature selection
3. ✅ **Generate embeddings** using the approach from lab3_embeddings_retrieval

## Files Created

### Main Implementation Files

1. **`nyc_airbnb_analysis.py`** (266 lines)
   - Python script for complete data pipeline
   - Downloads, cleans, and generates embeddings
   - Can be run standalone: `python nyc_airbnb_analysis.py`

2. **`nyc_airbnb_analysis.ipynb`** (Jupyter notebook)
   - Interactive notebook version with visualizations
   - Step-by-step analysis with explanations
   - Includes exploratory data analysis plots

3. **`example_semantic_search.py`** (172 lines)
   - Demonstrates how to use the generated embeddings
   - Implements semantic search functionality
   - Shows how to find similar listings

### Testing Files

4. **`test_unit.py`** (173 lines)
   - Comprehensive unit tests for data cleaning functions
   - Tests with mock data
   - Verifies all cleaning operations work correctly
   - ✅ All tests pass

5. **`test_cleaning.py`** (129 lines)
   - Integration test for downloading and cleaning real data
   - Can be used to verify the full pipeline

### Configuration Files

6. **`requirements.txt`**
   - Lists all required Python packages
   - Includes pandas, numpy, sentence-transformers, etc.

7. **`.gitignore`**
   - Excludes data files, cache, and build artifacts
   - Prevents committing large files to the repository

8. **`README.md`**
   - Comprehensive documentation
   - Installation and usage instructions
   - Explains the embeddings approach

## Data Cleaning Details

### 1. Price Column Cleaning ✅

The price column in raw Airbnb data contains strings like `"$150"` or `"$1,200"`.

**Solution:**
```python
df['price'] = df['price'].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
```

**Result:** Numeric price column suitable for analysis

### 2. Duplicate Removal ✅

Airbnb data may contain duplicate listings with the same ID.

**Solution:**
```python
df = df.drop_duplicates(subset=['id'])
```

**Result:** Each listing appears only once in the dataset

### 3. Feature Selection ✅

The raw data contains 75+ columns, many of which are not useful for analysis.

**Selected Features:**
- **Identification:** id
- **Description:** name, description, neighborhood_overview
- **Host Info:** host_id, host_name
- **Location:** neighbourhood_cleansed, neighbourhood_group_cleansed, latitude, longitude
- **Property:** property_type, room_type, accommodates, bathrooms_text, bedrooms, beds, amenities
- **Pricing:** price, minimum_nights, maximum_nights
- **Reviews:** number_of_reviews, review_scores_*, reviews_per_month
- **Booking:** instant_bookable

**Result:** 24 carefully selected columns that are useful for analysis and embeddings

### 4. Data Quality Filters ✅

Applied multiple filters to ensure data quality:
- Remove listings without name or price (essential fields)
- Remove listings with invalid prices (≤ $0)
- Remove extreme outliers (> $10,000/night)

**Result:** High-quality, analysis-ready dataset

## Embedding Generation

### Approach (Following lab3_embeddings_retrieval) ✅

1. **Model Selection:** `nomic-ai/nomic-embed-text-v1.5`
   - State-of-the-art sentence transformer
   - Produces 768-dimensional embeddings
   - Supports asymmetric search (query vs document)

2. **Text Representation:**
   ```python
   def create_text_for_embedding(row):
       # Combines: name, description, neighborhood, property type,
       #           room type, amenities, location
       return " ".join([...])
   ```

3. **Embedding Generation:**
   ```python
   # Add "search_document:" prefix (as per lab3)
   texts_prefixed = ["search_document: " + text for text in texts]
   
   # Generate normalized embeddings
   embeddings = model.encode(
       texts_prefixed,
       normalize_embeddings=True,  # L2 norm = 1
       batch_size=32,
       show_progress_bar=True
   )
   ```

4. **Storage:**
   - Embeddings stored as numpy arrays in parquet format
   - Each listing has a 768-dimensional vector

### Verification ✅

- Embeddings are normalized (L2 norm ≈ 1.0)
- Semantic similarity works correctly
- Can perform semantic search using dot product

## Usage Examples

### 1. Run the Main Script

```bash
python nyc_airbnb_analysis.py
```

Downloads data, cleans it, generates embeddings, and saves to `data/nyc_listings_cleaned_embedded.parquet`

### 2. Use the Jupyter Notebook

```bash
jupyter notebook nyc_airbnb_analysis.ipynb
```

Interactive analysis with visualizations

### 3. Semantic Search Example

```bash
python example_semantic_search.py
```

Demonstrates how to:
- Search for listings by natural language query
- Find similar listings
- Rank results by semantic similarity

### 4. Run Tests

```bash
python test_unit.py
```

Verifies all data cleaning functions work correctly

## Key Features

### 1. Robust Data Cleaning
- Handles missing data gracefully
- Converts price to numeric format
- Removes duplicates and outliers
- Selects only useful features

### 2. Semantic Understanding
- Embeddings capture meaning, not just keywords
- Can find "luxury apartment" even if listing doesn't use word "luxury"
- Understands synonyms and related concepts

### 3. Scalable Design
- Batch processing for efficiency
- Normalized embeddings for fast similarity computation
- Parquet format for efficient storage

### 4. Well-Tested
- Unit tests verify each cleaning operation
- Tests with mock data ensure robustness
- Clear error messages for debugging

## Technical Highlights

1. **Following Lab3 Best Practices:**
   - Using `search_document:` prefix for documents
   - Using `search_query:` prefix for queries (in examples)
   - Normalizing embeddings for cosine similarity via dot product
   - Batch processing for efficiency

2. **Data Engineering:**
   - Pandas for data manipulation
   - NumPy for numerical operations
   - Parquet for efficient storage

3. **Code Quality:**
   - Clear function documentation
   - Type hints where appropriate
   - Modular design for reusability
   - Comprehensive error handling

## Future Enhancements

Possible extensions (not required but enabled by this implementation):

1. **Advanced Search:**
   - Filter by price range, location, amenities
   - Multi-modal search (text + filters)
   - Personalized recommendations

2. **Clustering:**
   - Group similar listings
   - Discover listing types automatically
   - Market segmentation analysis

3. **Visualization:**
   - t-SNE or UMAP of embedding space
   - Interactive search interface
   - Geographic visualization with embeddings

4. **Price Prediction:**
   - Use embeddings as features for ML model
   - Predict fair price based on similar listings
   - Detect overpriced/underpriced listings

## Conclusion

This implementation successfully addresses all requirements:

✅ Imports NYC Airbnb data from Inside Airbnb
✅ Cleans the dataset (price, duplicates, features)
✅ Generates embeddings using lab3 approach
✅ Well-documented and tested
✅ Provides usage examples

The code is production-ready, well-structured, and follows best practices from the course materials.
