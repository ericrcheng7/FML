# FML - Fundamentals of Machine Learning

## NYC Airbnb Data Analysis and Embeddings

This repository contains code for downloading, cleaning, and generating embeddings for NYC Airbnb listings data from [Inside Airbnb](https://insideairbnb.com/get-the-data/).

### Features

- **Data Download**: Automatically downloads the latest NYC Airbnb listings data
- **Data Cleaning**: 
  - Fixes the price column (removes $ signs and commas, converts to numeric)
  - Removes duplicate listings
  - Selects useful room features
  - Filters out invalid data and extreme outliers
- **Embedding Generation**: Uses the `nomic-ai/nomic-embed-text-v1.5` model from sentence-transformers to generate 768-dimensional embeddings for each listing

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run the Python script

```bash
python nyc_airbnb_analysis.py
```

This will:
1. Download the NYC listings data
2. Clean the data
3. Generate embeddings
4. Save the results to `data/nyc_listings_cleaned_embedded.parquet`

#### Option 2: Use the Jupyter notebook

```bash
jupyter notebook nyc_airbnb_analysis.ipynb
```

The notebook provides an interactive environment with visualizations and step-by-step analysis.

### Testing

Run the unit tests to verify the data cleaning functionality:

```bash
python test_unit.py
```

This tests the core data cleaning logic with mock data.

### Output

The processed data is saved as a Parquet file containing:
- All useful listing features (name, description, location, price, etc.)
- A 768-dimensional embedding vector for each listing

### Embeddings Approach

The embeddings are generated using the approach learned in [lab3_embeddings_retrieval](https://github.com/kyunghyuncho/CSCI-UA-473-Fundamentals-of-MachineLearning-Spring-2026/tree/main/labs/lab3_embeddings_retrieval):

1. Load the `SentenceTransformer` model (`nomic-ai/nomic-embed-text-v1.5`)
2. Create text representations combining listing name, description, amenities, and location
3. Prefix text with "search_document:" for optimal encoding
4. Generate normalized embeddings (L2 norm = 1)
5. Store embeddings alongside listing data

### Data Cleaning Details

The cleaning process includes:

1. **Price Column Cleaning**:
   - Removes "$" symbols and commas
   - Converts to numeric type
   - Handles conversion errors gracefully

2. **Duplicate Removal**:
   - Identifies and removes duplicate listings by ID

3. **Feature Selection**:
   - Keeps only useful columns for analysis
   - Includes: id, name, description, location, property type, room type, amenities, price, reviews, etc.

4. **Data Quality Filters**:
   - Removes listings without name or price
   - Removes listings with invalid prices (≤ $0)
   - Removes extreme outliers (> $10,000/night)

### Use Cases

The generated embeddings can be used for:
- Semantic search of listings
- Finding similar properties
- Clustering listings by features
- Building recommendation systems
- Price prediction based on similar listings