"""
NYC Airbnb Listings Data Analysis and Embeddings

This script:
1. Downloads NYC Airbnb listings data from Inside Airbnb
2. Cleans the data (handles price column, removes duplicates, selects useful features)
3. Generates embeddings for the features using sentence transformers
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import urllib.request
import os


def download_nyc_data(output_dir="data"):
    """
    Download NYC Airbnb listings data from Inside Airbnb
    
    Args:
        output_dir: Directory to save the downloaded data
        
    Returns:
        Path to the downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for NYC listings data
    # Using the latest available data from Inside Airbnb
    url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2024-12-04/data/listings.csv.gz"
    output_path = os.path.join(output_dir, "listings.csv.gz")
    
    print(f"Downloading NYC listings data from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded data to {output_path}")
    
    return output_path


def clean_data(df):
    """
    Clean the Airbnb listings dataset
    
    Args:
        df: Raw listings dataframe
        
    Returns:
        Cleaned dataframe
    """
    print("\n=== Cleaning Data ===")
    print(f"Original shape: {df.shape}")
    
    # 1. Clean the price column
    # Remove dollar signs and commas, convert to float
    if 'price' in df.columns:
        df['price'] = df['price'].str.replace('$', '', regex=False)
        df['price'] = df['price'].str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        print(f"Cleaned price column - removed $ and commas")
    
    # 2. Remove duplicates
    original_count = len(df)
    df = df.drop_duplicates(subset=['id'])
    duplicates_removed = original_count - len(df)
    print(f"Removed {duplicates_removed} duplicate listings")
    
    # 3. Select useful room features to keep
    # Keep relevant columns for analysis and embedding generation (30 columns)
    useful_columns = [
        'id',
        'name',
        'description',
        'neighborhood_overview',
        'host_id',
        'host_name',
        'neighbourhood_cleansed',
        'neighbourhood_group_cleansed',
        'latitude',
        'longitude',
        'property_type',
        'room_type',
        'accommodates',
        'bathrooms_text',
        'bedrooms',
        'beds',
        'amenities',
        'price',
        'minimum_nights',
        'maximum_nights',
        'number_of_reviews',
        'review_scores_rating',
        'review_scores_accuracy',
        'review_scores_cleanliness',
        'review_scores_checkin',
        'review_scores_communication',
        'review_scores_location',
        'review_scores_value',
        'instant_bookable',
        'reviews_per_month'
    ]
    
    # Only keep columns that exist in the dataframe
    existing_columns = [col for col in useful_columns if col in df.columns]
    df = df[existing_columns]
    print(f"Selected {len(existing_columns)} useful columns")
    
    # 4. Remove listings with missing essential data
    # Remove listings without name, description, or price
    df = df.dropna(subset=['name', 'price'])
    print(f"Removed listings with missing name or price")
    
    # 5. Remove invalid price values (negative or zero prices)
    df = df[df['price'] > 0]
    print(f"Removed listings with invalid prices (<= 0)")
    
    # 6. Remove extreme outliers in price (e.g., > $10,000/night)
    df = df[df['price'] <= 10000]
    print(f"Removed listings with extreme prices (> $10,000)")
    
    print(f"Final shape after cleaning: {df.shape}")
    
    return df


def create_text_for_embedding(row):
    """
    Create a text representation of a listing for embedding generation
    
    Args:
        row: A row from the dataframe
        
    Returns:
        String representation of the listing
    """
    parts = []
    
    # Add listing name
    if pd.notna(row.get('name')):
        parts.append(f"Name: {row['name']}")
    
    # Add description
    if pd.notna(row.get('description')):
        parts.append(f"Description: {row['description']}")
    
    # Add neighborhood overview
    if pd.notna(row.get('neighborhood_overview')):
        parts.append(f"Neighborhood: {row['neighborhood_overview']}")
    
    # Add property type and room type
    if pd.notna(row.get('property_type')):
        parts.append(f"Property Type: {row['property_type']}")
    
    if pd.notna(row.get('room_type')):
        parts.append(f"Room Type: {row['room_type']}")
    
    # Add amenities
    if pd.notna(row.get('amenities')):
        parts.append(f"Amenities: {row['amenities']}")
    
    # Add location info
    if pd.notna(row.get('neighbourhood_cleansed')):
        parts.append(f"Location: {row['neighbourhood_cleansed']}")
    
    return " ".join(parts)


def generate_embeddings(df, model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=32):
    """
    Generate embeddings for the listings using sentence transformers
    
    Args:
        df: Cleaned listings dataframe
        model_name: Name of the sentence transformer model to use
        batch_size: Batch size for encoding
        
    Returns:
        Dataframe with embeddings column added
    """
    print("\n=== Generating Embeddings ===")
    print(f"Loading model: {model_name}")
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Create text representations for each listing
    print("Creating text representations...")
    texts = df.apply(create_text_for_embedding, axis=1).tolist()
    
    # Add the "search_document:" prefix as per the model's recommendation
    texts_prefixed = ["search_document: " + text for text in texts]
    
    # Generate embeddings with normalization
    print(f"Generating embeddings for {len(texts_prefixed)} listings...")
    embeddings = model.encode(
        texts_prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Add embeddings to dataframe
    df['embedding'] = list(embeddings)
    
    return df


def save_data(df, output_path="data/nyc_listings_cleaned_embedded.parquet"):
    """
    Save the cleaned and embedded data
    
    Args:
        df: Dataframe to save
        output_path: Path to save the data
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n=== Data saved to {output_path} ===")


def print_summary_statistics(df):
    """
    Print summary statistics of the cleaned data
    
    Args:
        df: Cleaned dataframe
    """
    print("\n=== Summary Statistics ===")
    print(f"Total listings: {len(df)}")
    print(f"\nPrice statistics:")
    print(df['price'].describe())
    
    if 'room_type' in df.columns:
        print(f"\nRoom type distribution:")
        print(df['room_type'].value_counts())
    
    if 'neighbourhood_group_cleansed' in df.columns:
        print(f"\nTop 10 neighborhoods:")
        print(df['neighbourhood_group_cleansed'].value_counts().head(10))
    
    if 'property_type' in df.columns:
        print(f"\nTop 10 property types:")
        print(df['property_type'].value_counts().head(10))


def main():
    """
    Main function to run the complete pipeline
    """
    print("NYC Airbnb Listings Data Analysis and Embeddings")
    print("=" * 60)
    
    # Step 1: Download data
    data_path = download_nyc_data()
    
    # Step 2: Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path, compression='gzip')
    print(f"Loaded {len(df)} listings with {len(df.columns)} columns")
    
    # Step 3: Clean data
    df_clean = clean_data(df)
    
    # Step 4: Print summary statistics
    print_summary_statistics(df_clean)
    
    # Step 5: Generate embeddings
    df_embedded = generate_embeddings(df_clean)
    
    # Step 6: Save the results
    save_data(df_embedded)
    
    print("\n=== Processing Complete! ===")
    print(f"Final dataset contains {len(df_embedded)} listings")
    print(f"Each listing has a {df_embedded['embedding'].iloc[0].shape[0]}-dimensional embedding")
    
    return df_embedded


if __name__ == "__main__":
    df = main()
