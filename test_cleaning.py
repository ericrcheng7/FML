"""
Test script for NYC Airbnb data cleaning (without embeddings)

This script tests the data download and cleaning functionality
without requiring the embedding model.
"""

import pandas as pd
import numpy as np
import urllib.request
import os
import sys


def download_nyc_data(output_dir="data"):
    """Download NYC Airbnb listings data from Inside Airbnb"""
    os.makedirs(output_dir, exist_ok=True)
    
    url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2024-12-04/data/listings.csv.gz"
    output_path = os.path.join(output_dir, "listings.csv.gz")
    
    print(f"Downloading NYC listings data from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded data to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return None


def clean_data(df):
    """Clean the Airbnb listings dataset"""
    print("\n=== Cleaning Data ===")
    print(f"Original shape: {df.shape}")
    
    # 1. Clean the price column
    if 'price' in df.columns:
        df['price'] = df['price'].str.replace('$', '', regex=False)
        df['price'] = df['price'].str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        print(f"✓ Cleaned price column")
    
    # 2. Remove duplicates
    original_count = len(df)
    df = df.drop_duplicates(subset=['id'])
    duplicates_removed = original_count - len(df)
    print(f"✓ Removed {duplicates_removed} duplicate listings")
    
    # 3. Select useful columns
    useful_columns = [
        'id', 'name', 'description', 'neighborhood_overview',
        'host_id', 'host_name', 'neighbourhood_cleansed', 
        'neighbourhood_group_cleansed', 'latitude', 'longitude',
        'property_type', 'room_type', 'accommodates', 'bathrooms_text',
        'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights',
        'maximum_nights', 'number_of_reviews', 'review_scores_rating',
        'instant_bookable', 'reviews_per_month'
    ]
    
    existing_columns = [col for col in useful_columns if col in df.columns]
    df = df[existing_columns]
    print(f"✓ Selected {len(existing_columns)} useful columns")
    
    # 4. Remove listings with missing essential data
    df = df.dropna(subset=['name', 'price'])
    print(f"✓ Removed listings with missing name or price")
    
    # 5. Remove invalid price values
    df = df[df['price'] > 0]
    print(f"✓ Removed listings with invalid prices")
    
    # 6. Remove extreme outliers in price
    df = df[df['price'] <= 10000]
    print(f"✓ Removed listings with extreme prices (> $10,000)")
    
    print(f"\nFinal shape after cleaning: {df.shape}")
    
    return df


def print_summary_statistics(df):
    """Print summary statistics of the cleaned data"""
    print("\n=== Summary Statistics ===")
    print(f"Total listings: {len(df)}")
    print(f"\nPrice statistics:")
    print(df['price'].describe())
    
    if 'room_type' in df.columns:
        print(f"\nRoom type distribution:")
        print(df['room_type'].value_counts())
    
    if 'neighbourhood_group_cleansed' in df.columns:
        print(f"\nTop 5 neighborhoods:")
        print(df['neighbourhood_group_cleansed'].value_counts().head(5))


def main():
    """Main test function"""
    print("NYC Airbnb Data Cleaning Test")
    print("=" * 60)
    
    # Step 1: Download data
    data_path = download_nyc_data()
    if not data_path:
        print("\n✗ Test failed: Could not download data")
        return False
    
    # Step 2: Load data
    print("\nLoading data...")
    try:
        df = pd.read_csv(data_path, compression='gzip')
        print(f"✓ Loaded {len(df)} listings with {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Step 3: Clean data
    try:
        df_clean = clean_data(df)
        print(f"✓ Data cleaning completed successfully")
    except Exception as e:
        print(f"✗ Error cleaning data: {e}")
        return False
    
    # Step 4: Print summary
    try:
        print_summary_statistics(df_clean)
    except Exception as e:
        print(f"✗ Error generating statistics: {e}")
        return False
    
    # Step 5: Save cleaned data
    try:
        output_path = "data/nyc_listings_cleaned.csv"
        df_clean.to_csv(output_path, index=False)
        print(f"\n✓ Saved cleaned data to {output_path}")
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print(f"✓ Cleaned dataset contains {len(df_clean)} listings")
    print("\nNote: Embedding generation requires network access to download models.")
    print("The main implementation (nyc_airbnb_analysis.py) includes full embedding functionality.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
