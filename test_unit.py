"""
Unit tests for NYC Airbnb data cleaning functions

This script tests the data cleaning functionality with mock data
to verify the implementation works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from nyc_airbnb_analysis import clean_data, create_text_for_embedding


def create_mock_data():
    """Create mock NYC Airbnb data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5],  # Note: 5 is duplicated
        'name': ['Cozy Studio in Manhattan', 'Spacious Brooklyn Loft', 'Luxury Penthouse', 'Budget Room', None, 'Another Room'],
        'description': ['A cozy place', 'Great location', 'Amazing views', 'Simple room', 'No desc', 'Duplicate room'],
        'neighborhood_overview': ['Great area', 'Nice neighborhood', 'Upscale', 'Basic', 'Area info', 'Area info'],
        'host_id': [101, 102, 103, 104, 105, 105],
        'host_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Charlie'],
        'neighbourhood_cleansed': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Bronx'],
        'neighbourhood_group_cleansed': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Bronx'],
        'latitude': [40.7589, 40.6782, 40.7614, 40.7282, 40.8448, 40.8448],
        'longitude': [-73.9851, -73.9442, -73.9776, -73.7949, -73.8648, -73.8648],
        'property_type': ['Entire rental unit', 'Entire home', 'Entire home', 'Private room', 'Shared room', 'Shared room'],
        'room_type': ['Entire home/apt', 'Entire home/apt', 'Entire home/apt', 'Private room', 'Shared room', 'Shared room'],
        'accommodates': [2, 4, 6, 1, 1, 1],
        'bathrooms_text': ['1 bath', '2 baths', '3 baths', '1 shared bath', '1 shared bath', '1 shared bath'],
        'bedrooms': [1, 2, 3, 1, 1, 1],
        'beds': [1, 2, 3, 1, 1, 1],
        'amenities': ['["Wifi", "Kitchen"]', '["Wifi", "TV", "Gym"]', '["Wifi", "Pool", "Gym"]', '["Wifi"]', '["Wifi"]', '["Wifi"]'],
        'price': ['$150', '$200', '$-500', '$50', None, '$25'],  # Note: invalid prices
        'minimum_nights': [2, 1, 3, 1, 1, 1],
        'maximum_nights': [30, 365, 90, 30, 30, 30],
        'number_of_reviews': [45, 120, 5, 200, 0, 0],
        'review_scores_rating': [4.8, 4.9, 5.0, 4.2, None, None],
        'instant_bookable': ['t', 't', 'f', 't', 't', 't'],
        'reviews_per_month': [2.5, 5.2, 0.3, 8.1, 0.0, 0.0]
    })


def test_price_cleaning():
    """Test that price column is cleaned correctly"""
    print("\n=== Testing Price Cleaning ===")
    
    df = create_mock_data()
    print(f"Before cleaning: {df['price'].tolist()}")
    
    df_clean = clean_data(df)
    
    # Check that prices are numeric
    assert df_clean['price'].dtype in [np.float64, np.int64], "Price should be numeric"
    
    # Check that invalid prices are removed
    assert all(df_clean['price'] > 0), "All prices should be positive"
    
    print(f"After cleaning: {df_clean['price'].tolist()}")
    print("✓ Price cleaning test passed")
    
    return df_clean


def test_duplicate_removal():
    """Test that duplicates are removed"""
    print("\n=== Testing Duplicate Removal ===")
    
    df = create_mock_data()
    print(f"Before: {len(df)} rows (including duplicate id=5)")
    
    df_clean = clean_data(df)
    
    # Check that duplicates are removed
    assert len(df_clean['id'].unique()) == len(df_clean), "All IDs should be unique"
    
    print(f"After: {len(df_clean)} rows (duplicates removed)")
    print("✓ Duplicate removal test passed")
    
    return df_clean


def test_missing_data_removal():
    """Test that rows with missing essential data are removed"""
    print("\n=== Testing Missing Data Removal ===")
    
    df = create_mock_data()
    print(f"Before: {len(df)} rows")
    print(f"Missing name: {df['name'].isna().sum()}")
    print(f"Missing price: {df['price'].isna().sum()}")
    
    df_clean = clean_data(df)
    
    # Check that no missing names or prices
    assert df_clean['name'].notna().all(), "All names should be present"
    assert df_clean['price'].notna().all(), "All prices should be present"
    
    print(f"After: {len(df_clean)} rows (missing data removed)")
    print("✓ Missing data removal test passed")
    
    return df_clean


def test_text_embedding_creation():
    """Test that text for embedding is created correctly"""
    print("\n=== Testing Text Embedding Creation ===")
    
    df = create_mock_data()
    df_clean = clean_data(df)
    
    if len(df_clean) > 0:
        row = df_clean.iloc[0]
        text = create_text_for_embedding(row)
        
        print(f"Sample text for embedding:")
        print(f"{text[:200]}...")
        
        # Check that text contains key information
        assert 'Name:' in text, "Text should contain name"
        assert 'Room Type:' in text or 'Property Type:' in text, "Text should contain room/property type"
        
        print("✓ Text embedding creation test passed")
    else:
        print("⚠ No data to test text embedding creation")


def test_column_selection():
    """Test that only useful columns are kept"""
    print("\n=== Testing Column Selection ===")
    
    df = create_mock_data()
    print(f"Original columns: {len(df.columns)}")
    
    df_clean = clean_data(df)
    
    # Check that essential columns are present
    essential_columns = ['id', 'name', 'price', 'room_type']
    for col in essential_columns:
        if col in df.columns:
            assert col in df_clean.columns, f"Essential column {col} should be present"
    
    print(f"Selected columns: {len(df_clean.columns)}")
    print(f"Columns: {df_clean.columns.tolist()}")
    print("✓ Column selection test passed")


def main():
    """Run all tests"""
    print("NYC Airbnb Data Cleaning Unit Tests")
    print("=" * 60)
    
    try:
        # Run tests
        test_price_cleaning()
        test_duplicate_removal()
        test_missing_data_removal()
        test_column_selection()
        test_text_embedding_creation()
        
        print("\n" + "=" * 60)
        print("✓ All unit tests passed!")
        print("\nThe implementation correctly:")
        print("  1. Cleans the price column (removes $, commas, converts to numeric)")
        print("  2. Removes duplicate listings")
        print("  3. Removes listings with missing essential data")
        print("  4. Filters out invalid prices")
        print("  5. Selects useful columns")
        print("  6. Creates text representations for embeddings")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
