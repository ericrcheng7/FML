"""
Example: Using embeddings for semantic search of NYC Airbnb listings

This script demonstrates how to use the generated embeddings to perform
semantic search on the cleaned NYC Airbnb listings data.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedded_data(data_path="data/nyc_listings_cleaned_embedded.parquet"):
    """
    Load the cleaned and embedded data
    
    Args:
        data_path: Path to the parquet file with embeddings
        
    Returns:
        DataFrame with listings and embeddings
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} listings")
    return df


def semantic_search(query, df, model, top_k=5):
    """
    Perform semantic search on the listings
    
    Args:
        query: Search query (e.g., "luxury apartment near Central Park")
        df: DataFrame with listings and embeddings
        model: SentenceTransformer model
        top_k: Number of results to return
        
    Returns:
        DataFrame with top matching listings
    """
    # Encode the query with the proper prefix
    query_text = f"search_query: {query}"
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    
    # Convert all embeddings to a matrix
    embedding_matrix = np.stack(df['embedding'].values)
    
    # Calculate similarity scores (dot product for normalized vectors)
    scores = embedding_matrix @ query_embedding
    
    # Get top k indices
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    
    # Get the top listings
    results = df.iloc[top_k_indices].copy()
    results['similarity_score'] = scores[top_k_indices]
    
    return results


def display_results(results, query):
    """
    Display search results in a readable format
    
    Args:
        results: DataFrame with search results
        query: The search query
    """
    print(f"\n{'='*80}")
    print(f"Search Results for: '{query}'")
    print(f"{'='*80}\n")
    
    for i, (idx, row) in enumerate(results.iterrows(), 1):
        print(f"{i}. {row['name']}")
        print(f"   Similarity Score: {row['similarity_score']:.4f}")
        print(f"   Price: ${row['price']:.2f}/night")
        print(f"   Room Type: {row.get('room_type', 'N/A')}")
        print(f"   Location: {row.get('neighbourhood_cleansed', 'N/A')}")
        
        if pd.notna(row.get('description')):
            desc = str(row['description'])
            # Truncate long descriptions
            if len(desc) > 150:
                desc = desc[:150] + "..."
            print(f"   Description: {desc}")
        
        print()


def find_similar_listings(listing_id, df, top_k=5):
    """
    Find listings similar to a given listing
    
    Args:
        listing_id: ID of the listing to find similar ones for
        df: DataFrame with listings and embeddings
        top_k: Number of results to return
        
    Returns:
        DataFrame with similar listings
    """
    # Find the listing
    listing_idx = df[df['id'] == listing_id].index[0]
    query_embedding = df.iloc[listing_idx]['embedding']
    
    # Convert all embeddings to a matrix
    embedding_matrix = np.stack(df['embedding'].values)
    
    # Calculate similarity scores
    scores = embedding_matrix @ query_embedding
    
    # Get top k indices (excluding the query listing itself)
    top_k_indices = np.argsort(scores)[::-1][1:top_k+1]
    
    # Get the top listings
    results = df.iloc[top_k_indices].copy()
    results['similarity_score'] = scores[top_k_indices]
    
    return results


def main():
    """
    Main function demonstrating semantic search capabilities
    """
    print("NYC Airbnb Semantic Search Example")
    print("="*80)
    
    # Load the embedded data
    try:
        df = load_embedded_data()
    except FileNotFoundError:
        print("\nError: Embedded data not found!")
        print("Please run 'python nyc_airbnb_analysis.py' first to generate the data.")
        return
    
    # Load the model
    print("\nLoading SentenceTransformer model...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    print("Model loaded!")
    
    # Example 1: Semantic search with various queries
    queries = [
        "luxury apartment near Central Park",
        "cozy studio for budget travelers",
        "spacious family home with backyard",
        "modern loft in artistic neighborhood",
        "romantic getaway with skyline views"
    ]
    
    for query in queries:
        results = semantic_search(query, df, model, top_k=3)
        display_results(results, query)
    
    # Example 2: Find similar listings
    if len(df) > 0:
        sample_id = df.iloc[0]['id']
        print(f"\n{'='*80}")
        print(f"Finding listings similar to: {df.iloc[0]['name']}")
        print(f"{'='*80}\n")
        
        similar = find_similar_listings(sample_id, df, top_k=3)
        
        for i, (idx, row) in enumerate(similar.iterrows(), 1):
            print(f"{i}. {row['name']}")
            print(f"   Similarity Score: {row['similarity_score']:.4f}")
            print(f"   Price: ${row['price']:.2f}/night")
            print(f"   Room Type: {row.get('room_type', 'N/A')}")
            print()
    
    print("\nSemantic search demo complete!")
    print("\nYou can modify this script to:")
    print("  - Try your own search queries")
    print("  - Adjust the number of results (top_k)")
    print("  - Filter by price range, location, or other features")
    print("  - Build a recommendation system")


if __name__ == "__main__":
    main()
