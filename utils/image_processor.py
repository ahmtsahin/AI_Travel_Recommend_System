import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from .data_handler import load_features

def extract_features(image_data, model):
    """
    Extract features from an image using the provided model.
    
    Args:
        image_data (bytes): Raw image data
        model: Pre-trained model for feature extraction
        
    Returns:
        numpy.ndarray: Extracted features
    """
    try:
        # Load and preprocess the image
        preprocessed_image = load_features(image_data)
        
        if preprocessed_image is None:
            return None
            
        # Extract features using the model
        features = model.predict(preprocessed_image)
        
        return features
        
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        return None

def find_top_matches_city(query_features, image_df, top_n=3):
    """
    Find top matching cities based on image features.
    
    Args:
        query_features (numpy.ndarray): Features of the query image
        image_df (pandas.DataFrame): DataFrame containing city images and their features
        top_n (int): Number of top matches to return
        
    Returns:
        pandas.DataFrame: Top matching cities with their information
    """
    try:
        # Calculate similarity scores
        similarities = []
        for idx, row in image_df.iterrows():
            sim_score = cosine_similarity(
                query_features.reshape(1, -1),
                row['features'].reshape(1, -1)
            )[0][0]
            similarities.append((idx, sim_score))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N matches
        top_indices = [idx for idx, _ in similarities[:top_n]]
        top_matches = image_df.iloc[top_indices].copy()
        
        # Add similarity scores
        top_matches['similarity_score'] = [sim for _, sim in similarities[:top_n]]
        
        return top_matches
        
    except Exception as e:
        print(f"Error in find_top_matches_city: {str(e)}")
        return pd.DataFrame()
