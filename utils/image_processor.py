import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PIL import Image
import io
import os
from tensorflow.keras.applications.vgg16 import preprocess_input

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
        # Open and preprocess the image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to VGG16 input size
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = model.predict(img_array)
        
        return features.flatten()
        
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        return None

def load_city_image(image_path, image_type='splash'):
    """
    Load city image from local directory.
    
    Args:
        image_path (str): Path to the city image
        image_type (str): Type of image ('splash' or 'serp')
        
    Returns:
        PIL.Image: Loaded image or None if failed
    """
    try:
        if not image_path:
            return None
            
        # Adjust path based on image type
        base_dir = "images_database"
        if image_type == 'splash':
            image_dir = os.path.join(base_dir, "images_splash")
        else:
            image_dir = os.path.join(base_dir, "images_serp")
            
        # Construct full path
        full_path = os.path.join(image_dir, os.path.basename(image_path))
        
        # Check if file exists
        if os.path.exists(full_path):
            return Image.open(full_path)
            
        # Try alternative image type if first one fails
        alt_type = 'serp' if image_type == 'splash' else 'splash'
        alt_dir = os.path.join(base_dir, f"images_{alt_type}")
        alt_path = os.path.join(alt_dir, os.path.basename(image_path))
        
        if os.path.exists(alt_path):
            return Image.open(alt_path)
            
        return None
    except Exception as e:
        print(f"Error loading city image: {str(e)}")
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
        if query_features is None:
            return pd.DataFrame()
            
        # Ensure query_features is 2D
        query_features = query_features.reshape(1, -1)
        
        # Calculate similarities
        similarities = []
        for idx, row in image_df.iterrows():
            if 'features' in row and row['features'] is not None:
                city_features = np.array(row['features']).reshape(1, -1)
                sim = cosine_similarity(query_features, city_features)[0][0]
                similarities.append((idx, sim))
        
        if not similarities:
            return pd.DataFrame()
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N unique cities
        seen_cities = set()
        top_indices = []
        for idx, _ in similarities:
            city = image_df.iloc[idx]['city']
            if city not in seen_cities and len(top_indices) < top_n:
                seen_cities.add(city)
                top_indices.append(idx)
        
        # Create result DataFrame
        result_df = image_df.iloc[top_indices].copy()
        
        # Add similarity scores
        result_df['similarity'] = [sim for _, sim in similarities[:len(top_indices)]]
        
        return result_df
        
    except Exception as e:
        print(f"Error in find_top_matches_city: {str(e)}")
        return pd.DataFrame()

def verify_image_paths():
    """
    Verify that the image directories exist and are accessible.
    
    Returns:
        tuple: (bool, str) indicating success/failure and any error message
    """
    try:
        base_dir = "images_database"
        required_dirs = ["images_splash", "images_serp"]
        
        if not os.path.exists(base_dir):
            return False, f"Base directory '{base_dir}' not found"
            
        for dir_name in required_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            if not os.path.exists(dir_path):
                return False, f"Required directory '{dir_name}' not found in {base_dir}"
                
        return True, "Image directories verified successfully"
    except Exception as e:
        return False, f"Error verifying image paths: {str(e)}"
