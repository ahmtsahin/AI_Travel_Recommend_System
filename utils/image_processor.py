import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PIL import Image
import io
import requests
from tensorflow.keras.applications.vgg16 import preprocess_input

# Google Drive folder IDs
GDRIVE_FOLDER_ID = "1QZB_4f3XQozXlnDv5NPU9XnbOABkGvYO"
SPLASH_FOLDER_ID = "1f9HKf9YpyJRFu97U95GM5UHcuJAtV7P1" 
SERP_FOLDER_ID = "1DX0idsg7fAQ_2N3d9fAeq_ffqxNJl12A"
PEXEL_FOLDER_ID = "1YV3P0oHAciPJqlzKni_uRUZbx1BEC7iO"
FLICKR_FOLDER_ID = "15isvmtC5W0ZDsDioK5ZNcYuenoB3StcS"


def get_gdrive_direct_link(file_id):
    """Convert Google Drive file ID to direct download link"""
    return f"https://drive.google.com/uc?id={file_id}"

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

def load_city_image_from_gdrive(image_name, image_type='splash'):
    """
    Load city image from Google Drive.
    
    Args:
        image_name (str): Name of the image file
        image_type (str): Type of image ('splash' or 'serp')
        
    Returns:
        PIL.Image: Loaded image or None if failed
    """
    try:
        if not image_name:
            return None
            
        # Get the appropriate folder ID based on image type
        folder_id = SPLASH_FOLDER_ID if image_type == 'splash' else SERP_FOLDER_ID
        
        # Construct Google Drive API request
        file_url = f"https://drive.google.com/uc?id={folder_id}/{image_name}"
        
        response = requests.get(file_url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
            
        return None
    except Exception as e:
        print(f"Error loading city image from Google Drive: {str(e)}")
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
        
        # Add similarity scores and adjust image paths
        result_df['similarity'] = [sim for _, sim in similarities[:len(top_indices)]]
        result_df['gdrive_image_id'] = result_df['image_path'].apply(
            lambda x: x.split('/')[-1] if x else None
        )
        
        return result_df
        
    except Exception as e:
        print(f"Error in find_top_matches_city: {str(e)}")
        return pd.DataFrame()
