import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PIL import Image
import io
import requests
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

# Google Drive folder IDs
GDRIVE_FOLDER_ID = "1QZB_4f3XQozXlnDv5NPU9XnbOABkGvYO"
SPLASH_FOLDER_ID = "1f9HKf9YpyJRFu97U95GM5UHcuJAtV7P1"
SERP_FOLDER_ID = "1DX0idsg7fAQ_2N3d9fAeq_ffqxNJl12A"
PEXEL_FOLDER_ID = "1YV3P0oHAciPJqlzKni_uRUZbx1BEC7iO"
FLICKR_FOLDER_ID = "15isvmtC5W0ZDsDioK5ZNcYuenoB3StcS"

def get_gdrive_direct_link(folder_id, filename):
    """Convert Google Drive folder ID and filename to direct download link"""
    return f"https://drive.google.com/uc?id={folder_id}/{filename}"

def extract_features(image_data, model):
    """Extract features from an image using the provided model."""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        return None

def load_city_image_from_gdrive(local_path):
    """
    Load city image from Google Drive using the local path information.
    
    Args:
        local_path (str): Original local path from pickle file
        
    Returns:
        PIL.Image: Loaded image or None if failed
    """
    try:
        if not local_path:
            return None

        # Extract filename and determine source folder from path
        filename = os.path.basename(local_path)
        folder_type = None

        if 'splash' in local_path.lower():
            folder_id = SPLASH_FOLDER_ID
            folder_type = 'splash'
        elif 'serp' in local_path.lower():
            folder_id = SERP_FOLDER_ID
            folder_type = 'serp'
        elif 'pexel' in local_path.lower():
            folder_id = PEXEL_FOLDER_ID
            folder_type = 'pexel'
        elif 'flickr' in local_path.lower():
            folder_id = FLICKR_FOLDER_ID
            folder_type = 'flickr'
        else:
            return None

        # Try to load from determined folder
        image_url = f"https://drive.google.com/uc?id={folder_id}/{filename}"
        response = requests.get(image_url)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))

        # If primary source fails, try other folders in order
        folder_ids = {
            'splash': SPLASH_FOLDER_ID,
            'serp': SERP_FOLDER_ID,
            'pexel': PEXEL_FOLDER_ID,
            'flickr': FLICKR_FOLDER_ID
        }
        
        # Remove the already tried folder
        if folder_type:
            del folder_ids[folder_type]

        # Try remaining folders
        for backup_folder_id in folder_ids.values():
            backup_url = f"https://drive.google.com/uc?id={backup_folder_id}/{filename}"
            response = requests.get(backup_url)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))

        return None
    except Exception as e:
        print(f"Error loading city image from Google Drive: {str(e)}")
        return None

def find_top_matches_city(query_features, image_df, top_n=3):
    """Find top matching cities based on image features."""
    try:
        if query_features is None:
            return pd.DataFrame()
            
        query_features = query_features.reshape(1, -1)
        similarities = []
        
        for idx, row in image_df.iterrows():
            if 'features' in row and row['features'] is not None:
                city_features = np.array(row['features']).reshape(1, -1)
                sim = cosine_similarity(query_features, city_features)[0][0]
                similarities.append((idx, sim))
        
        if not similarities:
            return pd.DataFrame()
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        seen_cities = set()
        top_indices = []
        
        for idx, _ in similarities:
            city = image_df.iloc[idx]['city']
            if city not in seen_cities and len(top_indices) < top_n:
                seen_cities.add(city)
                top_indices.append(idx)
        
        result_df = image_df.iloc[top_indices].copy()
        result_df['similarity'] = [sim for _, sim in similarities[:len(top_indices)]]
        
        return result_df
        
    except Exception as e:
        print(f"Error in find_top_matches_city: {str(e)}")
        return pd.DataFrame()
