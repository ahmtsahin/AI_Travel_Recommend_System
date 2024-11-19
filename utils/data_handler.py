import numpy as np
import pandas as pd
from PIL import Image
import io

def load_features(image_data):
    """
    Load and preprocess image features from raw image data.
    
    Args:
        image_data (bytes): Raw image data
        
    Returns:
        numpy.ndarray: Processed image features
    """
    try:
        # Convert image data to numpy array
        image = Image.open(io.BytesIO(image_data))
        
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image to standard size (e.g., 224x224 for VGG16)
        image = image.resize((224, 224))
        
        # Convert to numpy array and preprocess
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error in load_features: {str(e)}")
        return None

def process_city_data(df):
    """
    Process city data from DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing city information
        
    Returns:
        pandas.DataFrame: Processed city data
    """
    try:
        # Remove duplicates
        df = df.drop_duplicates(subset=['city'])
        
        # Sort by city name
        df = df.sort_values('city')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error in process_city_data: {str(e)}")
        return pd.DataFrame()

def save_features(features, filepath):
    """
    Save extracted features to a file.
    
    Args:
        features (numpy.ndarray): Extracted image features
        filepath (str): Path to save the features
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        np.save(filepath, features)
        return True
    except Exception as e:
        print(f"Error in save_features: {str(e)}")
        return False
