import pandas as pd
import numpy as np
import pickle
import streamlit as st

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    # Create sample hotel data
    sample_hotels = pd.DataFrame({
        'city': ['Paris', 'London', 'New York', 'Tokyo', 'Dubai'],
        'country': ['France', 'UK', 'USA', 'Japan', 'UAE'],
        'hotel': ['Grand Hotel Paris', 'London Luxury', 'NYC Plaza', 'Tokyo Star', 'Dubai Palm'],
        'number_of_rooms': [100, 150, 200, 180, 250],
        'budget': [200, 300, 400, 350, 500],
        'website': ['https://example.com'] * 5
    })
    return sample_hotels

@st.cache_data
def load_sample_features():
    """Load sample image features for demonstration"""
    # Create sample image features
    sample_features = pd.DataFrame({
        'city': ['Paris', 'London', 'New York', 'Tokyo', 'Dubai'],
        'features': [np.random.rand(512) for _ in range(5)],  # Sample 512-dimensional features
        'image_path': ['sample_images/paris.jpg', 'sample_images/london.jpg', 
                      'sample_images/newyork.jpg', 'sample_images/tokyo.jpg', 
                      'sample_images/dubai.jpg']
    })
    return sample_features

def verify_data_files():
    """Verify that all required files are present."""
    return True, []  # Always return success for demo
