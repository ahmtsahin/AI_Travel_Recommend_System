import streamlit as st
import pandas as pd
from tensorflow.keras.applications import VGG16
import numpy as np
from PIL import Image
import os
import random

from utils.image_processor import extract_features, find_top_matches_city
from utils.hotel_recommender import top_hotels, display_hotel_card
from utils.data_handler import load_sample_data, load_sample_features

# Set page config
st.set_page_config(page_title="Travel Recommender", layout="wide")

# Custom CSS
def load_custom_css():
    st.markdown("""
        <style>
        .header-title {
            font-size: 3em;
            font-weight: 700;
            color: #4A90E2;
            margin-bottom: 0.2em;
        }
        .subheader-title {
            font-size: 1.5em;
            color: #4A90E2;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .recommendation-card {
            background-color: white;
            padding: 1.5em;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5em;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return VGG16(weights='imagenet', include_top=False, pooling='avg')

def init_session_state():
    if 'budget' not in st.session_state:
        st.session_state.budget = 300
    if 'number_of_rooms' not in st.session_state:
        st.session_state.number_of_rooms = 1
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_hotel_recommendations(city, budget, number_of_rooms, df):
    st.subheader(f"🏨 Top Hotels in {city}")
    recommendations = top_hotels(city, budget, number_of_rooms, df)
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            st.markdown(f"""
                <div class='recommendation-card'>
                    <h3>{row['hotel']}</h3>
                    <p><strong>City:</strong> {row['city']}</p>
                    <p><strong>Country:</strong> {row['country']}</p>
                    <p><strong>Price per Night:</strong> ${row['budget']}</p>
                    <p><strong>Available Rooms:</strong> {row['number_of_rooms']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"Sorry, no hotels match your criteria in {city}.")

def main():
    try:
        # Initialize session state
        init_session_state()
        
        # Load custom CSS
        load_custom_css()
        
        # Load model and sample data
        model = load_model()
        df = load_sample_data()
        image_df = load_sample_features()
        
        # Main layout
        st.markdown("<h1 class='header-title'>🌍 Travel Recommender</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subheader-title'>Discover your next destination!</p>", unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("Upload Your Image")
            uploaded_file = st.file_uploader("Choose an image that inspires your travel mood", type=["jpg", "jpeg", "png"])
            
            st.markdown("---")
            st.header("Hotel Preferences")
            budget = st.number_input("Budget per Night ($)", min_value=50, max_value=1000, value=st.session_state.budget)
            number_of_rooms = st.number_input("Number of Rooms", min_value=1, max_value=5, value=st.session_state.number_of_rooms)
        
        # Main content
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner('Analyzing image...'):
                # Extract features from uploaded image
                image_features = extract_features(uploaded_file.read(), model)
                if image_features is not None:
                    # Find matching cities
                    matches = find_top_matches_city(image_features, image_df)
                    
                    st.subheader("Top Matching Destinations")
                    cols = st.columns(3)
                    for idx, (city, similarity) in enumerate(matches.iterrows()):
                        with cols[idx]:
                            st.write(f"### {city}")
                            st.write(f"Match Score: {similarity:.2f}")
                            if st.button(f"View Hotels in {city}"):
                                display_hotel_recommendations(city, budget, number_of_rooms, df)
        
        else:
            # Display sample destinations
            st.subheader("Popular Destinations")
            cols = st.columns(3)
            sample_cities = df['city'].unique()[:3]
            for idx, city in enumerate(sample_cities):
                with cols[idx]:
                    st.write(f"### {city}")
                    if st.button(f"View Hotels in {city}"):
                        display_hotel_recommendations(city, budget, number_of_rooms, df)
        
        # Footer
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
