import streamlit as st
import pandas as pd
from tensorflow.keras.applications import VGG16
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import spacy
import os
import random
from PIL import Image
from utils.data_handler import load_features
from utils.image_processor import extract_features, find_top_matches_city
from utils.hotel_recommender import top_hotels, display_hotel_card
from utils.chatbot import setup_chatbot, extract_city_nlp
import requests
from io import BytesIO

# Set page config
st.set_page_config(page_title="Travel Recommender & Chatbot", layout="wide")

# Google Drive file ID
GDRIVE_FILE_ID = "1lF_srcpxCneo7TgCtNgEol8HD-cEg7PL"
GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Load custom CSS
def load_custom_css():
    try:
        with open('assets/styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")
    except Exception as e:
        st.error(f"Error loading CSS: {str(e)}")

@st.cache_resource(show_spinner=False)
def load_models_and_data():
    try:
        # Load HuggingFace token from environment variable
        hf_token = "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"
        if not hf_token:
            st.error("HuggingFace token not found. Please set HUGGINGFACE_TOKEN environment variable.")
            st.stop()
        
        login(hf_token)
        
        # Load pickle file from Google Drive
        try:
            response = requests.get(GDRIVE_DOWNLOAD_URL)
            if response.status_code != 200:
                st.error("Failed to download file from Google Drive")
                st.stop()
            image_df = pd.read_pickle(BytesIO(response.content))
        except Exception as e:
            st.error(f"Error loading pickle file from Google Drive: {str(e)}")
            st.stop()

        # Load other models and data
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        df = pd.read_csv('data/combined.csv')
        
        # Load NLP model
        nlp = spacy.load("en_core_web_sm")
        
        # Setup LLM and embeddings
        hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = HuggingFaceEndpoint(repo_id=hf_model)
        
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Load vector database
        vector_db = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        return image_df, model, df, nlp, llm, vector_db
    
    except FileNotFoundError as e:
        st.error(f"Required file not found: {str(e)}")
        st.error("Please ensure all required files are present in the data directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models and data: {str(e)}")
        st.error("Please check your setup and try again.")
        st.stop()

# Rest of the code remains unchanged
def init_session_state():
    state_vars = {
        'budget': 100,
        'number_of_rooms': 1,
        'image_features': None,
        'top_cities': pd.DataFrame(),
        'uploaded_image': None,
        'image_analyzed': False,
        'messages': [],
        'hotel_chat': False
    }
    
    for var, default in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def display_hotel_recommendations(city, budget, number_of_rooms, df):
    try:
        st.subheader(f"🏨 Top Hotels in {city}")
        recommendations = top_hotels(city, budget, number_of_rooms, df)
        if not recommendations.empty:
            for _, row in recommendations.head(5).iterrows():
                st.markdown(display_hotel_card(row), unsafe_allow_html=True)
        else:
            st.warning(f"Sorry, no hotels match your criteria in {city}.")
    except Exception as e:
        st.error(f"Error displaying hotel recommendations: {str(e)}")

def hotel_chatbot(df):
    try:
        st.subheader("Hotel Recommender")
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = pd.DataFrame()
        
        city = st.session_state.get("selected_city", "").strip().title()
        
        budget = st.number_input("What is your budget per night?", min_value=0.0, step=1.0)
        number_of_rooms = st.number_input("How many rooms do you need?", min_value=1, step=1)

        if st.button("Find Hotels"):
            if city and budget > 0 and number_of_rooms > 0:
                recommendations = top_hotels(city, budget, number_of_rooms, df)
                if recommendations.empty:
                    st.write(f"Sorry, no hotels match your criteria in {city}.")
                else:
                    st.write(f"Here are the top hotel recommendations for you in {city}:")
                    display_columns = ['city', 'country', 'hotel', 'number_of_rooms', 'budget', 'website']
                    st.dataframe(recommendations[display_columns].head(5))
            else:
                st.warning("Please enter all the required fields correctly.")
        
        if st.button("Clear Search"):
            st.session_state.recommendations = pd.DataFrame()
            st.session_state.hotel_chat = False
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Error in hotel chatbot: {str(e)}")

def main():
    try:
        # Initialize session state
        init_session_state()
        
        # Load custom CSS
        load_custom_css()
        
        # Load models and data
        image_df, model, df, nlp, llm, vector_db = load_models_and_data()
        
        # Setup chatbot
        chain = setup_chatbot(llm, vector_db)
        
        # Main layout
        st.markdown("<h1 class='header-title'>🌍 Travel Recommender & Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subheader-title'>Discover your next destination and perfect stay!</p>", unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("Upload Your Image")
            uploaded_file = st.file_uploader("Choose an image that inspires your travel mood", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None and uploaded_file != st.session_state.uploaded_image:
                st.session_state.uploaded_image = uploaded_file
                st.session_state.image_features = None
                st.session_state.top_cities = pd.DataFrame()
                st.session_state.image_analyzed = False
            
            st.markdown("---")
            st.header("Hotel Preferences")
            st.session_state.budget = st.number_input("Budget per Night ($)", 
                                                    min_value=0, 
                                                    max_value=1000, 
                                                    value=st.session_state.budget)
            st.session_state.number_of_rooms = st.number_input("Number of Rooms", 
                                                             min_value=1, 
                                                             max_value=10, 
                                                             value=st.session_state.number_of_rooms)
        
        # Main content tabs
        tab1, tab2 = st.tabs(["Image-Based Search", "Travel Chatbot"])
        
        # Image-Based Search Tab
        with tab1:
            if st.session_state.uploaded_image:
                st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
                
                if not st.session_state.image_analyzed:
                    with st.spinner('Analyzing image and fetching recommendations...'):
                        image_data = st.session_state.uploaded_image.read()
                        st.session_state.image_features = extract_features(image_data, model)
                        if st.session_state.image_features is not None:
                            st.session_state.top_cities = find_top_matches_city(
                                st.session_state.image_features, 
                                image_df
                            )
                            st.session_state.image_analyzed = True
                        else:
                            st.error("Failed to extract features from the image. Please try again with a different image.")
                
                if not st.session_state.top_cities.empty:
                    st.subheader("Top Matching Destinations")
                    cols = st.columns(3)
                    for idx, row in st.session_state.top_cities.iterrows():
                        with cols[idx]:
                            try:
                                city_image = Image.open(row['image_path'])
                                st.image(city_image, caption=f"{row['city']}", use_column_width=True)
                                if st.button(f"Select {row['city']}", key=f"select_{idx}"):
                                    display_hotel_recommendations(
                                        row['city'],
                                        st.session_state.budget,
                                        st.session_state.number_of_rooms,
                                        df
                                    )
                            except Exception as e:
                                st.error(f"Error displaying city image: {str(e)}")
        
        # Chatbot Tab
        with tab2:
            st.subheader("Chat with our Travel Assistant")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Handle user input
            if prompt := st.chat_input("What's your travel question?"):
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                try:
                    if "inspire me where to go" in prompt.lower():
                        suggested_cities = random.sample(df['city'].unique().tolist(), 3)
                        response = f"Here are some inspiring destinations for you:\n\n" + "\n".join([f"- {city}" for city in suggested_cities])
                    elif "hotel" in prompt.lower():
                        city = extract_city_nlp(prompt, nlp)
                        st.session_state.selected_city = city
                        response = f"Sure! Let's find a hotel for you in {city}."
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.hotel_chat = True
                        st.experimental_rerun()
                    else:
                        chain_response = chain.invoke({"question": prompt})
                        response = chain_response['answer']
                    
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error processing your request: {str(e)}")
        
        # Show hotel chatbot if requested
        if st.session_state.get("hotel_chat"):
            hotel_chatbot(df)
        
        # Footer
        st.markdown(
            "<div class='footer'>Developed with ❤️ by "
            "<a href='https://github.com/yourusername' target='_blank'>Your Name</a></div>", 
            unsafe_allow_html=True
        )
    
    except Exception as e:
        st.error(f"An error occurred in the main application: {str(e)}")
        st.error("Please refresh the page and try again. If the error persists, contact support.")

if __name__ == "__main__":
    main()
