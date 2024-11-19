import streamlit as st
import pandas as pd
import os
import random
from PIL import Image
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow and ensure tf-keras is used
import tensorflow as tf
tf.keras.backend.clear_session()  # Clear any existing Keras sessions

# Force TensorFlow to use tf-keras
os.environ['TF_KERAS'] = '1'

# Import VGG16 from tf-keras applications
try:
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
except ImportError:
    st.error("Failed to import VGG16 from tensorflow.keras")
    st.stop()

# Rest of the imports
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import spacy

from utils.data_handler import load_features
from utils.image_processor import extract_features, find_top_matches_city
from utils.hotel_recommender import top_hotels, display_hotel_card
from utils.chatbot import setup_chatbot, extract_city_nlp

# Set page config
st.set_page_config(page_title="Travel Recommender & Chatbot", layout="wide")

# Custom CSS definition (same as before)
def load_custom_css():
    """Load custom CSS styles"""
    css = """
    <style>
        .header-title {
            font-size: 3rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
            padding: 1rem;
        }
        
        .subheader-title {
            font-size: 1.5rem;
            color: #424242;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .stButton>button {
            width: 100%;
            background-color: #1E88E5;
            color: white;
        }
        
        .stButton>button:hover {
            background-color: #1565C0;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 0.8rem;
            color: #666;
            margin-top: 50px;
        }
        
        .hotel-card {
            border: 1px solid #ddd;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .hotel-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
            transition: all 0.2s ease;
        }
        
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        
        .stTextInput>div>div>input {
            background-color: white;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_models_and_data():
    try:
        # Verify TensorFlow and Keras versions
        tf_version = tf.__version__
        keras_version = tf.keras.__version__
        st.info(f"TensorFlow version: {tf_version}")
        st.info(f"Keras version: {keras_version}")
        
        # Load HuggingFace token
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN", "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")
        if not hf_token:
            st.error("HuggingFace token not found in environment variables.")
            st.stop()
        
        # Initialize HuggingFace
        login(hf_token)
        
        # Load pickle file from Google Drive
        GDRIVE_FILE_ID = "1lF_srcpxCneo7TgCtNgEol8HD-cEg7PL"
        GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        
        response = requests.get(GDRIVE_DOWNLOAD_URL)
        if response.status_code != 200:
            st.error("Failed to download file from Google Drive")
            st.stop()
        image_df = pd.read_pickle(BytesIO(response.content))
        
        # Load VGG16 model
        try:
            model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        except Exception as e:
            st.error(f"Error loading VGG16 model: {str(e)}")
            st.stop()
        
        # Load other components
        df = pd.read_csv('data/combined.csv')
        nlp = spacy.load("en_core_web_sm")
        
        # Setup LLM and embeddings
        hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = HuggingFaceEndpoint(
            repo_id=hf_model,
            token=hf_token,
            task="text-generation",
            temperature=0.7,
            max_length=512
        )
        
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector database
        vector_db = FAISS.load_local(
            "data/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        return image_df, model, df, nlp, llm, vector_db
    
    except Exception as e:
        st.error(f"Error loading models and data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()

def init_session_state():
    """Initialize session state variables"""
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
    """Display hotel recommendations for a given city"""
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

def main():
    try:
        # Load custom CSS
        load_custom_css()
        
        # Initialize session state
        init_session_state()
        
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
        
        # Rest of your main() function code...
        
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
