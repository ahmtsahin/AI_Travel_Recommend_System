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
        try:
            # Display uploaded image
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            if not st.session_state.image_analyzed:
                with st.spinner('Analyzing image and fetching recommendations...'):
                    # Read image data
                    image_data = st.session_state.uploaded_image.read()
                    
                    # Extract features
                    features = extract_features(image_data, model)
                    
                    if features is not None:
                        st.session_state.image_features = features
                        st.session_state.top_cities = find_top_matches_city(
                            features,
                            image_df,
                            top_n=3
                        )
                        st.session_state.image_analyzed = True
                    else:
                        st.error("Failed to extract features from the image. Please try a different image.")
            
            # Display recommendations if available
            if not st.session_state.top_cities.empty:
                st.subheader("Top Matching Destinations")
                
                # Create columns for displaying cities
                cols = st.columns(min(3, len(st.session_state.top_cities)))
                
                # Display each city
                for idx, (_, row) in enumerate(st.session_state.top_cities.iterrows()):
                    if idx < len(cols):  # Ensure we don't exceed available columns
                        with cols[idx]:
                            try:
                                # Display city name and similarity score
                                st.markdown(f"### {row['city']}")
                                if 'similarity' in row:
                                    st.markdown(f"Match Score: {row['similarity']*100:.1f}%")
                                
                                # Display city image if available
                                if 'image_path' in row and row['image_path']:
                                    try:
                                        city_image = Image.open(row['image_path'])
                                        st.image(city_image, use_column_width=True)
                                    except Exception as e:
                                        st.warning("City image not available")
                                
                                # Add selection button
                                if st.button(f"Select {row['city']}", key=f"select_{idx}"):
                                    display_hotel_recommendations(
                                        row['city'],
                                        st.session_state.budget,
                                        st.session_state.number_of_rooms,
                                        df
                                    )
                            except Exception as e:
                                st.error(f"Error displaying city {idx+1}: {str(e)}")
            else:
                if st.session_state.image_analyzed:
                    st.warning("No matching destinations found. Please try a different image.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.info("👆 Upload an image in the sidebar to get destination recommendations!")
        
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
