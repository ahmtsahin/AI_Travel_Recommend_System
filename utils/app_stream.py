import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import os
from io import BytesIO
from PIL import Image
from streamlit.components.v1 import html
from huggingface_hub import login
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import random
import spacy
from streamlit_super_slider import st_slider
import gdown
import requests

# Constants for Google Drive
GDRIVE_FOLDER_ID = "1QZB_4f3XQozXlnDv5NPU9XnbOABkGvYO"
SPLASH_FOLDER_ID = "1f9HKf9YpyJRFu97U95GM5UHcuJAtV7P1"
SERP_FOLDER_ID = "1DX0idsg7fAQ_2N3d9fAeq_ffqxNJl12A"
PEXEL_FOLDER_ID = "1YV3P0oHAciPJqlzKni_uRUZbx1BEC7iO"
FLICKR_FOLDER_ID = "15isvmtC5W0ZDsDioK5ZNcYuenoB3StcS"

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    if not os.path.exists(destination):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

# Download and load pickle file
@st.cache_resource
def load_pickle_file():
    pickle_file_id = "1lF_srcpxCneo7TgCtNgEol8HD-cEg7PL"
    pickle_path = "image_features_model_2.pkl"
    download_file_from_google_drive(pickle_file_id, pickle_path)
    return pd.read_pickle(pickle_path)

# Authenticate with Hugging Face
@st.cache_resource
def setup_huggingface():
    login(st.secrets["hf_token"])

# Load model and embeddings
@st.cache_resource
def setup_llm():
    hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
    return HuggingFaceEndpoint(repo_id=hf_model)

@st.cache_resource
def setup_embeddings():
    embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
    return HuggingFaceEmbeddings(model_name=embedding_model)

# Load Data and Models
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    image_df = load_pickle_file()
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    df = pd.read_csv('data/combined.csv')  # Move this to a data folder in your repo
    return image_df, model, df

# Function to get image from Google Drive
def get_image_from_drive(folder_id, image_path):
    # Extract the image filename from the path
    filename = os.path.basename(image_path)
    # Construct the full Google Drive URL
    url = f"https://drive.google.com/uc?id={folder_id}/{filename}"
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Modified find_top_matches_city function
def find_top_matches_city(image_features, image_df, top_n=3):
    image_df['similarity'] = image_df['features'].apply(
        lambda x: cosine_similarity([image_features], [x]).flatten()[0]
    )
    top_matches = image_df.loc[image_df.groupby('city')['similarity'].idxmax()]
    top_matches = top_matches.sort_values(by='similarity', ascending=False).head(top_n)
    return top_matches.reset_index(drop=True)

# Rest of your functions remain the same
[... rest of your existing functions ...]

def main():
    # Initialize HuggingFace and models
    setup_huggingface()
    llm_stat = setup_llm()
    embeddings_stat = setup_embeddings()
    
    # Load vector database
    vector_db = FAISS.load_local("faiss_index", embeddings_stat, allow_dangerous_deserialization=True)
    
    # Rest of your main function remains the same
    [... rest of your main function code ...]

if __name__ == "__main__":
    main()
