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
import requests
from streamlit_super_slider import st_slider

# Configuration and Environment Variables
HF_TOKEN = st.secrets["HF_TOKEN"]  # Store your Hugging Face token in Streamlit secrets
GDRIVE_URL = "https://drive.google.com/file/d/1lF_srcpxCneo7TgCtNgEol8HD-cEg7PL/view?usp=drive_link"
GDRIVE_FILE_ID = GDRIVE_URL.split('/')[-2]
GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Authenticate with Hugging Face
login(HF_TOKEN)

# Load model and embeddings
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm_stat = HuggingFaceEndpoint(repo_id=hf_model)

# Load CSV data from GitHub/remote source
@st.cache_data
def load_csv_data():
    return pd.read_csv('combined.csv')  # Upload this to your GitHub repo

df = load_csv_data()

# Setup embeddings
@st.cache_resource
def setup_embeddings():
    embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
    return HuggingFaceEmbeddings(model_name=embedding_model)

embeddings_stat = setup_embeddings()

# Load vector database - this should be stored in your GitHub repo
@st.cache_resource
def load_vector_db():
    return FAISS.load_local("faiss_index", embeddings_stat, allow_dangerous_deserialization=True)

vector_db = load_vector_db()

# Load NLP model
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# Custom CSS remains the same
def load_custom_css():
    custom_css = """
    <style>
        /* General Styling */
       body {
            background-color: #c2c2c2 !important;
        }
        .stApp {
            background-color: #c2c2c2 !important;
        }
        /* ... rest of your CSS ... */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Load Data and Models with Google Drive Integration
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    # Download pickle file from Google Drive
    response = requests.get(GDRIVE_DOWNLOAD_URL)
    image_df = pd.read_pickle(BytesIO(response.content))
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    return image_df, model, df

# Feature Extraction Function
def extract_features(image_data, model):
    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img = img.resize((224, 224))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data)
        return features.flatten()
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def find_top_matches_city(image_features, image_df, top_n=3):
    image_df['similarity'] = image_df['features'].apply(
        lambda x: cosine_similarity([image_features], [x]).flatten()[0]
    )
    top_matches = image_df.loc[image_df.groupby('city')['similarity'].idxmax()]
    top_matches = top_matches.sort_values(by='similarity', ascending=False).head(top_n)
    return top_matches.reset_index(drop=True)

# ------------------------------
# Hotel Recommendation Function
# ------------------------------
def top_hotels(city, budget, number_of_rooms, df):
    filtered_df = df[
        (df['city'].str.lower() == city.lower()) &
        (df['budget'] <= budget) &
        (df['number_of_rooms'] >= number_of_rooms)
    ]
    return filtered_df.reset_index(drop=True)

# Setup retriever and memory for chatbot
retriever = vector_db.as_retriever(search_kwargs={"k": 4})
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

# Create prompt template
template = """You are a friendly travel recommendation chatbot. Answer the question based on the following context, previous conversation, and the information from the dataframe. 
If asked to inspire with travel destinations, suggest cities from the dataframe. When a user expresses interest in a specific city, provide relevant information including attractions, images, and links about this city.
Only ask about hotel preferences (like number of rooms and budget) if the user specifically requests hotel recommendations.

Previous conversation: {chat_history}
Context to answer question: {context}
New human question: {question}
Response:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

# Setup the conversational chain
chain = ConversationalRetrievalChain.from_llm(
    llm_stat,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt}
)

def main():
    st.markdown("<h1 class='header-title'>🌍 Travel Recommender & Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader-title'>Discover your next destination and perfect stay!</p>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'budget' not in st.session_state:
        st.session_state.budget = 100
    if 'number_of_rooms' not in st.session_state:
        st.session_state.number_of_rooms = 1
    if 'image_features' not in st.session_state:
        st.session_state.image_features = None
    if 'top_cities' not in st.session_state:
        st.session_state.top_cities = pd.DataFrame()
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'image_analyzed' not in st.session_state:
        st.session_state.image_analyzed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "hotel_chat" not in st.session_state:
        st.session_state.hotel_chat = False

    # Sidebar for image upload and preferences
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
        st.session_state.budget = st.number_input("Budget per Night ($)", min_value=0, max_value=1000, value=st.session_state.budget)
        st.session_state.number_of_rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=st.session_state.number_of_rooms)

    # Main area
    tab1, tab2 = st.tabs(["As similar as your Image", "Travel Chatbot"])

    with tab1:
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True,width=600)

        if st.session_state.uploaded_image and not st.session_state.image_analyzed:
            with st.spinner('Analyzing image and fetching recommendations...'):
                image_data = st.session_state.uploaded_image.read()
                st.session_state.image_features = extract_features(image_data, model)
                if st.session_state.image_features is not None:
                    st.session_state.top_cities = find_top_matches_city(st.session_state.image_features, image_df, top_n=3)
                    st.session_state.image_analyzed = True
                else:
                    st.error("Failed to extract features from the image. Please try again with a different image.")

        if not st.session_state.top_cities.empty:
            st.subheader("Top Matching Destinations")
            cols = st.columns(3)
            for idx, row in st.session_state.top_cities.iterrows():
                with cols[idx]:
                    city_image = Image.open(row['image_path'])
                    st.image(city_image, caption=f"{row['city']}", use_column_width=True,width=1000)
                    if st.button(f"Select {row['city']}", key=f"select_{idx}"):
                        display_hotel_recommendations(row['city'], st.session_state.budget, st.session_state.number_of_rooms)
    

    with tab2:
        st.subheader("Chat with our Travel Assistant")
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What's your travel question?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            if "inspire me where to go" in prompt.lower():
                suggested_cities = random.sample(df['city'].unique().tolist(), 3)
                response = f"Here are some inspiring destinations for you:\n\n" + "\n".join([f"- {city}" for city in suggested_cities])
            elif "hotel" in prompt.lower():
                city = extract_city_nlp(prompt)
                st.session_state.selected_city = city 
                response = f"Sure! Let's find a hotel for you in {st.session_state.get('selected_city', 'the city')}."
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.hotel_chat = True
                st.rerun()
            else:
                response = chain.invoke({"question": prompt})
                response = response['answer']

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.get("hotel_chat"):
        hotel_chatbot()

    st.markdown("<div class='footer'>Developed with ❤️ by <a href='https://openai.com' target='_blank'>Ahmet and Sina</a></div>", unsafe_allow_html=True)
# ------------------------------
# Display Hotel Recommendations
# ------------------------------
def display_hotel_recommendations(city, budget, number_of_rooms):
    st.subheader(f"🏨 Top Hotels in {city}")
    recommendations = top_hotels(city, budget, number_of_rooms, df)
    if not recommendations.empty:
        for idx, row in recommendations.head(5).iterrows():
            with st.container():
                st.markdown(f"""
                    <div class='recommendation-card'>
                        <h3>{row['hotel']}</h3>
                        <p><strong>City:</strong> {row['city']}</p>
                        <p><strong>Country:</strong> {row['country']}</p>
                        <p><strong>Price per Night:</strong> ${row['budget']}</p>
                        <p><strong>Available Rooms:</strong> {row['number_of_rooms']}</p>
                        <a href='{row['website']}' target='_blank' class='city-selection-button'>Book Now</a>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"Sorry, no hotels match your criteria in {city}.")
def hotel_chatbot():
    st.subheader("Hotel Recommender")
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = pd.DataFrame()
    
    city = st.session_state.get("selected_city", "").strip().title()
    
    budget = st.number_input("What is your budget per night?", min_value=0.0, step=1.0)
    number_of_rooms = st.number_input("How many rooms do you need?", min_value=1, step=1)

    if st.button("Find Hotels"):
        if city and budget > 0 and number_of_rooms > 0:
            try:
                recommendations = top_hotels(city, budget, number_of_rooms,df)
                if recommendations.empty:
                    st.write(f"Sorry, no hotels match your criteria in {city}.")
                else:
                    st.write(f"Here are the top hotel recommendations for you in {city}:")
                    display_columns = ['city', 'country', 'hotel', 'number_of_rooms', 'budget', 'website']
                    top_recommendations = recommendations[display_columns].head(5)
                    html = top_recommendations.to_html(escape=False, index=False, formatters={
                        'website': lambda x: f'<a href="{x}" target="_blank">Book Here</a>'
                    })
                    st.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                st.write(f"An error occurred: {str(e)}")
        else:
            st.write("Please enter all the required fields correctly.")
    
    if st.button("Clear Search"):
        st.session_state.recommendations = pd.DataFrame()
        st.session_state.hotel_chat = False
        st.rerun()

if __name__ == "__main__":
    main()
