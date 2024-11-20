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
    login("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")

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
# Initialize HuggingFace and models
    
setup_huggingface()
llm_stat = setup_llm()
embeddings_stat = setup_embeddings()
    
# Load vector database
vector_db = FAISS.load_local("data/faiss_index", embeddings_stat, allow_dangerous_deserialization=True)
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

def extract_city_nlp(user_input):
    doc = nlp(user_input)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE (Geopolitical Entity) is used for city names
            return ent.text
    return "City not found"

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

