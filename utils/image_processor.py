import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

def extract_features(image_data, model):
    """Extract features from image using VGG16 model."""
    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img = img.resize((224, 224))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data)
        return features.flatten()
    except Exception as e:
        raise Exception(f"Error processing image: {e}")

def find_top_matches_city(image_features, image_df, top_n=3):
    """Find top matching cities based on feature similarity."""
    similarities = []
    for _, row in image_df.iterrows():
        sim = cosine_similarity([image_features], [row['features']])[0][0]
        similarities.append(sim)
    
    # Create similarity series
    similarity_series = pd.Series(similarities, index=image_df['city'])
    return similarity_series.sort_values(ascending=False).head(top_n)
