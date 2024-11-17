import numpy as np
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
    """Find top matching cities based on image features."""
    image_df['similarity'] = image_df['features'].apply(
        lambda x: cosine_similarity([image_features], [x]).flatten()[0]
    )
    top_matches = image_df.loc[image_df.groupby('city')['similarity'].idxmax()]
    top_matches = top_matches.sort_values(by='similarity', ascending=False).head(top_n)
    return top_matches.reset_index(drop=True)