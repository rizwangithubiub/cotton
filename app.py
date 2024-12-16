import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Correct model path
MODEL_PATH = 'model_resnet152V2.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Assuming model expects 224x224 input
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def predict(image):
    preprocessed_image = preprocess_image(image)
    preds = model.predict(preprocessed_image)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The leaf is diseased cotton leaf"
    elif preds == 1:
        preds = "The leaf is diseased cotton plant"
    elif preds == 2:
        preds = "The leaf is fresh cotton leaf"
    else:
        preds = "The leaf is fresh cotton plant"
    return preds

# Streamlit app
st.title("Cotton Disease Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    prediction = predict(image)
    
    st.write(f"Prediction: {prediction}")

