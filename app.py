import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import gdown
import os

# Load model from Google Drive if not present
@st.cache_resource
def load_model():
    model_path = "model.keras"
    if not os.path.exists(model_path):
        # Replace with your actual file ID from Google Drive
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# Prediction function
def predict_xray(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_batch)[0][0]
    label = 'PNEUMONIA' if pred >= 0.5 else 'NORMAL'
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence, img

# UI
st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered", page_icon="🫁")
st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**.")

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    label, confidence, img = predict_xray(uploaded_file)
    st.image(img, caption=f"Prediction: {label} ({confidence*100:.2f}%)", use_column_width=True)

