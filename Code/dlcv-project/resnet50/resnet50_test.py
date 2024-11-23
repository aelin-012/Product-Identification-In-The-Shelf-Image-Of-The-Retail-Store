import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from PIL import Image
import json
import time

# Load the trained model
model_path = 'resnet50_model.keras'
model = tf.keras.models.load_model(model_path)

# Load class indices
with open('dataset-details.json', 'r') as f:
    class_indices = json.load(f)

def predict_image(image, model, class_indices):
    # Preprocess the image
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    # Preprocess for ResNet50
    input_arr = preprocess_input(input_arr)

    # Predict the image
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    predicted_class = list(class_indices.keys())[result_index]  # Map index to class name

    return predicted_class, predictions

# Streamlit App
st.set_page_config(page_title="Image Classification with ResNet50", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    body {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        text-align: center;
        color: #ffffff;
        margin-bottom: 20px;
    }
    .uploaded-image {
        border: 5px solid #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s;
        margin: auto;
        max-width: 300px;  /* Set max width for the image */
        max-height: 300px; /* Set max height for the image */
        display: block;     /* Center the image */
    }
    .uploaded-image:hover {
        transform: scale(1.05);
    }
    .probability-table {
        width: 100%;
        margin-top: 20px;
        border-spacing: 0;
        border-collapse: separate;
    }
    .probability-table th, .probability-table td {
        padding: 15px 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #ddd;
    }
    .probability-table th {
        background-color: #4CAF50;
        color: white;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
    }
    .probability-table td {
        background-color: rgba(0, 0, 0, 0.05);
        color: #333333;
    }
    .probability-table tr:last-child td {
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
    }
    .highlight {
        background-color: #ffcc00;
        font-weight: bold;
        color: #000000;
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    /* Celebration Animation */
    .celebration {
        position: relative;
        display: inline-block;
        padding: 20px;
        font-size: 30px;
        font-weight: bold;
        color: #ffcc00;
    }
    .celebration::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.5) 10%, transparent 20%);
        border-radius: 50%;
        opacity: 0;
        animation: celebrate 1.5s ease-out;
    }
    @keyframes celebrate {
        0% {
            transform: scale(0.5);
            opacity: 0;
        }
        50% {
            transform: scale(1.5);
            opacity: 1;
        }
        100% {
            transform: scale(1);
            opacity: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("Image Classification with ResNet50")
st.write("Upload an image to classify it using a pre-trained ResNet50 model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  # Resize for model input
    
    # Display the image with custom styling, set size for display
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)  # Center the image
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.write("Classifying...")

    # Show loading spinner
    with st.spinner("Processing..."):
        time.sleep(1)  # Simulate processing time
        # Predict the image
        predicted_class, predictions = predict_image(image, model, class_indices)
    
    # Display the results with celebration effect
    st.write(f'<div class="celebration">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)
    
    # Display the prediction probabilities in a table with rounded corners
    st.write('<table class="probability-table">', unsafe_allow_html=True)
    st.write('<thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>', unsafe_allow_html=True)
    for class_name, prob in zip(class_indices.keys(), predictions[0]):
        if class_name == predicted_class:
            st.write(f'<tr><td>{class_name}</td><td class="highlight">{prob:.4f}</td></tr>', unsafe_allow_html=True)
        else:
            st.write(f'<tr><td>{class_name}</td><td>{prob:.4f}</td></tr>', unsafe_allow_html=True)
    st.write('</tbody></table>', unsafe_allow_html=True)
    