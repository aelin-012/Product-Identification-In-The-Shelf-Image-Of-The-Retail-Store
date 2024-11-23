import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

# Load your Keras models
model_paths = {
    "mobilenet_v2": "models/mobilenet_v2_model.keras",
    "resnet50": "models/resnet50_model.keras",
    "densenet169": "models/densenet169_model.keras",
}

# Load models into a dictionary for easy access
models_dict = {name: load_model(path) for name, path in model_paths.items()}

def preprocess_image(img_data):
    # Convert bytes data to a suitable format for Keras
    img = image.load_img(BytesIO(img_data), target_size=(224, 224))  # Adjust target size to your model's requirements
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_image(model_name: str, img_data: bytes):
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found.")

    model = models_dict[model_name]
    img_array = preprocess_image(img_data)

    # Predict using the Keras model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get class with highest probability
    confidence = np.max(predictions[0])  # Get the maximum confidence score

    # You can map predicted_class to actual class names if necessary
    return predicted_class, confidence
