import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
import cv2

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Load image and resize
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    return img_array

# Function to compute Grad-CAM
def compute_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])  # Use the predicted class index
        output_class = predictions[:, class_idx]

    grads = tape.gradient(output_class, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    conv_outputs *= pooled_grads

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Normalize the heatmap
    return heatmap

# Function to overlay the heatmap on the original image with a red highlight
def overlay_red_heatmap(img_path, heatmap):
    img = image.load_img(img_path)
    img = image.img_to_array(img)  # Convert to array
    img = np.uint8(img)  # Ensure original image is of type uint8

    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Convert heatmap to 8-bit

    # Create a red heatmap
    red_heatmap = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    red_heatmap[..., 0] = heatmap  # Red channel
    red_heatmap[..., 1] = 0        # Green channel
    red_heatmap[..., 2] = 0        # Blue channel

    # Blend the original image with the red heatmap using a higher weight for visibility
    superimposed_img = cv2.addWeighted(red_heatmap, 0.7, img, 0.3, 0)  # Adjust weights as needed
    return superimposed_img

# Load and preprocess the image
img_path = "grocery-store-dataset/sample-image/Oatly-Natural-Oatghurt.jpg"  # Replace with your image path
img_array = preprocess_image(img_path)

# Make predictions
preds = model.predict(img_array)

# Decode predictions based on output shape
decoded_preds = decode_predictions(preds, top=1)[0]
print("Predicted class:", decoded_preds)

# Get the last convolutional layer name (ensure this is correct for your model)
last_conv_layer_name = "Conv_1"  

# Compute Grad-CAM heatmap
heatmap = compute_gradcam(model, img_array, last_conv_layer_name)

# Overlay the red heatmap on the image
superimposed_img = overlay_red_heatmap(img_path, heatmap)

# Display the Grad-CAM result with red highlight
plt.imshow(superimposed_img[..., ::-1])  # Convert from BGR to RGB for display
plt.axis('off')
plt.show()