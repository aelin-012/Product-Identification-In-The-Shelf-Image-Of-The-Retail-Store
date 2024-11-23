import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
import matplotlib.pyplot as plt
import cv2
import os
import json

# Define paths
train_data_dir = 'grocery-store-dataset/train'
val_data_dir = 'grocery-store-dataset/val'
test_data_dir = 'grocery-store-dataset/test'
model_path = 'mobilenet_v2_model.keras'
log_dir = 'logs/mobilenet_v2'  # Updated log directory for MobileNet

# Data augmentation for training data with preprocessing function
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only preprocessing for validation and test data
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load the train, validation, and test datasets
train_set = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_set = val_test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_set = val_test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load a pre-trained model (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model

# Add custom layers on top of the base model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout for regularization
    tf.keras.layers.Dense(len(train_set.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and model checkpoint
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('mobilenet_v2_model.keras', save_best_only=True, monitor='val_loss'),  # Updated extension
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # TensorBoard callback
]

# Train the model with validation data
history = model.fit(
    train_set,
    epochs=10,
    validation_data=val_set,  # Include validation data here
    callbacks=callbacks
)

# Save the trained model
model.save(model_path)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_set)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Function to preprocess and predict a single image
def predict_image(image_path, model):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = preprocess_input(input_arr)  # Apply preprocessing function here
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return predictions

# Visualize and predict test images
for image_file in os.listdir(test_data_dir):
    if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
        image_path = os.path.join(test_data_dir, image_file)

        # Read and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Predict the image
        predictions = predict_image(image_path, model)
        result_index = np.argmax(predictions)
        predicted_class = list(test_set.class_indices.keys())[result_index]  # Map index to class name

        # Display the image
        plt.imshow(img)
        plt.title(f'Test Image: Predicted Class - {predicted_class}')
        plt.xticks([])
        plt.yticks([])
        plt.show()

        # Print prediction
        print(f'Image: {image_file}, Predicted Class: {predicted_class}, Prediction Probabilities: {predictions}')

        # Store prediction summary
        predictions_summary = []
        predictions_summary.append((image_file, predicted_class, predictions))

# Save the results to a file
with open('D:/#1 OneDrive/SEM 7 Private/19CSE437/#1 Project/dlcv-project/mobilenet_v2_test_results.txt', 'w') as f:
    f.write(f'Test Accuracy: {test_acc:.4f}\n')
    f.write(f'Test Loss: {test_loss:.4f}\n')
    for image_file in os.listdir(test_data_dir):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(test_data_dir, image_file)
            predictions = predict_image(image_path, model)
            result_index = np.argmax(predictions)
            predicted_class = list(test_set.class_indices.keys())[result_index]
            f.write(f'Image: {image_file}, Predicted Class: {predicted_class}, Prediction Probabilities: {predictions}\n')

print("Testing complete.")

# Save class indices
with open('dataset-details.json', 'w') as f:
    json.dump(train_set.class_indices, f)