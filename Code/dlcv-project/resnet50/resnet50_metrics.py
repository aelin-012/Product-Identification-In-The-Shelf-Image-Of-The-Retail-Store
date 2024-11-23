import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
model = tf.keras.models.load_model('resnet50_model.keras')

# Assuming you have already loaded train, validation, and test sets
train_data_dir = 'grocery-store-dataset/train'
val_data_dir = 'grocery-store-dataset/val'
test_data_dir = 'grocery-store-dataset/test'

# Data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_set = train_datagen.flow_from_directory(
    train_data_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical', 
    shuffle=True
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

# Function to evaluate model performance on train, validation, and test datasets
def evaluate_model_performance(model, dataset):
    loss, accuracy = model.evaluate(dataset)
    return loss, accuracy

# Generate predictions and calculate confusion matrix
def get_predictions_and_labels(model, dataset):
    y_pred = model.predict(dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = dataset.classes
    return y_pred_classes, y_true

# Calculate classification metrics
def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, cm

# Calculate sensitivity and specificity
def calculate_sensitivity_specificity(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)
    
    sensitivity = TP / (TP + FN + 1e-7)  # Add small value to avoid division by zero
    specificity = TN / (TN + FP + 1e-7)
    
    return np.mean(sensitivity), np.mean(specificity)

# Plot confusion matrix and save as PNG
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_resnet50.png'):
    plt.figure(figsize=(12, 10), dpi=150)  # Increased figure size and DPI for clarity
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, 
                cbar=True, square=True, linewidths=0.5, linecolor='black', annot_kws={"size": 12})
    
    plt.title('Confusion Matrix - ResNet50', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    plt.tight_layout()  # Ensures the plot fits well within the figure window
    plt.savefig(save_path, bbox_inches='tight')  # Centers the image and removes extra white space
    plt.close()

# Plot performance metrics as horizontal bar plot
def plot_performance_metrics(metrics):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Sensitivity', 'Specificity', 'Loss']
    metrics_values = [
        metrics['Accuracy'],
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1 Score'],
        metrics['Sensitivity'],
        metrics['Specificity'],
        metrics['Loss']
    ]
    
    plt.figure(figsize=(10, 6))
    plt.barh(metrics_names, metrics_values, color='skyblue')
    plt.xlabel('Value')
    plt.title('Performance Metrics - ResNet50')
    for index, value in enumerate(metrics_values):
        plt.text(value, index, f'{value:.4f}', va='center')
    plt.savefig('performance_metrics_resnet50.png')
    plt.close()

# Plot training and validation accuracy and save as PNG
def plot_accuracy(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy - ResNet50')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('model_accuracy_resnet50.png')
    plt.close()

# Plot training and validation loss and save as PNG
def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss - ResNet50')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('model_loss_resnet50.png')
    plt.close()

# Train the model
history = model.fit(
    train_set,
    epochs=10,
    validation_data=val_set,
    callbacks=[]  # Add any callbacks you want
)

# Get metrics for validation dataset
print("Evaluating dataset...")

# Loss and Accuracy
loss, accuracy = evaluate_model_performance(model, test_set)

# Predictions and Labels
y_pred, y_true = get_predictions_and_labels(model, test_set)

# Classification Metrics
acc, precision, recall, f1, cm = calculate_classification_metrics(y_true, y_pred)

# Calculate sensitivity and specificity
sensitivity, specificity = calculate_sensitivity_specificity(cm)

# Store metrics in a dictionary
results = {
    'Loss': loss,
    'Accuracy': acc,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'Sensitivity': sensitivity,
    'Specificity': specificity
}

# Class names from dataset
class_names = list(test_set.class_indices.keys())

# Save confusion matrix and performance metrics
plot_confusion_matrix(cm, class_names)
plot_performance_metrics(results)

# Plot accuracy and loss
plot_accuracy(history.history)
plot_loss(history.history)

print("All metrics, loss and accuracy plots, and confusion matrix for ResNet50 are saved.")
