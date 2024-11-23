import os
import json

# Define the dataset directory structure
dataset_dir = 'grocery-store-dataset'
output_json = 'dataset-details.json'

# Function to generate label JSON
def generate_label_json(dataset_dir, output_file):
    labels = {}
    categories = ['test', 'train', 'val']

    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        if os.path.exists(category_path):
            labels[category] = {item: idx for idx, item in enumerate(os.listdir(category_path))}
    
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=4)

# Generate the JSON file
generate_label_json(dataset_dir, output_json)
