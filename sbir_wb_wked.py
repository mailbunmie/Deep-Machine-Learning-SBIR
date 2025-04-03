!pip install pyngrok

!pip install flask ngrok
!pip install flask-ngrok
!pip install flask flask-ngrok
!pip install DeepImageSearch

import os
import torch
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from torchvision import transforms
from torchvision.models import resnet50
from timm import create_model
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from google.colab import drive
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Mount Google Drive (only works in Colab)
drive.mount('/content/drive', force_remount=True)

## Start ngrok tunnel for Flask app
#from pyngrok import ngrok
#ngrok.set_auth_token("xxxxxxxx")  # Replace with your ngrok auth token
#public_url = ngrok.connect(5000)
#print(f"Public URL: {public_url}")

import os
import torch
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from torchvision import transforms
from torchvision.models import resnet50
from timm import create_model
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from google.colab import drive
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Mount Google Drive (only works in Colab)
drive.mount('/content/drive', force_remount=True)

# Start ngrok tunnel for Flask app
from pyngrok import ngrok
ngrok.set_auth_token("xxxxxxxx")  # Replace with your ngrok auth token
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Function to load image paths from a folder
def load_images_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image extensions
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

# Load images from folder
image_folder_path = '/content/drive/Othercomputers/My Laptop/All_Apps/imgs/used/objects/testBSame'
image_list = load_images_from_folder(image_folder_path)

# Helper function to calculate dynamic padding
def calculate_padding(size):
    width, height = size
    if width > height:
        delta = width - height
        return (0, delta // 2, 0, delta - delta // 2)
    elif height > width:
        delta = height - width
        return (delta // 2, 0, delta - delta // 2, 0)
    else:
        return (0, 0, 0, 0)  # No padding needed for square images

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.pad(
        img,
        padding=calculate_padding(img.size),
        fill=0,  # Border color, 0 for black
        padding_mode='constant'
    )),
    transforms.Resize((224, 224)),  # Resize to 224x224 after padding
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load ViT and ResNet models
model_vit = create_model('vit_base_patch16_224', pretrained=True)
model_vit.eval()

model_resnet = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
model_resnet.fc = torch.nn.Identity()  # Remove the fully connected layer
model_resnet.eval()

projection_layer = torch.nn.Conv2d(in_channels=768, out_channels=3, kernel_size=1)
projection_layer.eval()

# Function to extract features using ViT up to the transformer encoder level
def extract_vit_features(image_list):
    feature_list = []
    for img_path in image_list:
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB mode
            img_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                features = model_vit.forward_features(img_tensor)
            feature_list.append(features)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return feature_list

# Function to classify features using ResNet-50
def classify_features(vit_features):
    feature_list = []
    with torch.no_grad():
        for features in vit_features:
            # Reshape features to match ResNet-50 input requirements
            features_tensor = features.permute(0, 2, 1).unsqueeze(3)  # (batch_size, 768, 197, 1)
            features_tensor = projection_layer(features_tensor)  # Project to 3 channels
            class_features = model_resnet(features_tensor)
            feature_list.append(class_features.squeeze().numpy())
    return feature_list

# Extract features using ViT
vit_features = extract_vit_features(image_list)

# Classify features using ResNet-50
resnet_features = classify_features(vit_features)

# Convert ResNet features to numpy array
resnet_features = np.array(resnet_features)

# Indexing with ResNet features
knn_index = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn_index.fit(resnet_features)

# Function to get similar images
def get_similar_images(image_path, top_k=10):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        # Extract features using ViT
        with torch.no_grad():
            vit_feat = model_vit.forward_features(img_tensor)

        # Reshape and project to match ResNet input
        vit_feat = vit_feat.permute(0, 2, 1).unsqueeze(3).float()  # (batch_size, 768, 197, 1)
        vit_feat = projection_layer(vit_feat)  # Project to 3 channels

        # Classify features using ResNet-50
        with torch.no_grad():
            resnet_feat = model_resnet(vit_feat)

        # Query the index
        distances, indices = knn_index.kneighbors(resnet_feat.numpy(), n_neighbors=top_k)

        similar_images = [image_list[i] for i in indices[0]]
        return similar_images
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

# Function to calculate performance metrics
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1, accuracy

# Route for handling image upload
@app.route('/query', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Save the uploaded file
            filename = os.path.join("uploads", file.filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file.save(filename)

            # Process the file to find similar images
            similar_images = get_similar_images(filename)

            # Calculate and return performance metrics
            true_labels = [0, 1, 1, 0, 1]  # Replace with actual labels
            predicted_labels = [0, 1, 0, 0, 1]  # Replace with actual predictions
            precision, recall, f1, accuracy = calculate_metrics(true_labels, predicted_labels)

            return jsonify({
                'similar_images': similar_images,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy
            })
    except Exception as e:
        print(f"Error in upload route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app with ngrok tunneling
    run_with_ngrok(app)  # Automatically triggers ngrok to open a tunnel
    app.run()

