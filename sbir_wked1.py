import torch
from torchvision import transforms
from torchvision.models import resnet50
from timm import create_model
from DeepImageSearch import Load_Data
from sklearn.neighbors import NearestNeighbors
from google.colab import drive
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load images from a folder
image_list = Load_Data().from_folder(['/content/drive/Othercomputers/My Laptop/All_Apps/imgs/used/objects/testBSame'])

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

# Updated preprocessing pipeline
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

# Define global models and projection layer
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

# Example usage
image_path = '/content/drive/Othercomputers/My Laptop/All_Apps/imgs/used/1a.jpg'
similar_images = get_similar_images(image_path)

# Simulate true labels and predicted labels (for demonstration purposes)
true_labels = [0, 1, 1, 0, 1]  # Replace with actual labels
predicted_labels = [0, 1, 0, 0, 1]  # Replace with actual predictions

# Calculate and print metrics
precision, recall, f1, accuracy = calculate_metrics(true_labels, predicted_labels)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# Function to plot similar images
def plot_similar_images(image_path, similar_images):
    num_images = min(10, len(similar_images) + 1)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Output Objects')
    axes = axes.flatten()

    # Plot the query image
    query_img = Image.open(image_path)
    query_name = image_path.split('/')[-1]  # Extract the query image name
    axes[0].imshow(query_img)
    axes[0].set_title(f'Query Sketch: {query_name}')
    axes[0].axis('off')

    # Plot similar images
    for i, img_path in enumerate(similar_images[:num_images-1]):
        img = Image.open(img_path)
        original_name = img_path.split('/')[-1]  # Extract the original image name
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f'Object {i + 1}: {original_name}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

# Plot similar images
plot_similar_images(image_path, similar_images)