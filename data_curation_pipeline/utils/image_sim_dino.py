import torch
import torchvision.transforms as T
from torchvision.io import read_image
from timm import create_model
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Load DINO model
def load_dino_model(device):
    model = create_model('vit_base_patch16_224_dino', pretrained=True)
    model.eval().to(device)
    return model

# Preprocess image to fit DINO input requirements
def preprocess_image(image_path, device):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = read_image(image_path)
    return transform(image).unsqueeze(0).to(device)

# Extract features from images
def extract_features(model, image_paths, device):
    features = []
    with torch.no_grad():
        for img_path in image_paths:
            img_tensor = preprocess_image(img_path, device)
            feature = model(img_tensor).squeeze(0)
            features.append(feature.cpu().numpy())
    return np.array(features)

# Compute cosine similarity matrix
def compute_similarity(features):
    similarity_matrix = cosine_similarity(features)
    return similarity_matrix

# Visualize similarity matrix with images and save to file
def visualize_similarity(similarity_matrix, image_paths, save_path="similarity_matrix.png"):
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, num_images, figsize=(15, 15))

    for i in range(num_images):
        for j in range(num_images):
            if i == j:
                img = read_image(image_paths[i]).permute(1, 2, 0).numpy() / 255.0
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
            else:
                axes[i, j].text(0.5, 0.5, f'{similarity_matrix[i, j]:.2f}',
                                horizontalalignment='center', verticalalignment='center', fontsize=12)
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Similarity matrix saved to {save_path}")

def get_image_name(image_dir):
    image_names = set()

    for path in os.listdir(image_dir):
        image_name = path.split("_", 2) # sa_223755_40.jpg -> [sa,223755,40.....]
        image_name = '_'.join([image_name[0], image_name[1]])
        image_names.add(image_name)

    return list(image_names)

# Main function
def main(image_dir, visualize_save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_names = get_image_name(image_dir)

    model = load_dino_model(device)

    start_time = time.time()

    for image_name in sorted(image_names)[:100]:
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if (img.endswith(('png', 'jpg', 'jpeg')) and "cropped" in img and image_name in img)]

        features = extract_features(model, image_paths, device)
        similarity_matrix = compute_similarity(features)

        # Visualize similarity and save the output
        os.makedirs(visualize_save_dir, exist_ok=True)
        save_path = os.path.join(visualize_save_dir, image_name + ".png")

        # visualize_similarity(similarity_matrix, image_paths, save_path)

    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    image_directory = "/path/to/image/"
    visualize_save_dir = "/path/to/save/"
    main(image_directory, visualize_save_dir)