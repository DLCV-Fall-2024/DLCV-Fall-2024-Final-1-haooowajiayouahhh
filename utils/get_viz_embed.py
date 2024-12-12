from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor
import torch
from PIL import Image
import numpy as np 
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import argparse


class ImageEmbedder:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)

    def encode_images_with_vit(self, batch_images):

        with torch.no_grad():
            batch_images = batch_images.to(self.device)
            outputs = self.model(batch_images)
            embeddings = outputs.last_hidden_state[:, 0]

        return embeddings


def encode_single_image(image):
    """
    Encode a single image using ViT model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: The embedding vector for the image
    """
    # Load and initialize the model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load and process the image
    processed = processor(images=image, return_tensors="pt")
    pixel_values = processed['pixel_values'].to(device)
    
    # Get embedding
    with torch.no_grad():
        outputs = model(pixel_values)
        embedding = outputs.last_hidden_state[:, 0].cpu().numpy().squeeze()
        
    return embedding

# Example usage:
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Extract ViT embeddings from images')
    parser.add_argument('--output_dir', type=str, default='./vit-images',
                        help='Directory to save the embeddings')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to process')
    
    args = parser.parse_args()
    
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=args.split)
    
    print(f"Dataset size: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}")
    
    encode_images_with_vit(dataset, output_dir=args.output_dir)