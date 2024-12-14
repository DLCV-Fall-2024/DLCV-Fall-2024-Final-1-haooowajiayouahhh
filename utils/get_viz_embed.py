from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np 
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import argparse



def encode_single_image(image,model_type='default'):
    """
    Encode a single image using ViT model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: The embedding vector for the image
    """
    # Load and initialize the model and processor
    if model_type=='default':
        print("encoding single image using default ViT...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
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
    elif model_type=='dino':
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        processor= AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        processed = processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].to(device)
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