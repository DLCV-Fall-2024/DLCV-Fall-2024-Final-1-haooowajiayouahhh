from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image
import numpy as np 
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import argparse

class ImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-224-in21k')
    
    def __getitem__(self, idx):
        # Load and process the image
        image = self.dataset[idx]['image']
        if isinstance(image, str):  # If image is a path
            image = Image.open(image).convert('RGB')
        # Process image and convert to tensor
        processed = self.processor(images=image, return_tensors="pt")
        return processed['pixel_values'].squeeze(0), self.dataset[idx]['id']
    
    def __len__(self):
        return len(self.dataset)

def encode_images_with_vit(dataset, output_dir='./vit-images'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(device)
    model.eval()
    print("model loaded")
    
    # Create dataset and dataloader
    image_dataset = ImageDataset(dataset)
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for batch_images, batch_ids in tqdm(dataloader, desc="Processing images"):
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            # Save embeddings
            for embedding, idx in zip(embeddings, batch_ids):
                save_path = os.path.join(output_dir, f'{idx}.npy')
                np.save(save_path, embedding)

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