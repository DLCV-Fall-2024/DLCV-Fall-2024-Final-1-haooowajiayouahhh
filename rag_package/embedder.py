from transformers import ViTModel, ViTFeatureExtractor
import os
import torch
import numpy as np 
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from dlcv_datasets import ImageDataset
from datasets import load_dataset
from dlcv_datasets import create_small_subset

class ImageEmbedder:
    def __init__(self, dataset_name="ntudlcv/dlcv_2024_final1", test_mode=False, model_type='default', task='general'):
        if test_mode:
            dataset = create_small_subset(dataset_name=dataset_name, split="train", num_samples=200)
        else:
            if task == 'general':
                dataset = load_dataset(dataset_name, split="train[:4884]")
            if task == 'regional':
                dataset = load_dataset(dataset_name, split="train[4884:21926]")
            if task ==  'suggestion':
                dataset = load_dataset(dataset_name, split="train[21926:]")

        self.dataset = ImageDataset(dataset, model_type=model_type)
        self.task = task

    def encode_images_with_vit(self, output_dir='./vit-images'):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(device)
        model.eval()
        print("model loaded")
        
        # Create dataset and dataloader
        image_dataset = self.dataset
        dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
        

        all_embeddings = []
        all_ids = []

        with torch.no_grad():
            for batch_images, batch_ids in tqdm(dataloader, desc="Processing images"):
                # print(f"batch_ids: {batch_ids}")
                batch_images = batch_images.to(device)
                outputs = model(batch_images)
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                all_embeddings.extend(embeddings)
                all_ids.extend(batch_ids)
                
        return all_embeddings, all_ids
    
    def encode_images_with_dino(self, output_dir='./dino-images'):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        from transformers import AutoModel
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(device)
        model.eval()
        print("Dinov2 model loaded")
        
        # Create dataset and dataloader
        image_dataset = self.dataset
        dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
        

        all_embeddings = []
        all_ids = []

        with torch.no_grad():
            for batch_images, batch_ids in tqdm(dataloader, desc="Processing images"):
                # print(f"batch_ids: {batch_ids}")
                batch_images = batch_images.to(device)
                # batch_images = processor(batch_images, return_tensors="pt")
                outputs = model(batch_images)
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                all_embeddings.extend(embeddings)
                all_ids.extend(batch_ids)
        return all_embeddings, all_ids
    
