from transformers import ViTModel, ViTFeatureExtractor
import os
import torch
from transformers import ViTImageProcessor, ViTModel
import numpy as np 
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from dlcv_datasets import ImageDataset
from datasets import load_dataset
from dlcv_datasets import create_small_subset

class ImageEmbedder:
    def __init__(self, dataset_name="ntudlcv/dlcv_2024_final1", model_name="google/vit-base-patch16-224", test_mode=False):
        if test_mode:
            dataset = create_small_subset(dataset_name=dataset_name, split="train", num_samples=200)
        else:
            dataset = load_dataset(dataset_name, split="train")
        self.model = ViTModel.from_pretrained(model_name)
        self.dataset = ImageDataset(dataset)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    def encode_images_with_vit(self, output_dir='./vit-images'):
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
                
                # # Save embeddings
                # for embedding, idx in zip(embeddings, batch_ids):
                #     save_path = os.path.join(output_dir, f'{idx}.npy')

                #     np.save(save_path, embedding)
        # print(f"all_embeddings_shape: {np.array(all_embeddings).shape}")
        # print(f"all_ids_shape: {np.array(all_ids).shape}")
        return all_embeddings, all_ids


