from transformers import CLIPProcessor,AutoImageProcessor
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, dataset, model_type='default'):
        self.dataset = dataset
        if model_type=='default':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        elif model_type=='dino':
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            print("[ImageDataset] loaded processor for dino... ")

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

  