from transformers import ViTImageProcessor
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, dataset, model_name):
        self.dataset = dataset
        self.processor = ViTImageProcessor.from_pretrained(model_name)
    
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

