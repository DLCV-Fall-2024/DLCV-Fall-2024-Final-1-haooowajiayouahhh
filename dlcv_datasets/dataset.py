from transformers import ViTImageProcessor
from PIL import Image
from torch.utils.data import Dataset

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