import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader

class CLIPClassifier:
    def __init__(self, model_name="ViT-L/14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.text_features = None
        self.labels = None

    def set_labels(self, labels):
        self.labels = labels
        text_tokens = clip.tokenize(labels).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def classify(self, pil_image):
        with torch.no_grad():
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ self.text_features.T).softmax(dim=-1)
        best_label_idx = similarities.argmax().item()
        return self.labels[best_label_idx]
