FAISS_PATH = "data/"
JSON_INPUT_PATH = "data/input.json"
VIT_EMBEDDING_DIM = 768  # Example for ViT-base

class Config:
    def __init__(self, config:dict):
        self.config = config