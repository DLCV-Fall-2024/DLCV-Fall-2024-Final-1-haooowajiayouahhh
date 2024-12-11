FAISS_INDEX_PATH = "data/image_index.faiss"
JSON_INPUT_PATH = "data/input.json"
EMBEDDING_DIM = 768  # Example for ViT-base

class Config:
    def __init__(self, config:dict):
        self.config = config