import os
import numpy as np
from rag_package import ImageEmbedder, FAISSDatabase
import rag_package.config as cfg
from rag_package import RAG


def test_pipeline():
    config={
        "dataset_name": "ntudlcv/dlcv_2024_final1",
        "model_name": "google/vit-base-patch32-224-in21k",
        "FAISS_INDEX_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/test_index.faiss",
        "METADATA_PATH": "metadata.db",
        "EMBEDDING_DIM": 768,
        "testing": True
    }

    myRag = RAG(config)

