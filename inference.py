import torch
import tqdm
import json
from datasets import load_dataset
from dlcv_datasets import ImageDataset
from torch.utils.data import DataLoader
from utils import ImageEmbedder, preprocess_batch_images, create_object_presence_vector
from rag_package import RAG
from PIL import Image
from utils import encode_single_image, create_object_presence_vector, preprocess_single_image



def main():
    # Set RAG 
    config={
    "dataset_name": "ntudlcv/dlcv_2024_final1", #huggingface dataset name
    "model_name": "google/vit-base-patch32-224-in21k", #ViT name
    "FAISS_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/vector_database", #needs full path, arbirary_name.faiss will do
    "METADATA_PATH": "vector_database/metadata.db", #arbitrary db name is suffice,
    "JSON_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs/train_metadata.json",
    "testing": True #set to True to test on small subset
    }

    myRag = RAG(config)

    # Load test dataset    
    test_dataset = load_dataset("your_dataset_name", split="test")
    test_dataset = ImageDataset(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # Adjust batch size as needed
        shuffle=False,  # Keep False for test data
        num_workers=4   # Adjust based on your CPU cores
    )
    image_embedder = ImageEmbedder(model_name="vit model name")
    results_dict = {}
    
    for batch_images, batch_ids in tqdm(test_loader, desc = "Processing Image"):
        batch_embeddings = image_embedder.encode_images_with_vit(batch_images)
        batch_results = preprocess_batch_images(batch_images, batch_ids)
        batch_presence_vectors = create_object_presence_vector(batch_results)

        # Save embeddings to a file or use them further
        # ...
        for i, embedding, obj_presence in enumerate(zip(batch_embeddings, batch_presence_vectors)):
            _, _, vit_image_ids = myRag.vit_emb_query.search_by_vector(embedding, k=5)
            _, _, obj_image_ids = myRag.obj_pre_query.search_by_vector(obj_presence, k=5)

            results_dict[batch_ids[i]] = {
                'vit_similar_images': vit_image_ids,
                'obj_similar_images': obj_image_ids
            }

    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)        
    
    return results_dict
        
        




