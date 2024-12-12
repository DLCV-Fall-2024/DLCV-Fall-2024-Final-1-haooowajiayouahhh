import torch
from tqdm import tqdm
import json
from datasets import load_dataset
from dlcv_datasets import ImageDataset
from torch.utils.data import DataLoader
from rag_package import RAG
from PIL import Image
from utils import encode_single_image, create_single_object_presence_vector, preprocess_single_image



def rag_usage():
    # Set RAG 
    config={
    "dataset_name": "ntudlcv/dlcv_2024_final1", #huggingface dataset name
    "model_name": "google/vit-base-patch32-224-in21k", #ViT name
    "FAISS_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/vector_database", #needs full path, arbirary_name.faiss will do
    "METADATA_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/vector_database/metadata.db", #arbitrary db name is suffice,
    "JSON_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs/train_metadata.json",
    "testing": True #set to True to test on small subset
    }

    myRag = RAG(config)
    print("rag set")

    # Load test dataset    
    dataset = load_dataset(config["dataset_name"], split="test")
    dataset = dataset.take(10)
    print("Processing images...")
    results_dict = {}
    
    for item in tqdm(dataset, 
                    desc="Processing",
                    bar_format='{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'):
        image_id = item['id']
        task = image_id.split('_')[1]
        image = item['image']
        embedding = encode_single_image(image)
        formatted_object = preprocess_single_image(image, task)
        presence_vector = create_single_object_presence_vector(formatted_object)
        _, _, vit_image_ids = myRag.vit_emb_query.search_by_vector(embedding, k=5)
        _, _, obj_image_ids = myRag.obj_pre_query.search_by_vector(presence_vector, k=5)

        results_dict[image_id] = {
            'vit_similar_images': vit_image_ids,
            'obj_similar_images': obj_image_ids
        }

    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)        
    
    return results_dict

rag_usage()

 
        




