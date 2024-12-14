import torch
from tqdm import tqdm
import json
from datasets import load_dataset
from dlcv_datasets import ImageDataset
# from dlcv_datasets import ImageDataset, ConversationDataset
from torch.utils.data import DataLoader
from rag_package import RAG
from PIL import Image
from utils import encode_single_image, create_single_object_presence_vector, preprocess_single_image
import os


def rag_usage():
    # Set RAG 
    config={
    "dataset_name": "ntudlcv/dlcv_2024_final1", #huggingface dataset name
    "embedding_model_type": "default", #dino or default(clip-vitL14)
    "FAISS_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/dino_vector_database", #needs full path, arbirary_name.faiss will do
    "JSON_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs/train_metadata.json",
    "test": False, #set to True to test on small subset of training dataset (28.8k or 200)
    "init": True # init a new database, or just load a old one
    }
    if config["init"]:
        if os.path.exists(config["FAISS_PATH"]):
            os.system("rm -r " + config["FAISS_PATH"])
    myRag = {}
    myRag['general'] = RAG(config, 'general')
    myRag['regional'] = RAG(config, 'regional')
    myRag['suggestion'] = RAG(config, 'suggestion')
    print("rag set")

    # Load test dataset    
    dataset = load_dataset(config["dataset_name"], split="test")
    print("Processing images...")
    results_dict = {}
    
    for item in tqdm(dataset, 
                    desc="Processing",
                    bar_format='{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'):
        image_id = item['id']
        image = item['image']
        task = image_id.split('_')[1]
        embedding = encode_single_image(image,model_type=config['embedding_model_type'])
        formatted_object = preprocess_single_image(image, task)
        obj_image_ids = []
        vit_image_ids = []
        
        # Select the corresponding RAG instance based on the task
        if task == 'general':
            _, _, vit_image_ids = myRag['general'].vit_emb_query.search_by_vector(embedding, k=5)
            presence_vector = create_single_object_presence_vector(formatted_object)
            _, _, obj_image_ids = myRag['general'].obj_pre_query.search_by_vector(presence_vector, k=5)
        elif task == 'regional':
            _, _, vit_image_ids = myRag['regional'].vit_emb_query.search_by_vector(embedding, k=5)
            obj_image_ids = []  # No object query for regional task
        elif task == 'suggestion':
            _, _, vit_image_ids = myRag['suggestion'].vit_emb_query.search_by_vector(embedding, k=5)
            presence_vector = create_single_object_presence_vector(formatted_object)
            _, _, obj_image_ids = myRag['suggestion'].obj_pre_query.search_by_vector(presence_vector, k=5)

        results_dict[image_id] = {
            'vit_similar_images': vit_image_ids,
            'obj_similar_images': obj_image_ids
        }

    output_path = f"storage/{config['embedding_model_type']}_rag_test.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)        
    
    return results_dict



class PromptProcessor:
    def __init__(self, convdata, rag_results):
        self.convdata = convdata
        self.rag_results = rag_results
    
    def get_prompts(self, image_id, question_message, baseOn):
        prompt_task_description = question_message
        prompt_study_examples = "These examples are provided solely to help you understand. Here are the examples"
        prompt_examples = ""
        prompt_generate = "According to the image provided and the examples given, generate the response"
        example_image_ids = self.rag_results[image_id]
        for idx in example_image_ids[baseOn]:
            conv = self.convdata[idx]
            prompt_examples += f", {conv}"
        prompt_examples += "."
        final_prompt = prompt_task_description + prompt_study_examples + prompt_examples + prompt_generate

        return final_prompt
        
if __name__ == "__main__":
    rag_usage()






 
        




