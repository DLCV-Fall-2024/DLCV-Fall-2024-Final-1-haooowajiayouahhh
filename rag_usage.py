from rag_package import RAG
from PIL import Image
from utils import encode_single_image, create_object_presence_vector, preprocess_single_image
# Construct RAG first

config={
    "dataset_name": "ntudlcv/dlcv_2024_final1", #huggingface dataset name
    "model_name": "google/vit-base-patch32-224-in21k", #ViT name
    "FAISS_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/vector_database", #needs full path, arbirary_name.faiss will do
    "METADATA_PATH": "vector_database/metadata.db", #arbitrary db name is suffice,
    "JSON_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs/train_metadata.json",
    "testing": True #set to True to test on small subset
}

myRag = RAG(config)

# define the inference data

# print(myRag.vit_emb_query_image_by_vector(data.get_vit_emb(), k=5))
# print(myRag.obj_pre_query_image_by_vector(data.get_obj_pre_vec(), k=5))
