from rag_package import RAG


config={
    "dataset_name": "ntudlcv/dlcv_2024_final1", #huggingface dataset name
    "model_name": "google/vit-base-patch32-224-in21k", #ViT name
    "FAISS_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/", #needs full path, arbirary_name.faiss will do
    "METADATA_PATH": "metadata.db", #arbitrary db name is suffice,
    "EMBEDDING_DIM": 768, #ViT embedding dimension
    "testing": True #set to True to test on small subset
}

myRag = RAG(config)

# Query an image
image_id = "Test_general_0"
print(myRag.query_image(image_id, k=5))
