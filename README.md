# utils/viz_embed.py: 
Using ViT to encode training images in datasets
```bash
python3 viz_embed.py --output_dir <outputdir>
```
# preprocess.py:
Generate the json file containing a result of object detection and depth info for each object.
```bash
python3 preprocess.py
```

# rag_usage.py:
Create a vector database included vit-embedding and object_onehot_vec.
(modify the config to meet your demand first)
config={
    "dataset_name": "ntudlcv/dlcv_2024_final1", #huggingface dataset name
    "embedding_model_type": "dino", #dino or default 
    "FAISS_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/dino_vector_database", #needs full path, arbirary_name.faiss will do
    "JSON_PATH": "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs/train_metadata.json",
    "test": False, #set to True to test on small subset of training dataset (28.8k or 200)
    "init": True # init a new database, or just load a old one
}
Set the config:
    - Specify the correct faiss_path to ensure the path you store or load your database.
    - Make sure the preprocess of all the training dataset is run, and the result is store in JSON_PATH.
    - Set the embedding_model_type, dino or default(vitpatch32)
    - "test" is always False, unless you need to debug the whole rag system.
    - "init", when it is True FAISS_PATH(dir) will be removed!!!, and the vector database will be regenerated. Otherwise, just load the vectordatabase in FAISS_PATH and generate the ./storage/{embedding_model_type}_rag_test.json 
    
```bash
python3 rag_usage.py
```

#Inference
1. install checkpoints dir (ask murphy)
2. generate rag_results, a json file that tell which examples are chosen by vit embedding. (use rag_usage.py to generate it, or use the one that store at ./storage/*_rag_test.json)
3.generate convdata (use utils/get_convdata.py, or use the one store at ./storage/conversations.json )
4. run run run
```bash
python3 inference.py --output_path --checkpoint_dir --batch_size --device --rag_results --convdata
```

