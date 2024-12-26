# Inference

## Setup Environment
```bash
conda create -n myenv python=3.11
cd LLaVA/
pip install -r llava_requirements.txt
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install transformers==4.46.0
pip install accelerate==0.26.0
pip install protobuf==3.20.*
pip install --upgrade Pillow
pip install gdown
```
## Download ckpt and run
```bash
cd ..
bash download_ckpt.sh
bash inference.sh
```

# finetune model
1. cd LLaVA
2. ensure you have convdata train_metadata and rag_results (already pushed to github...)
3. To finetune LoRAs on different tasks, choose and run the corresponding training script 

```bash
bash scripts/v1_5/finetune_{task_name}_lora.sh $1 $2 $3 $4
#$1: lora_rank $2:lora_alpha $3: metadata_file $4: rag_file 
# task_name: general, regional, suggestion 
```

4. you can modify the --output_dir in the finetune script to specify the output checkpoint dir


#haotian2hf
1. cd LLaVA/llava
2. run 
```bash
python3 convert_haotian2hf.py --old_ckpt_dir "path to the ckptdir of haotianllava" --save_path "a path to save the result"
```
3. or just import function convert_llava_llama_to_hf(), it will return a prepared llava model and processor
4. Caution!!! the model need to transform to torch.float16 when you use it.
for example,
```bash
condition 1:
model, processor=convert_llava_llama_to_hf()
model.to("cuda", dtype=torch.float16)

condition 2:
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
state_dict = torch.load(state_dict_path, map_location="cpu")
model.load_state_dict(state_dict, strict=True, assign=True)
model.to('cuda', dtype=torch.float16)
```

# Preprocess

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