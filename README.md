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
```bash
python3 rag_usage.py
```

