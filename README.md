# Directory Tree
```bash
.
|-- README.md
|-- README_OFFICAL.md
|-- few_shot
|   |-- scene_few_shot
|   |   |-- high.json
|   |   `-- low.json
|   `-- suggestion_few_shot
|       |-- high.json
|       `-- low.json
|-- gemini_eval.py
|-- grounding_dino.py
|-- hugging_face_depth_anything.py
|-- images
|   |-- dataset.png
|   `-- submission.png
|-- llama_eval.py
|-- preprocess.py
|-- requirement.txt
|-- scorer.py
`-- utils
    `-- viz_embed.py
```

# utils/viz_embed.py: 
Using ViT to encode training images in datasets
```bash
python3 viz_embed.py --output_dir <outputdir>
```