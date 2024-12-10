import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from datasets import load_dataset
import os

def draw_boxes(image, results):
    # Create a copy of the image to draw on
    draw = ImageDraw.Draw(image)
    
    # Get colors for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Dark Red
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Blue
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 128),# Gray
        (192, 192, 192),# Light Gray
        (128, 128, 192),# Light Purple
        (128, 192, 128),# Light Green
        (192, 128, 128),# Light Red
        (192, 128, 192),# Light Magenta
        (128, 192, 192),# Light Cyan
        (192, 192, 128),# Light Olive
        (192, 192, 0),  # Yellow
        (192, 0, 192),  # Pink
        (0, 192, 192),  # Cyan
        (0, 192, 0),    # Green
        (0, 0, 192),    # Blue
        (192, 0, 0),    # Red
        (0, 192, 128),  # Teal
        (0, 128, 192),  # Sky Blue
        (128, 192, 0),  # Lime
        (192, 0, 128),  # Magenta
        (128, 0, 192),  # Purple
        (192, 128, 0),  # Orange
        (192, 192, 128),# Light Yellow
        (128, 192, 192),# Light Cyan
        (192, 128, 192),# Light Magenta
        (192, 192, 192),# Light Gray
        (128, 128, 128),# Dark Gray
        (0, 0, 0)       # Black
    ]
    
    for score, label, box in zip(results[0]['scores'], results[0]['labels'], results[0]['boxes']):
        # Convert box coordinates to int
        box = box.cpu().numpy()
        box = [int(x) for x in box]
        
        # Get color for this class
        color = colors[results[0]['labels'].index(label)]
        
        # Draw rectangle
        draw.rectangle(box, outline=color, width=3)
        
        # Draw label with score
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1]-20), label_text, fill=color)
    
    return image

def detect(image, model, processor, device = "cuda"):
    
    text = "car . truck . bus . motorcycle . bicycle . tricycle . van . suv . trailer . construction vehicle . moped . recreational vehicle . pedestrian . cyclist . wheelchair . stroller . traffic light . traffic sign . traffic cone . traffic island . traffic box . barrier . bollard . warning sign . debris . machinery . dustbin . concrete block . cart . chair . basket . suitcase . dog . phone booth ."
    # text = "a car. a truck. a bus. a motorcycle. a bicycle. a tricycle. a van. an suv. a trailer. a construction vehicle. a moped. a recreational vehicle. a pedestrian. a cyclist. a wheelchair. a stroller. a traffic light. a traffic sign. a traffic cone. a traffic island. a traffic box. a barrier. a bollard. a warning sign. a debris. a machinery. a dustbin. a concrete block. a cart. a chair. a basket. a suitcase. a dog. a phone booth."
    
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    
    return results
    
if __name__ == "__main__":
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming=True)
    os.makedirs("dino_samples", exist_ok = True)
    for item in dataset:
        image = item["image"]
        results = detect(image, model, processor, device)
        output_path = f"./dino_samples/{item['id']}_detected.jpg"
        annotated_image = draw_boxes(image, results)
        annotated_image.save(output_path)
    """
    results:{
        labels:[], labels of detected objects
        boxes:[], bounding boxes of detected objects
    }
    """
    # Draw boxes and save"
    