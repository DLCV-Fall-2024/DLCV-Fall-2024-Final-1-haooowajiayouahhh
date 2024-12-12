import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline
import torch.nn.functional as F
from utils.get_object_vec import create_object_presence_vector
from preprocess import (
    detect_objects,
    process_image,
    draw_boxes,
    format_objects,
    get_depth_category,
    get_position,
    calculate_iou
)

def preprocess_single_image(image, task="general"):
    """
    Preprocess a single image for object detection and depth estimation
    
    Args:
        image_path: Path to the input image
        task: "general", "suggestion", or "regional" (default: "general")
        
    Returns:
        formatted_objects: Processed detection results in CODA format
        annotated_image: PIL Image with drawn bounding boxes
    """
    # Initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-base"
    obj_processor = AutoProcessor.from_pretrained(model_id)
    obj_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-large-hf", device=device)
    
    # Detect objects and estimate depth
    objects = process_image(image, depth_pipe, obj_model, obj_processor, device)
    
    # Draw boxes on image
    annotated_image = draw_boxes(image.copy(), objects)
    
    # Format results according to task
    formatted_objects = format_objects(task, objects, image, "single_image")
    
    return formatted_objects, annotated_image

def preprocess_batch_images(images, batch_ids):
    """
    Preprocess a batch of images for object detection and depth estimation
    
    Args:
        images: List of PIL Images
        batch_ids: List of dictionaries containing 'id' and 'task' for each image
                  e.g., [{'id': '1', 'task': 'general'}, {'id': '2', 'task': 'suggestion'}]
        
    Returns:
        batch_results: Dictionary mapping image IDs to their processed results
        annotated_images: Dictionary mapping image IDs to annotated PIL Images
    """
    # Initialize models (only once for batch processing)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-base"
    obj_processor = AutoProcessor.from_pretrained(model_id)
    obj_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-large-hf", device=device)
    
    batch_results = []
    # annotated_images = {}
    
    # Process each image in the batch
    for image, image_id in zip(images, batch_ids):
         # Default to 'general' if task not specified
        if 'general' in image_id['id']:
            task = 'general'
        elif 'regional' in image_id['id']:
            task = 'regional'
        elif 'suggestion' in image_id['id']:
            task = 'suggestion'
        else:
            task = 'general' 
        # Detect objects and estimate depth
        objects = process_image(image, depth_pipe, obj_model, obj_processor, device)
        
        # Draw boxes on image
        annotated_image = draw_boxes(image.copy(), objects)
        
        # Format results according to task
        formatted_objects = format_objects(task, objects, image, f"batch_image_{image_id}")
        
        # Store results
        batch_results.append(formatted_objects)
        # annotated_images[image_id] = annotated_image
    
    # return batch_results, annotated_images
    return batch_results

# Import necessary functions from preprocess.py


if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    results, annotated_image = preprocess_single_image(image_path, task="general")
    
    # Save results
    annotated_image.save("output_detected.jpg")
    print("Detection results:", results)
