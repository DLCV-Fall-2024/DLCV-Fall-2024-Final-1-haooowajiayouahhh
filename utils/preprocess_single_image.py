import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline
import torch.nn.functional as F
from utils.get_object_vec import create_object_presence_vector

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

# Import necessary functions from preprocess.py
from preprocess import (
    detect_objects,
    process_image,
    draw_boxes,
    format_objects,
    get_depth_category,
    get_position,
    calculate_iou
)

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    results, annotated_image = preprocess_single_image(image_path, task="general")
    
    # Save results
    annotated_image.save("output_detected.jpg")
    print("Detection results:", results)
