import torch
from PIL import Image,ImageDraw
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline
from datasets import load_dataset
import torch.nn.functional as F
import cv2
import os
import json
from tqdm import tqdm

# def draw_boxes(image, results):
#     draw = ImageDraw.Draw(image)
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Simplified colors for brevity
    
#     for score, label, box in zip(results[0]['scores'], results[0]['labels'], results[0]['boxes']):
#         box = box.cpu().numpy()
#         box = [int(x) for x in box]
#         color = colors[results[0]['labels'].index(label) % len(colors)]
#         draw.rectangle(box, outline=color, width=3)
#         label_text = f"{label}: {score:.2f}"
#         draw.text((box[0], box[1]-20), label_text, fill=color)
#     return image
def draw_boxes(image, objects):
    """
    Draw bounding boxes on the image with labels including depth category
    
    Args:
        image: PIL Image object
        objects: List of dictionaries containing detection and depth information
        
    Returns:
        PIL Image with drawn boxes and labels
    """
    draw = ImageDraw.Draw(image)
    colors = {
        "immediate": (255, 0, 0),     # Red for immediate range
        "short range": (0, 255, 0),   # Green for short range
        "mid range": (0, 0, 255),      # Blue for mid range
        "long range": (255, 165, 0)   # Orange for long range
    }
    
    for obj in objects:
        box = obj['bbox']
        label = obj['label']
        depth_category = obj['depth_category']
        position=obj['position']
        color = colors[depth_category]
        
        # Draw bounding box
        draw.rectangle(box, outline=color, width=3)
        
        # Create label text with both object class and depth category
        label_text = f"{label} ({position})"
        # print(label_text)
        
        # Calculate text size to create background rectangle
        text_bbox = draw.textbbox((0, 0), label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw semi-transparent background for text
        draw.rectangle(
            [box[0], box[1]-text_height-4, box[0]+text_width+4, box[1]],
            fill=(0, 0, 0, 128)
        )
        
        # Draw text
        draw.text((box[0]+2, box[1]-text_height-2), label_text, fill=color)
    
    return image
def detect_objects(image, model, processor, device="cuda"):
    text = "car . truck . bus . motorcycle . bicycle . tricycle . van . suv . trailer . construction vehicle . moped . recreational vehicle . pedestrian . cyclist . wheelchair . stroller . traffic light . traffic sign . traffic cone . traffic island . traffic box . barrier . bollard . warning sign . debris . machinery . dustbin . concrete block . cart . chair . basket . suitcase . dog . phone booth ."
    
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    return results

def get_depth_category(depth_value):
    thresholds = {
        1.0: "immediate",
        1.0: "immediate",
        0.6: "short range",
        0.4: "mid range",
        0.15: "long range"
    }
    
    for threshold, category in sorted(thresholds.items()):
        if depth_value <=threshold:
            return category
    # return "long range"
def get_position(bbox, image_width):
    """
    Determine the horizontal position of an object based on its bounding box center
    
    Args:
        bbox: List of [x1, y1, x2, y2] coordinates
        image_width: Width of the full image
        
    Returns:
        str: 'left', 'middle', or 'right'
    """
    # Calculate center x-coordinate of the bounding box
    center_x = (bbox[0] + bbox[2]) / 2
    
    # Define the boundaries for three equal sections
    third_width = image_width / 3
    
    if center_x < third_width:
        return "left"
    elif center_x < 2 * third_width:
        return "middle"
    else:
        return "right"
def process_image(image, depth_pipe, obj_model, obj_processor, device):
    # Object detection
    detection_results = detect_objects(image, obj_model, obj_processor, device)
    
    # Depth estimation
    depth_result = depth_pipe(image)
    depth_map = np.array(depth_result["predicted_depth"])
    
    # Resize depth map to match image size
    w, h = image.size
    depth_map_tensor = torch.tensor(depth_map)
    depth_map = F.interpolate(depth_map_tensor[None, None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_map = depth_map.cpu().numpy()
    depth_max=depth_map.max()
    depth_min=depth_map.min()
    # Process each detected object
    objects = []
    for score, label, box in zip(detection_results[0]['scores'], detection_results[0]['labels'], detection_results[0]['boxes']):
        box = box.cpu().numpy().astype(int)
        roi_depth = depth_map[box[1]:box[3], box[0]:box[2]]
        avg_depth = float(np.mean(roi_depth))
        avg_depth=(avg_depth-depth_min)/(depth_max-depth_min)
        # print("avg_depth: ",avg_depth)
        position = get_position(box, w)
        objects.append({
            "label": label,
            # "confidence": float(score),
            "bbox": box.tolist(),
            "depth_value": avg_depth,
            "depth_category": get_depth_category(avg_depth),
            "position":position
        })
    
    return objects

def main():
    # Initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-base"
    obj_processor = AutoProcessor.from_pretrained(model_id)
    obj_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-large-hf", device=device)
    
    # Setup directories
    output_dir = "processed_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test",streaming=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    # Process each image
    print("Processing images...")
    results = {}

    for item in tqdm(dataset, 
                    desc="Processing",
                    bar_format='{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'):
        
        image_id = item['id']
        image = item['image']
        
        # Process image and get results
        objects = process_image(image, depth_pipe, obj_model, obj_processor, device)
        

        # Store results
        annotated_image = draw_boxes(image.copy(), objects)
        annotated_image.save(os.path.join(output_dir, f"{image_id}_detected.jpg"))
        
        results[image_id] = objects

    
    # Save metadata
    with open(os.path.join(output_dir, "train_metadata.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()