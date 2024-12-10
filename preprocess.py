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
from utils.get_object_vec import create_object_presence_vector

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
coda_categories = {
    'vehicles': [
        'car', 'truck', 'bus', 'van', 'suv', 'trailer', 'construction vehicle', 'recreational vehicle'
    ],
    
    'vulnerable_road_users': [
        'pedestrian', 'cyclist', 'motorcycle', 'bicycle', 'tricycle', 'moped', 'wheelchair', 'stroller'
    ],
    
    'traffic_signs': [
        'traffic sign', 'warning sign'
    ],
    
    'traffic_lights': [
        'traffic light'
    ],
    
    'traffic_cones': [
        'traffic cone'
    ],
    
    'barriers': [
        'barrier', 'bollard', 'concrete block'
    ],
    
    'other_objects': [
        'traffic island', 'traffic box', 'debris', 'machinery', 'dustbin', 'cart', 'chair', 'basket', 'suitcase', 'dog', 'phone booth'
    ]
}

def get_object_category(label):
    """Find which CODA category an object belongs to"""
    for category, objects in coda_categories.items():
        if label.lower() in objects:
            return category
    return "other_objects"

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
        "immediate": (255, 0, 0),       # Red for immediate range
        "short range": (255, 165, 0),   # Orange for short range
        "mid range": (255, 255, 0),     # Yellow for mid range
        "long range": (0, 255, 0)       # Green for long range
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
        box_threshold=0.27,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    return results

def get_depth_category(depth_value):
    thresholds = {
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

def format_objects(task, objects, image, image_id):
    """
    Format detected objects into CODA-LM categories
    
    Args:
        task: "general", "suggestion", or "regional"
        objects: List of detected objects with their properties
        image: PIL Image object
        image_id: Image identifier
        
    Returns:
        For general/suggestion: Dictionary with 7 CODA categories
        For regional: List of objects in red box
    """
    if task == "general":
        formatted_result = {
            'vehicles': [],
            'vulnerable_road_users': [],
            'traffic_signs': [],
            'traffic_lights': [],
            'traffic_cones': [],
            'barriers': [],
            'other_objects': []
        }
        
        # Sort each object into its appropriate category
        for obj in objects:
            category = get_object_category(obj['label'])
            formatted_result[category].append(obj)
            
        return formatted_result
    
    elif task == "suggestion":
        formatted_result = {
            'vehicles': [],
            'vulnerable_road_users': [],
            'traffic_signs': [],
            'traffic_lights': [],
            'traffic_cones': [],
            'barriers': [],
            'other_objects': []
        }
        
        # Sort each object into its appropriate category
        for obj in objects:
            category = get_object_category(obj['label'])
            formatted_result[category].append(obj)
            
        return formatted_result
    
    elif task == "regional" :
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Define range for red color in BGR
        lower_red = np.array([0, 0, 150])
        upper_red = np.array([50, 50, 255])
        mask = cv2.inRange(img_np, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            red_box = [x, y, x+w, y+h]
            
            # Find object with highest IoU
            best_obj = None
            best_iou = 0.8  # Minimum IoU threshold
            
            for obj in objects:
                obj_box = obj['bbox']
                iou = calculate_iou(red_box, obj_box)
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj
            
            # Print clean output
            # print(f"\nImage {image_id}:", end=" ")
            if best_obj:
                # print(best_obj['label'])
                return [best_obj]
            else:
                print("Nothing found in red box")
                return []
            
        print(f"\nImage {image_id}: No red box detected")
        return []

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 < x2 and y1 < y2:
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        return intersection / union
    return 0.0
    
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
    # dataset = dataset.take(200)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    # Process each image
    print("Processing images...")
    results = {}

    for item in tqdm(dataset, 
                    desc="Processing",
                    bar_format='{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'):
        
        image_id = item['id']
        task = image_id.split('_')[1]
            
        image = item['image']
        
        # Process image and get results
        objects = process_image(image, depth_pipe, obj_model, obj_processor, device) # after DINO and depth anything
        # save coda'd images
        annotated_image = draw_boxes(image.copy(), objects)
        annotated_image.save(os.path.join(output_dir, f"{image_id}_detected.jpg"))
        
        formatted_objects = format_objects(task, objects, image, image_id)
        print(f"\nImage {image_id}:", end=" ")
        print(formatted_objects)

        results[image_id] = formatted_objects
    
    # Save metadata
    with open(os.path.join(output_dir, "train_metadata.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()