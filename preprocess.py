import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline
from datasets import load_dataset
import torch.nn.functional as F
import cv2
import os
import json
from tqdm import tqdm

def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Simplified colors for brevity
    
    for score, label, box in zip(results[0]['scores'], results[0]['labels'], results[0]['boxes']):
        box = box.cpu().numpy()
        box = [int(x) for x in box]
        color = colors[results[0]['labels'].index(label) % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1]-20), label_text, fill=color)
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
        0.3: "immediate",
        0.5: "short range",
        0.7: "midrange"
    }
    
    for threshold, category in sorted(thresholds.items()):
        if depth_value <= threshold:
            return category
    return "long range"

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
    
    # Process each detected object
    objects = []
    for score, label, box in zip(detection_results[0]['scores'], detection_results[0]['labels'], detection_results[0]['boxes']):
        box = box.cpu().numpy().astype(int)
        roi_depth = depth_map[box[1]:box[3], box[0]:box[2]]
        avg_depth = float(np.mean(roi_depth))
        
        objects.append({
            "label": label,
            "confidence": float(score),
            "bbox": box.tolist(),
            "depth_value": avg_depth,
            "depth_category": get_depth_category(avg_depth)
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
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    # Process each image
    print("Processing images...")
    results = {}
    for item in tqdm(dataset):
        image_id = item['id']
        image = item['image']
        
        # Process image and get results
        objects = process_image(image, depth_pipe, obj_model, obj_processor, device)
        
        # # Save detection visualization
        # annotated_image = draw_boxes(image.copy(), detection_results)
        # annotated_image.save(os.path.join(output_dir, f"{image_id}_detected.jpg"))
        
        # Save depth visualization
        # depth_map = depth_result["predicted_depth"]
        # depth_vis = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        # cv2.imwrite(os.path.join(output_dir, f"{image_id}_depth.png"), depth_vis)
        
        # Store results
        results[image_id] = objects

    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()