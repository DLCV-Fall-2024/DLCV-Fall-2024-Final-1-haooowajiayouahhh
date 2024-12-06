from transformers import pipeline
from datasets import load_dataset
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
import torch
import cv2

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-large-hf")
dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming=True)
output_dir = "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/Depth-Anything/outputs"


def depth_estimation(item):
    """
    Estimate depth for an object within a bounding box and categorize it into ranges.
    
    Args:
        item: dictionary containing the image data and bbox for corresponding object from dataset
        bbox: tuple of (x1, y1, x2, y2) coordinates for the bounding box
        
    Returns:
        item: dictionary with depth category added
    """
    # Get depth map for the entire image
    try:
        image = item["image"]
        w, h = image.size
        result = pipe(image)
        depth_map = np.array(result["predicted_depth"])
        depth_map_tensor = torch.tensor(depth_map)
        depth_image = F.interpolate(depth_map_tensor[None, None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255.0
        depth_image = depth_image.cpu().numpy().astype(np.uint8)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, f"{item['id']}_depth.png"), depth_image)
        
        # Extract the region of interest using the bbox
        x1, y1, x2, y2 = map(int, bbox)
        roi_depth = depth_map[y1:y2, x1:x2]
        
        # Calculate average depth in the ROI
        avg_depth = np.mean(roi_depth)
        
        # Define thresholds for depth ranges
        # Note: These thresholds are preliminary and can be adjusted
        immediate_threshold = 0.3
        short_range_threshold = 0.5
        midrange_threshold = 0.7
        
        # Categorize the depth
        if avg_depth <= immediate_threshold:
            item["depth_category"] = "immediate"
        elif avg_depth <= short_range_threshold:
            item["depth_category"] = "short range"
        elif avg_depth <= midrange_threshold:
            item["depth_category"] = "midrange"
        else:
            item["depth_category"] = "long range"
        
        return item
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return item


if __name__ == "__main__":
    bbox = (50, 50, 150, 150)
    for item in dataset:
        print(type(item['image']))
        item = depth_estimation(item)
        