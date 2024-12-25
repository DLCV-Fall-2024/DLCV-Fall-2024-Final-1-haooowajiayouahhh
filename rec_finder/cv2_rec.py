# import os
# import json
# import cv2
# import numpy as np
# import torch
# import clip
# from PIL import Image
# from datasets import load_dataset
# from torch.utils.data import DataLoader

# from classifier import CLIPClassifier

# # ------------------------
# # 1. Define the rectangle-finding function
# # ------------------------
# def find_red_rectangle(image_np, image_id, min_width=10, min_height=10):
#     """
#     Detect a red rectangle in the image and return the cropped region (if any).
#     Args:
#         image_np (np.ndarray): OpenCV image in BGR format.
#     Returns:
#         np.ndarray or None: Cropped image if rectangle is found, None otherwise.
#     """
#     hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
#     lower_red1 = np.array([0, 130, 130])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 130, 130])
#     upper_red2 = np.array([180, 255, 255])

#     mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
#     red_mask = cv2.bitwise_or(mask1, mask2)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     rectangles = []
#     for contour in contours:
#         perimeter = cv2.arcLength(contour, True)
#         epsilon = 0.02 * perimeter
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) == 4 and cv2.isContourConvex(approx):
#             x, y, w, h = cv2.boundingRect(approx)
#             if w >= min_width and h >= min_height:
#                 rectangles.append(approx)

#     if len(rectangles) == 1:
#         x, y, w, h = cv2.boundingRect(rectangles[0])
#         cropped_img = image_np[y:y+h, x:x+w]
#         return cropped_img
#     return None







import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader

from classifier import CLIPClassifier


def get_region(x, y, img_width, img_height):
    """
    Determine which 9-region grid (3x3) the point (x, y) belongs to.
    Args:
        x (int): X-coordinate of the point.
        y (int): Y-coordinate of the point.
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.
    Returns:
        int: Region index (1-9) where 1 is top-left and 9 is bottom-right.
    """
    # Divide image into 3x3 grid
    # print(f"x: {x}, y: {y}, img_width: {img_width}, img_height: {img_height}")
    region_width = img_width // 3
    region_height = img_height // 3

    # Calculate region indices
    col = x // region_width
    row = y // region_height

    # Ensure the indices are within bounds (handle border cases)
    col = min(col, 2)
    row = min(row, 2)

    # Compute region index (1 to 9)
    return row * 3 + col
# ------------------------
# 1. Define the rectangle-finding function
# ------------------------
def find_red_rectangle(image_np, image_id, min_width=5, min_height=5, debug_path="/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/red_debug/debug_image"):
    """
    Detect a red rectangle in the image and return the cropped region (if any).
    Args:
        image_np (np.ndarray): OpenCV image in BGR format.
        image_id (str): Unique identifier for the image.
        min_width (int): Minimum width of the rectangle.
        min_height (int): Minimum height of the rectangle.
        debug_path (str): Path to save debug images.
    Returns:
        np.ndarray or None: Cropped image if rectangle is found, None otherwise.
    """
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # Convert to HSV
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(debug_path, f"{image_id}_hsv.png"), hsv_image)

    # Create masks
    lower_red1 = np.array([0, 150, 130])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 130])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    cv2.imwrite(os.path.join(debug_path, f"{image_id}_red_mask.png"), red_mask)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(debug_path, f"{image_id}_cleaned_mask.png"), cleaned_mask)

    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    near_rectangles = []
    contour_image = image_np.copy()
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(contour_image, [approx], -1, (255, 0, 0), 2)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if w >= min_width and h >= min_height:
                angles = []
                num_vertices = len(approx)
                for i in range(num_vertices):
                    pt1 = approx[i][0]
                    pt2 = approx[(i + 1) % num_vertices][0]
                    pt3 = approx[(i + 2) % num_vertices][0]

                    v1 = pt2 - pt1
                    v2 = pt3 - pt2

                    # Compute angle using dot product
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                    angles.append(angle)

                # Count angles near 90 degrees
                best_90_count = sum(85 <= angle <= 95 for angle in angles)
                near_90_count = sum(80 <= angle <= 100 for angle in angles)
                # Check if all angles are within the desired range
                # if all(80 <= angle <= 110 for angle in angles):
                if best_90_count == 4:
                    rectangles.append(approx)
                elif near_90_count >= 3:
                    near_rectangles.append(approx)

    # Draw rectangles on the original image for debugging
    debug_image = image_np.copy()
    cv2.imwrite(os.path.join(debug_path, f"{image_id}contour_image.png"), contour_image)
    for rect in rectangles:
        cv2.drawContours(debug_image, [rect], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(debug_path, f"{image_id}_rectangles.png"), debug_image)
    
    if len(rectangles) == 1:
        x, y, w, h = cv2.boundingRect(rectangles[0])
        img_height, img_width = image_np.shape[:2]
        position = get_region(x+w/2,y+h/2,img_width,img_height)
        # cropped_img = image_np[y:y+h, x:x+w]
        padding = 0.5  # Percentage of the rectangle's dimensions to expand
        new_x = max(0, int(x - w * padding))
        new_y = max(0, int(y - h * padding))
        new_w = min(img_width - new_x, int(w * (1 + 2 * padding)))
        new_h = min(img_height - new_y, int(h * (1 + 2 * padding)))

        cropped_img = image_np[new_y:new_y + new_h, new_x:new_x + new_w]
        cv2.imwrite(os.path.join("/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/red_debug/cropped_image", f"{image_id}_cropped.png"), cropped_img)
        return cropped_img, position
    elif len(near_rectangles) > 0:
        x, y, w, h = cv2.boundingRect(near_rectangles[0])
        img_height, img_width = image_np.shape[:2]
        position = get_region(x+w/2,y+h/2,img_width,img_height)
        # cropped_img = image_np[y:y+h, x:x+w]
        padding = 0.5  # Percentage of the rectangle's dimensions to expand
        new_x = max(0, int(x - w * padding))
        new_y = max(0, int(y - h * padding))
        new_w = min(img_width - new_x, int(w * (1 + 2 * padding)))
        new_h = min(img_height - new_y, int(h * (1 + 2 * padding)))

        cropped_img = image_np[new_y:new_y + new_h, new_x:new_x + new_w]
        cv2.imwrite(os.path.join("/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/red_debug/cropped_image", f"{image_id}_cropped.png"), cropped_img)
        return cropped_img, position
    else:
        return  None, -1
    # return None