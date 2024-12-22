import json
from PIL import Image, ImageDraw
from datasets import load_dataset
import os
from tqdm import tqdm
from collections import defaultdict

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

def get_horizontal_pos(position):
    if position % 3 == 0:
        return "left"
    elif position % 3 == 1:
        return "front"
    else:
        return "right"

class ObjectCleaner:
    def __init__(self):
        self.coda_categories = coda_categories
        
    def filter_by_count_and_area(self, objects: list, max_count: int) -> list:
        """Keep only the N largest objects by area."""
        if not objects:
            return []
        return sorted(objects, key=lambda x: x.get('area', 0), reverse=True)[:max_count]
    
    def filter_vehicles_general(self, vehicles: list) -> list:
        """Apply general task vehicle filtering rules."""
        common_vehicles = ['car', 'truck', 'bus', 'van', 'suv']
        special_vehicles = ['trailer', 'construction vehicle', 'recreational vehicle']
        
        # Group vehicles by position
        position_groups = defaultdict(list)
        for vehicle in vehicles:
            pos = get_horizontal_pos(vehicle.get('position', 0))
            position_groups[pos].append(vehicle)
        
        filtered_vehicles = []
        
        # Process each position
        for pos, vehicle_list in position_groups.items():
            common_vehicle_objects = [v for v in vehicle_list if v['label'].lower() in common_vehicles]
            special_vehicle_objects = [v for v in vehicle_list if v['label'].lower() in special_vehicles]
            
            filtered_vehicles.extend(self.filter_by_count_and_area(common_vehicle_objects, 2))
            filtered_vehicles.extend(self.filter_by_count_and_area(special_vehicle_objects, 1))
            
        return filtered_vehicles
    
    def filter_vehicles_suggestion(self, vehicles: list) -> list:
        """Apply suggestion task vehicle filtering rules."""
        return self.filter_by_count_and_area(vehicles, 2)
    
    def filter_road_users(self, users: list) -> list:
        """Apply vulnerable road user filtering rules."""
        user_groups = defaultdict(list)
        for user in users:
            user_groups[user['label']].append(user)
        
        filtered_users = []
        # Keep top 3 pedestrians
        filtered_users.extend(self.filter_by_count_and_area(user_groups.get('pedestrian', []), 3))
        
        # Keep top 1 of each other type
        for label, objects in user_groups.items():
            if label != 'pedestrian':
                filtered_users.extend(self.filter_by_count_and_area(objects, 1))
                
        return filtered_users
    
    def filter_traffic_cones(self, cones: list) -> list:
        """Keep at most 2 cones in each section for immediate and short range."""
        valid_cones = [cone for cone in cones if cone.get('depth_category') in ['immediate', 'short range']]
        
        position_groups = defaultdict(list)
        for cone in valid_cones:
            pos = get_horizontal_pos(cone.get('position', 0))
            position_groups[pos].append(cone)
        
        filtered_cones = []
        for pos in ['left', 'right', 'front']:
            filtered_cones.extend(self.filter_by_count_and_area(position_groups[pos], 2))
            
        return filtered_cones
    
    def clean_objects(self, data: dict) -> dict:
        """Apply task-specific filtering rules to clean the data."""
        cleaned_data = {}
        
        for test_name, test_data in data.items():
            task = test_name.split('_')[1]
            cleaned_data[test_name] = {}
            
            # For regional tasks, copy data as-is
            if task == 'regional':
                cleaned_data[test_name] = test_data
                continue
            
            # For general and suggestion tasks
            for category, objects in test_data.items():
                if not isinstance(objects, list):
                    cleaned_data[test_name][category] = objects
                    continue
                
                if category == 'vehicles':
                    if task == 'suggestion':
                        cleaned_data[test_name][category] = self.filter_vehicles_suggestion(objects)
                    else:  # general
                        cleaned_data[test_name][category] = self.filter_vehicles_general(objects)
                
                elif category == 'vulnerable_road_users':
                    if task == 'suggestion':
                        cleaned_data[test_name][category] = self.filter_by_count_and_area(objects, 1)
                    else:  # general
                        cleaned_data[test_name][category] = self.filter_road_users(objects)
                
                elif category in ['traffic_signs', 'traffic_lights']:
                    cleaned_data[test_name][category] = self.filter_by_count_and_area(objects, 1)
                
                elif category == 'traffic_cones':
                    if task == 'suggestion':
                        cleaned_data[test_name][category] = self.filter_by_count_and_area(objects, 1)
                    else:  # general
                        cleaned_data[test_name][category] = self.filter_traffic_cones(objects)
                
                elif category == 'barriers':
                    cleaned_data[test_name][category] = objects  # keep all barriers
                
                elif category == 'other_objects':
                    if task == 'suggestion':
                        cleaned_data[test_name][category] = self.filter_by_count_and_area(objects, 1)
                    else:  # general
                        other_groups = defaultdict(list)
                        for obj in objects:
                            other_groups[obj['label']].append(obj)
                        filtered_others = []
                        for group_objects in other_groups.values():
                            filtered_others.extend(self.filter_by_count_and_area(group_objects, 1))
                        cleaned_data[test_name][category] = filtered_others
                
        return cleaned_data

def draw_boxes(image, objects):
    """Draw bounding boxes on image for cleaned objects"""
    draw = ImageDraw.Draw(image)
    depth_colors = {
        "immediate": (255, 0, 0),       # Red for immediate range
        "short range": (255, 165, 0),   # Orange for short range
        "mid range": (255, 255, 0),     # Yellow for mid range
        "long range": (0, 255, 0)       # Green for long range
    }
    
    for category, items in objects.items():
        if not isinstance(items, list):
            continue
            
        for obj in items:
            box = obj['bbox']
            label = obj['label']
            depth = obj['depth_category']
            color = depth_colors[depth]
            
            # Draw bounding box
            draw.rectangle(box, outline=color, width=3)
            
            # Create label text
            label_text = f"{label} ({depth})"
            
            # Calculate text size for background
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

def main():
    # Initialize cleaner
    cleaner = ObjectCleaner()
    
    # Setup directories
    output_dir = "processed_outputs_v2"
    image_dir = "cleaned_images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    with open('processed_outputs_v2/test_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Clean objects
    print("Cleaning objects...")
    cleaned_data = cleaner.clean_objects(metadata)
    
    # Save cleaned metadata
    print("Saving cleaned metadata...")
    with open(os.path.join(output_dir, "cleaned_test_metadata.json"), "w") as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Load dataset and create visualizations
    print("Creating visualizations...")
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split='test', streaming=True)
    
    for item in tqdm(dataset, desc="Drawing boxes", bar_format='{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}{postfix}]'):
        image_id = item['id']
        
        if image_id not in cleaned_data:
            continue
            
        image = item['image']
        cleaned_objects = cleaned_data[image_id]
        
        if 'regional' in image_id:
            continue
        # Draw boxes for cleaned objects
        annotated_image = draw_boxes(image.copy(), cleaned_objects)
        
        # Save annotated image
        output_path = os.path.join(image_dir, f"{image_id}_cleaned.jpg")
        annotated_image.save(output_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()