def create_object_presence_vector(formatted_output):
    """
    Convert formatted output into binary vector indicating presence of each object type
    
    Args:
        formatted_output (dict): Dictionary containing detected objects by CODA categories
        
    Returns:
        list: Binary vector where 1 indicates object presence and 0 indicates absence
    """
    # Define all object categories from text
    categories = [
        'car', 'truck', 'bus', 'van', 'suv', 'trailer', 'construction vehicle', 'recreational vehicle',
        'pedestrian', 'cyclist', 'motorcycle', 'bicycle', 'tricycle', 'moped', 'wheelchair', 'stroller',
        'traffic sign', 'warning sign',
        'traffic light',
        'traffic cone',
        'barrier', 'bollard', 'concrete block',
        'traffic island', 'traffic box', 'debris', 'machinery', 'dustbin', 'cart', 'chair', 'basket', 'suitcase', 'dog', 'phone booth'
    ]
    
    # Initialize vector with zeros
    presence_vector = [0] * len(categories)
    
    # Check each category in formatted output
    for category_group in formatted_output.values():
        for obj in category_group:
            try:
                # Find index of detected object in categories list
                idx = categories.index(obj['label'].lower())
                presence_vector[idx] = 1
            except ValueError:
                print(f"Warning: Unknown object label {obj['label']}")
                continue
                
    return presence_vector

# Example usage:
if __name__ == "__main__":
    # Example formatted output
    example_output = {
        'vehicles': [], 
        'vulnerable_road_users': [], 
        'traffic_signs': [], 
        'traffic_lights': [], 
        'traffic_cones': [
            {'label': 'traffic cone', 'bbox': [951, 567, 981, 622], 
             'depth_value': 0.433, 'depth_category': 'short range', 
             'position': 'right'},
            {'label': 'traffic cone', 'bbox': [1063, 610, 1134, 718],
             'depth_value': 0.794, 'depth_category': 'immediate',
             'position': 'right'}
        ],
        'barriers': [],
        'other_objects': []
    }
    
    # Get presence vector
    vector = create_object_presence_vector(example_output)
    print(vector)