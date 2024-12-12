import numpy as np

def create_single_object_presence_vector(formatted_output):
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
    presence_vector = np.array(presence_vector)
    return presence_vector

def object_vec_dist(output1, output2):
    """
    Calculate L2 norm between presence vectors of two formatted outputs
    
    Args:
        output1 (dict): First formatted output dictionary
        output2 (dict): Second formatted output dictionary
        
    Returns:
        float: L2 norm (Euclidean distance) between the two presence vectors
    """
    # Get binary presence vectors for both outputs
    vec1 = np.array(create_object_presence_vector(output1))
    vec2 = np.array(create_object_presence_vector(output2))
    
    # Calculate L2 norm
    return int(np.sum((vec1 - vec2) ** 2))
    
def create_object_presence_vector(batch_formatted_outputs):
    """
    Convert a batch of formatted outputs into binary vectors indicating presence of each object type
    
    Args:
        batch_formatted_outputs (list): List of dictionaries, each containing detected objects by CODA categories
        
    Returns:
        list: List of binary vectors where each vector indicates object presence for a single formatted output
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
    
    # Initialize list to hold presence vectors for each formatted output
    batch_presence_vectors = []
    
    # Process each formatted output in the batch
    for formatted_output in batch_formatted_outputs:
        # Initialize vector with zeros for the current formatted output
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
        
        # Append the presence vector for the current formatted output to the batch list
        batch_presence_vectors.append(presence_vector)
    
    return batch_presence_vectors
    
# Example usage:
if __name__ == "__main__":
    # Example formatted outputs
    output1 = {
        'vehicles': [], 
        'vulnerable_road_users': [], 
        'traffic_signs': [], 
        'traffic_lights': [], 
        'traffic_cones': [
            {'label': 'traffic cone', 'bbox': [951, 567, 981, 622], 
             'depth_value': 0.433, 'depth_category': 'short range', 
             'position': 'right'}
        ],
        'barriers': [],
        'other_objects': []
    }
    
    output2 = {
        'vehicles': [
            {'label': 'car', 'bbox': [100, 100, 200, 200],
             'depth_value': 0.5, 'depth_category': 'mid range',
             'position': 'left'}
        ],
        'vulnerable_road_users': [],
        'traffic_signs': [],
        'traffic_lights': [],
        'traffic_cones': [
            {'label': 'traffic cone', 'bbox': [951, 567, 981, 622], 
             'depth_value': 0.433, 'depth_category': 'short range', 
             'position': 'right'}
            ],
        'barriers': [],
        'other_objects': []
    }
    
    distance = object_vec_dist(output1, output2)
    print(f"L2 distance between outputs: {distance}")