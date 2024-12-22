import json
from collections import defaultdict

coda_categories = {
        'vehicles': [
            'car', 'truck', 'bus', 'van', 'suv', 'trailer', 
            'construction vehicle', 'recreational vehicle'
        ],
        'vulnerable_road_users': [
            'pedestrian', 'cyclist', 'motorcycle', 'bicycle', 'tricycle',
            'moped', 'wheelchair', 'stroller'
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
            'traffic island', 'traffic box', 'debris', 'machinery', 'dustbin',
            'cart', 'chair', 'basket', 'suitcase', 'dog', 'phone booth'
        ]
    }

def get_ordered_object_list(categories_dict):
    """Create an ordered list of all objects from the categories dictionary"""
    all_objects = []
    for category_objects in categories_dict.values():
        all_objects.extend(category_objects)
    return all_objects

def process_regional_data(objects_list, ordered_objects):
    """Process data for regional images which have a flat list structure"""
    presence_vector = [0] * len(ordered_objects)
    if objects_list:  # if list is not empty
        for obj in objects_list:
            object_label = obj['label']
            if object_label in ordered_objects:
                idx = ordered_objects.index(object_label)
                presence_vector[idx] = 1
    return presence_vector

def process_general_data(image_data, ordered_objects):
    """Process data for general images which have a category-based structure"""
    presence_vector = [0] * len(ordered_objects)
    for category_name, objects_list in image_data.items():
        if objects_list:  # if list is not empty
            for obj in objects_list:
                object_label = obj['label']
                if object_label in ordered_objects:
                    idx = ordered_objects.index(object_label)
                    presence_vector[idx] = 1
    return presence_vector

def create_presence_vectors(json_file, categories_dict):
    # Get ordered list of all possible objects
    ordered_objects = get_ordered_object_list(categories_dict)
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize separate dictionaries for each task type
    general_vectors = {}
    regional_vectors = {}
    suggestion_vectors = {}
    
    # Process each image in the JSON
    for image_id, image_data in data.items():
        # Choose processing method based on image_id
        if 'regional' in image_id:
            presence_vector = process_regional_data(image_data, ordered_objects)
            regional_vectors[image_id] = presence_vector
        elif 'suggestion' in image_id:
            presence_vector = process_general_data(image_data, ordered_objects)
            suggestion_vectors[image_id] = presence_vector
        else:  # general case
            presence_vector = process_general_data(image_data, ordered_objects)
            general_vectors[image_id] = presence_vector
    
    return general_vectors, regional_vectors, suggestion_vectors, ordered_objects

def print_presence_vectors(vectors_dict, task_name):
    """Helper function to print the presence vectors in a readable format"""
    print(f"\nPresence Vectors for {task_name} task:")
    print("{")
    for image_id, vector in vectors_dict.items():
        print(f'    "{image_id}": {vector},')
    print("}")

def main():
    # Generate presence vectors
    general_vectors_for_rag, regional_vectors_for_rag, suggestion_vectors_for_rag, ordered_objects = create_presence_vectors(
        'processed_outputs_v2/cleaned_train_metadata.json', 
        coda_categories
    )
    
    general_vectors, regional_vectors, suggestion_vectors, ordered_objects = create_presence_vectors(
        'processed_outputs_v2/cleaned_test_metadata.json', 
        coda_categories
    )
    
    # Initialize a combined matches dictionary
    combined_matches = {}

    # Process matches for general, regional, and suggestion tasks
    for image_id, presence_vector in general_vectors.items():
        matches = [
            image_id_rag
            for image_id_rag, rag_vec in general_vectors_for_rag.items()
            if rag_vec == presence_vector and image_id != image_id_rag
        ]
        combined_matches[image_id] = matches

    for image_id, presence_vector in regional_vectors.items():
        matches = [
            image_id_rag
            for image_id_rag, rag_vec in regional_vectors_for_rag.items()
            if rag_vec == presence_vector and image_id != image_id_rag
        ]
        combined_matches[image_id] = matches

    for image_id, presence_vector in suggestion_vectors.items():
        matches = [
            image_id_rag
            for image_id_rag, rag_vec in suggestion_vectors_for_rag.items()
            if rag_vec == presence_vector and image_id != image_id_rag
        ]
        combined_matches[image_id] = matches

    # print(combined_matches)
    # Write combined matches to a new JSON file
    output_file = 'processed_outputs_v2/match_results.json'
    with open(output_file, 'w') as f:
        json.dump(combined_matches, f, indent=4)
    
    print(f"Match results saved to {output_file}")

if __name__ == "__main__":
    main()
