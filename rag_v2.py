import json
from collections import defaultdict
import random
import argparse

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

relaxed_categories = {
    'small_vehicles': [
        'car', 'truck'
    ],
    'mid_vehicles': [
        'bus', 'van', 'suv'
    ],
    'large_vehicles': [
        'trailer', 'construction vehicle', 'recreational vehicle'
    ],
    'human_vulnerable_road_users': [
        'pedestrian', 'cyclist'
    ],
    'object_vulnerable_road_users': [
        'motorcycle', 'bicycle'
    ],
    'man_powered_vulnerable_road_users': [
        'tricycle', 'moped', 'wheelchair', 'stroller'
    ],
    'traffic_sings_and_lights': [
        'traffic sign', 'warning sign', 'traffic light'
    ],
    'traffic_cones_and_barriers': [
        'traffic cone', 'barrier', 'bollard', 'concrete block'
    ],
    'other_objects': [
        'traffic island', 'traffic box', 'debris', 'machinery', 'dustbin',
        'cart', 'chair', 'basket', 'suitcase', 'dog', 'phone booth'
    ]
}

# relaxed_categories = {
#         'vehicles': [
#             'car', 'truck', 'bus', 'van', 'suv', 'trailer', 
#             'construction vehicle', 'recreational vehicle',
#             'pedestrian', 'cyclist', 'motorcycle', 'bicycle', 'tricycle',
#             'moped', 'wheelchair', 'stroller',
#             'traffic sign', 'warning sign',
#             'traffic light',
#             'traffic cone',
#             'barrier', 'bollard', 'concrete block',
#             'traffic island', 'traffic box', 'debris', 'machinery', 'dustbin',
#             'cart', 'chair', 'basket', 'suitcase', 'dog', 'phone booth'
#         ]
# }

# relaxed_categories = coda_categories

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
    relaxed_presence_vector = [0] * len(relaxed_categories)
    # print(f'presence_vector_len: {len(relaxed_categories)}')
    for category_name, objects_list in image_data.items():
        if objects_list:  # if list is not empty
            for obj in objects_list:
                object_label = obj['label']
                for category_index, (category_name, category_objects) in enumerate(relaxed_categories.items()):
                    # print(f"category_index: {category_index}")
                    # print(f"category_name: {category_name}")
                    # print(f"category_objects: {category_objects}")
                    if object_label in category_objects:
                        # print(f"fucking happened!")
                        # print(f"category_index: {category_index}")
                        relaxed_presence_vector[category_index] = 1
                if object_label in ordered_objects:
                    idx = ordered_objects.index(object_label)
                    presence_vector[idx] = 1


    return presence_vector, relaxed_presence_vector

def create_presence_vectors(json_file, categories_dict):
    # Get ordered list of all possible objects
    ordered_objects = get_ordered_object_list(categories_dict)
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize separate dictionaries for each task type
    general_vectors = {}
    relaxed_general_vectors = {}
    regional_vectors = {}
    suggestion_vectors = {}
    relaxed_suggestion_vectors = {}
    
    # Process each image in the JSON
    for image_id, image_data in data.items():
        # Choose processing method based on image_id
        if 'regional' in image_id:
            presence_vector = process_regional_data(image_data, ordered_objects)
            regional_vectors[image_id] = presence_vector
        elif 'suggestion' in image_id:
            presence_vector, relaxed_presence_vector = process_general_data(image_data, ordered_objects)
            suggestion_vectors[image_id] = presence_vector
            relaxed_suggestion_vectors[image_id] = relaxed_presence_vector
        else:  # general case
            presence_vector, relaxed_presence_vector = process_general_data(image_data, ordered_objects)
            general_vectors[image_id] = presence_vector
            relaxed_general_vectors[image_id] = relaxed_presence_vector
    
    return general_vectors, regional_vectors, suggestion_vectors, relaxed_general_vectors, relaxed_suggestion_vectors, ordered_objects

def print_presence_vectors(vectors_dict, task_name):
    """Helper function to print the presence vectors in a readable format"""
    print(f"\nPresence Vectors for {task_name} task:")
    print("{")
    for image_id, vector in vectors_dict.items():
        print(f'    "{image_id}": {vector},')
    print("}")

def main(args):
    # Generate presence vectors
    general_vectors_for_rag, regional_vectors_for_rag, suggestion_vectors_for_rag, relaxed_general_vectors_for_rag, relaxed_suggestion_vectors_for_rag, ordered_objects = create_presence_vectors(
        'processed_outputs_v2/cleaned_train_metadata.json', 
        coda_categories
    )
    
    general_vectors, regional_vectors, suggestion_vectors, relaxed_general_vectors, relaxed_suggestion_vectors, ordered_objects = create_presence_vectors(
        'processed_outputs_v2/cleaned_test_metadata.json', 
        coda_categories
    )
    
    # Initialize a combined matches dictionary
    combined_matches = {}

    # Process matches for general, regional, and suggestion tasks
    for (image_id, presence_vector), (relaxed_image_id, relaxed_presence_vecotr) in zip(general_vectors.items(), relaxed_general_vectors.items()):
        combined_matches[image_id] = {}

        matches = [
            image_id_rag
            for image_id_rag, rag_vec in general_vectors_for_rag.items()
            if rag_vec == presence_vector and image_id != image_id_rag
        ]
        if len(matches) > args.min_samples:
            matches = random.sample(matches, args.min_samples)
        combined_matches[image_id]["perfect_matches"] = matches
        match_count = len(matches)
        if match_count < args.min_samples:
            matches = []
            # print(f"empty_general_id: {image_id}")
            for image_id_rag, rag_vec in relaxed_general_vectors_for_rag.items():
                if rag_vec == relaxed_presence_vecotr and image_id != image_id_rag:
                    matches.append(image_id_rag)
                    match_count += 1
                if match_count >= args.min_samples:
                    break
            # print(f"matches: {matches}")
            combined_matches[image_id]["relaxed_matches"] = matches

    for image_id, presence_vector in regional_vectors.items():
        matches = [
            image_id_rag
            for image_id_rag, rag_vec in regional_vectors_for_rag.items()
            if rag_vec == presence_vector and image_id != image_id_rag
        ]
        if len(matches) > args.min_samples:
            matches = random.sample(matches, args.min_samples)
        combined_matches[image_id] = matches

    for (image_id, presence_vector), (relaxed_image_id, relaxed_presence_vecotr) in zip(suggestion_vectors.items(), relaxed_suggestion_vectors.items()):
        combined_matches[image_id] = {}
        matches = [
            image_id_rag
            for image_id_rag, rag_vec in suggestion_vectors_for_rag.items()
            if rag_vec == presence_vector and image_id != image_id_rag
        ]
        if len(matches) > args.min_samples:
            matches = random.sample(matches, args.min_samples)
        combined_matches[image_id]["perfect_matches"] = matches
        match_count = len(matches)
        if match_count < args.min_samples:
            # print(f"empty_index: {image_id}")
            matches = []
            for image_id_rag, rag_vec in relaxed_suggestion_vectors_for_rag.items():
                if rag_vec == relaxed_presence_vecotr and image_id != image_id_rag:
                    matches.append(image_id_rag)
                    match_count += 1
                if match_count >= args.min_samples:
                    break
            combined_matches[image_id]["relaxed_matches"] = matches

    # print(combined_matches)
    # Write combined matches to a new JSON file
    output_file = 'processed_outputs_v2/match_results.json'
    with open(output_file, 'w') as f:
        json.dump(combined_matches, f, indent=4)
    
    print(f"Match results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate presence vectors for RAG dataset')   
    parser.add_argument('--relax', action='store_true', help='Use relaxed categories')
    parser.add_argument('--min_samples', type=int, default=5, help='Minimum number of samples for each image_id')
    args = parser.parse_args()
    main(args)
