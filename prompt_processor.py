import json
from datasets import load_dataset

def get_format_objects(image_id, metadata_path):
    # Read metadata JSON
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get task type from image_id
    task_type = image_id.split('_')[1]  # general, regional, or suggestion
    
    # Get objects for this image
    objects = metadata.get(image_id, [])
    empty_message = "No objects are pre-detected in this image. Please analyze the image carefully and identify the relevant objects yourself."

    # Different descriptions for each task type
    if task_type == "general":
        if not objects:
            return empty_message
        description = """Here are some objects pre-detected in the image. Use this only as a reference and verify for yourself in the image - check the colors, orientations, and other details as shown in the example prompts above."""
        
    elif task_type == "suggestion":
        if not objects:
            return empty_message
        description = """Here are the immediately relevant objects detected in the image that may require immediate attention for driving decisions."""
        # Filter objects - keep only immediate and short range
        if isinstance(objects, dict):
            for category in objects:
                objects[category] = [obj for obj in objects[category] 
                                   if obj['depth_category'] in ['immediate', 'short range']]
                
    elif task_type == "regional":
        if not objects:
            return "No object is detected in the specified region. Please analyze the region marked by the red rectangle and describe what you see."
        description = """Here is the object detected in the specified region. Please verify this detection and provide a detailed description of the object's appearance and its potential impact on driving behavior."""
        obj = objects[0]
        return f"{description}\n{chr(10)}A {obj['label']} is detected at {obj['depth_category']} in the {obj['position']} portion of the image."

    # Format objects for general and suggestion tasks
    if task_type in ["general", "suggestion"]:
        formatted_text = [description]  # Start with the task-specific description
        
        object_lines = []  # Store object lines
        for category in ['vehicles', 'vulnerable_road_users', 'traffic_signs', 
                        'traffic_lights', 'traffic_cones', 'barriers', 'other_objects']:
            items = objects.get(category, [])
            if items:
                object_groups = {}
                for item in items:
                    key = (item['label'], item['depth_category'], item['position'])
                    object_groups[key] = object_groups.get(key, 0) + 1
                
                category_texts = []
                for (label, depth, position), count in object_groups.items():
                    count_text = f"{count} " if count > 1 else "a "
                    plural = "s" if count > 1 else ""
                    category_texts.append(f"{count_text}{label}{plural} at {depth} in the {position} portion.")
                
                if category_texts:
                    object_lines.append(f"{category.replace('_', ' ').title()}: {', '.join(category_texts)}")
        
        # Join description and objects with single newline
        if object_lines:
            return f"{description}\n{chr(10).join(object_lines)}"
        return empty_message

    return "Invalid task type"

class PromptProcessor:
    def __init__(self, convdata, rag_results, metadata_path):
        """
        convdata: conversations, e.g. gt convos
        rag_results: rag results of train or test sets
        metadata_path: processed outputs path
        
        """
        self.convdata = convdata
        self.rag_results = rag_results
        self.metadata_path = metadata_path
    
    def get_prompts(self, image_id, question_message, baseOn, num_examples=2):
        # Task description
        prompt_task_description = question_message.strip()

        # Study examples header
        prompt_study_examples = """Below are some example prompts provided solely to help you understand how to analyze the input image. They come from similar driving scenarios but may contain different objects and situations that you should not assume are present in the current image."""

        # Format examples with numbering - limit to num_examples
        prompt_examples = ""
        example_image_ids = self.rag_results[image_id]
        for i, idx in enumerate(example_image_ids[baseOn][:num_examples], 1):  # Limit examples
            conv = self.convdata[idx].strip()
            prompt_examples += f"\n{i}. {conv}\n"

        # Generate instruction
        prompt_generate = """Based on the provided image and detected objects, give a valid description according to the task."""
        
        prompt_objects = get_format_objects(image_id, self.metadata_path)
        
        # Combine all sections with controlled separation
        final_prompt = (
            f"{prompt_task_description}\n\n"
            f"{prompt_study_examples}\n"
            f"{prompt_examples}\n"
            "END OF EXAMPLES\n\n"
            f"{prompt_objects}\n\n"
            f"{prompt_generate}"
        )

        return final_prompt

def main():
    # Load resources
    with open("/work/dlhlp5689lab/DLCV/DLCV-Fall-2024-Final-1-haooowajiayouahhh/storage/vitpatch32rag_test.json", 'r') as file:
        rag_results = json.load(file)
    with open("/work/dlhlp5689lab/DLCV/DLCV-Fall-2024-Final-1-haooowajiayouahhh/storage/conversations.json", 'r') as file:
        convdata = json.load(file)
        
    metadata_path = "/work/dlhlp5689lab/DLCV/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs/test_metadata.json"
    
    # Initialize prompt processor
    prompt_processor = PromptProcessor(convdata, rag_results, metadata_path)

    # Load dataset
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test")

    # Set hyperparameters
    num_examples = 2  # Number of examples to show in prompts
    samples_per_task = 5  # Number of images to test per task type

    # Test for each task type
    task_types = ['general', 'regional', 'suggestion']
    
    for task_type in task_types:
        print(f"\n{'='*50}")
        print(f"Testing {task_type.upper()} task:")
        print(f"{'='*50}\n")
        
        # Find first two images for this task type
        count = 0
        for item in dataset:
            if task_type in item['id'].lower():
                # Get question message
                question_message = ""
                for conv in item['conversations']:
                    if conv["from"] in ["human", "system"]:
                        question_message += conv["value"] + " "
                question_message = question_message.strip()

                # Generate and print prompt
                print(f"Image ID: {item['id']}")
                print("-" * 30)
                final_prompt = prompt_processor.get_prompts(item['id'], question_message, 'vit_similar_images', num_examples)
                print(final_prompt)
                print("\n")
                
                count += 1
                if count >= samples_per_task:
                    break
                
if __name__ == "__main__":
    main()