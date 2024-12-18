import json
from datasets import load_dataset

def get_format_objects(image_id, metadata):
    # Read metadata JSON
    # with open(metadata_path, 'r') as f:
    #     metadata = json.load(f)
    
    # # Get task type from image_id
    task_type = image_id.split('_')[1]  # general, regional, or suggestion
    assert task_type in ["general", "regional", "suggestion"], "Invalid task type"
    assert image_id.split('_')[1] == task_type, "Mismatched task type"
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
            return "No objects detected. Please analyze the scene and provide general driving recommendations appropriate for this environment and situation."
    
        description = """Here are the key objects detected in the scene. Based on these and the overall driving environment, provide appropriate driving suggestions covering:
- General speed and following distance recommendations
- Basic safety precautions for this type of road/area
- Relevant traffic rules and regulations
- Common driving practices for this scenario"""
    
    # Filter objects - keep only immediate and short range
        if isinstance(objects, dict):
            for category in objects:
                objects[category] = [obj for obj in objects[category] 
                                if obj['depth_category'] in ['immediate', 'short range']]
            
            # If after filtering, no immediate/short range objects remain
            if not any(objects.values()):
                return "While no immediate objects are detected, please provide general driving recommendations appropriate for this road environment and situation."
                
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
        
        # Check if this is a regional task
        task_type = image_id.split('_')[1]
        if task_type == "regional":
            # For regional tasks, only include task description, object info and final instruction
            prompt_objects = get_format_objects(image_id, self.metadata_path)
            prompt_generate = """Based on the provided image and detected objects in the red rectangle, give a valid description of the object and its impact on driving behavior."""
            
            final_prompt = (
                f"{prompt_task_description}\n\n"
                f"{prompt_objects}\n\n"
                f"{prompt_generate}"
            )
            
            return final_prompt
        
        # For non-regional tasks, continue with original logic including examples
        prompt_study_examples = """Below are example prompts that demonstrate the expected FORMAT for your response. You should:
- Copy the writing style and structure
- Follow how objects and their impacts are described
- Match the level of detail and organization
These examples come from similar driving scenarios but may contain different objects and situations that you should not assume are present in the current image."""
        
        prompt_examples = ""
        example_image_ids = self.rag_results[image_id]
        for i, idx in enumerate(example_image_ids[baseOn][:num_examples], 1):
            conv = self.convdata[idx].strip()
            prompt_examples += f"\nExample {i}:\n{conv}\n"

        prompt_generate = """Based on the provided image and detected objects, give a valid description according to the task."""
        
        prompt_objects = get_format_objects(image_id, self.metadata_path)
        
        final_prompt = (
            f"{prompt_task_description}\n\n"
            f"{prompt_study_examples}\n"
            f"{prompt_examples}\n"
            "END OF EXAMPLES\n\n"
            f"{prompt_objects}\n\n"
            f"{prompt_generate}"
        )

        return final_prompt
    
def get_human_message(conversations):
    """Extract the human message value from conversations"""
    for message in conversations:
        if message["from"] == "human":
            return message["value"]
    return None

def main():
    # Specify your paths and configurations here
    rag_path = "storage/rag_test.json"
    metadata_path = "processed_outputs/test_metadata.json"
    conversation_path = "storage/conversations.json"
    num_examples = 5

    # Load test dataset without streaming
    test_dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming = True)
    print("Successfully loaded test dataset")

    # Load RAG results
    with open(rag_path, 'r') as f:
        rag_results = json.load(f)
    print("Successfully loaded RAG results")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print("Successfully loaded metadata")

    # Load conversations
    with open(conversation_path, 'r') as f:
        conversations = json.load(f)
    print("Successfully loaded conversations")

    # Initialize prompt processor
    processor = PromptProcessor(
        convdata=conversations,
        rag_results=rag_results,
        metadata_path=metadata
    )

    # Test for each task type
    task_types = ["general", "regional", "suggestion"]
    for task_type in task_types:
        print(f"\n=== Testing {task_type} perception task ===\n")
        
        count = 0
        for sample in test_dataset:
            image_id = sample['id']
            if image_id.split('_')[1] == task_type and count < num_examples:
                try:
                    # Get just the human message value
                    question_message = get_human_message(sample['conversations'])
                    if question_message is None:
                        raise ValueError(f"No human message found in conversations for {image_id}")
                    
                    # Generate prompt
                    prompt = processor.get_prompts(
                        image_id=image_id,
                        question_message=question_message,
                        baseOn='vit_similar_images'
                    )
                    
                    # Print output
                    print(f"Image ID: {image_id}")
                    print("Generated Prompt:")
                    print("-" * 50)
                    print(prompt)
                    print("-" * 50)
                    print("\n")
                    
                    count += 1
                except Exception as e:
                    print(f"Error processing {image_id}: {e}\n")
                    print(f"Sample conversations: {sample['conversations']}")
            
            if count >= num_examples:
                break

if __name__ == "__main__":
    main()