from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from pathlib import Path
import json

@dataclass
class DetectedObject:
    """Class representing a detected object in an image"""
    label: str
    depth_category: str  # 'immediate', 'short range', 'mid range', 'long range'
    position: str  # Position description in image

class RAGDataHandler:
    """Handler for RAG data loading and example retrieval"""
    def __init__(self, rag_file_path: str, train_data_path: str):
        self.rag_mapping = self._load_json(rag_file_path)
        self.train_data = self._load_json(train_data_path)
        
    def _load_json(self, file_path: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
            
    def get_similar_examples(self, test_id: str, num_examples: int = 2) -> List[Tuple[str, str, str]]:
        """
        Get examples for a test image, prioritizing perfect matches
        Returns list of tuples: (match_type, train_id, example_text)
        """
        matches = self.rag_mapping.get(test_id, {})
        examples = []
        
        # Get perfect matches first
        perfect_matches = matches.get('perfect_matches', [])
        for train_id in perfect_matches:
            if train_id in self.train_data and len(examples) < num_examples:
                examples.append(('Perfect match', train_id, self.train_data[train_id]))
                
        # If we have enough perfect matches, return them
        if len(examples) == num_examples:
            return examples
            
        # Otherwise, use relaxed matches to fill remaining slots
        relaxed_matches = matches.get('relaxed_matches', [])
        for train_id in relaxed_matches:
            if train_id in self.train_data and len(examples) < num_examples:
                examples.append(('Relaxed match', train_id, self.train_data[train_id]))
                
        return examples

class PromptBuilder:
    """Class for building task-specific prompts"""
    
    TASK_DESCRIPTIONS = {
        "general": """There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.""",
        
        "regional": """Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.""",
        
        "suggestion": """There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene."""
    }
    
    EXAMPLE_DESCRIPTIONS = {
        "general": """These are similar driving scenarios as references. Follow their format and writing style, but analyze the current image independently:""",
        
        "regional": """These are similar examples of describing marked objects. Follow their format and writing style, but focus on the current object:""",
        
        "suggestion": """These are similar driving scenarios as references. Follow their format and writing style for suggestions, but focus on the current scene:"""
    }
    
    @staticmethod
    def format_object_count(count: int, label: str) -> str:
        """Format object count and label"""
        if count == 1:
            return f"a {label}"
        return f"some {label}s"

    @staticmethod
    def get_position_text(position: str) -> str:
        """Convert numeric position (0-8) to directional text
        0,3,6 -> left
        1,4,7 -> front (use 'in')
        2,5,8 -> right
        """
        try:
            pos_num = int(position)
            if pos_num in [0, 3, 6]:
                return 'on the left'
            elif pos_num in [1, 4, 7]:
                return 'in front'
            elif pos_num in [2, 5, 8]:
                return 'on the right'
            return 'in front'  # default
        except ValueError:
            return 'in front'  # default if position is not a number
        
    @staticmethod
    def format_detected_objects(metadata: dict, task_type: str) -> str:
        """Format detected objects based on task type"""
        if not metadata:
            return "No objects detected. Please analyze the image carefully."
            
        if task_type == "regional":
            obj = metadata[0] if metadata else None
            if not obj:
                return "No object detected in the marked region. Please analyze the region carefully."
            position = PromptBuilder.get_position_text(obj['position'])
            return f"- {PromptBuilder.format_object_count(1, obj['label'])} {position}"
            
        # For general and suggestion tasks
        categories = [
            'vehicles', 'vulnerable_road_users', 'traffic_signs',
            'traffic_lights', 'traffic_cones', 'barriers', 'other_objects'
        ]
        
        sections = []
        for category in categories:
            if category in metadata and metadata[category]:
                objects = metadata[category]
                # Group objects by label and position
                grouped = {}
                for obj in objects:
                    position = PromptBuilder.get_position_text(obj['position'])
                    key = (obj['label'], position)
                    grouped[key] = grouped.get(key, 0) + 1
                
                if grouped:
                    items = []
                    for (label, position), count in grouped.items():
                        items.append(f"- {PromptBuilder.format_object_count(count, label)} {position}")
                    if items:
                        section = f"{category.replace('_', ' ').title()}:\n" + "\n".join(items)
                        sections.append(section)
                    
        return "\n\n".join(sections) if sections else "No relevant objects detected. Please analyze the image carefully."

class CODAPromptGenerator:
    """Main class for generating CODA challenge prompts"""
    def __init__(self, rag_handler: RAGDataHandler):
        self.rag_handler = rag_handler
        self.prompt_builder = PromptBuilder()
        
    def generate_prompt(self, image_id: str, question: str, metadata: dict) -> str:
        """Generate complete prompt for a given task"""
        task_type = image_id.split('_')[1]
        
        # Get task-specific components
        task_description = self.prompt_builder.TASK_DESCRIPTIONS[task_type]
        example_description = self.prompt_builder.EXAMPLE_DESCRIPTIONS[task_type]
        
        # Get examples with match type
        examples = self.rag_handler.get_similar_examples(image_id)
        formatted_examples = "\n\n".join([
            f"Example {i+1}:\n{example}" 
            for i, (match_type, _, example) in enumerate(examples)
        ])
        
        # Check if metadata exists for this image
        image_metadata = metadata.get(image_id, {})
        
        # Format detected objects with new formatting style
        if not image_metadata:
            objects_section = "Note: No objects pre-detected. The above examples are only for format reference. Please analyze the image carefully."
        else:
            objects_section = "The following objects have been detected (verify these in the image):\n" + \
                            self.prompt_builder.format_detected_objects(image_metadata, task_type)
        
        # Combine all components
        prompt_parts = [
            task_description,
            example_description,
            formatted_examples,
            "END OF EXAMPLES",
            objects_section
        ]
        
        return "\n\n".join(prompt_parts)

def main():
    # Initialize paths
    rag_file = "processed_outputs_v2/match_results.json"
    train_data_file = "storage/conversations.json"
    metadata_file = "processed_outputs_v2/cleaned_test_metadata.json"
    
    # Load data
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Initialize handlers
    rag_handler = RAGDataHandler(rag_file, train_data_file)
    generator = CODAPromptGenerator(rag_handler)
    
    # Load test dataset
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming=True)
    
    # Test each task type
    for task_type in ['general', 'regional', 'suggestion']:
        print(f"\n=== Testing {task_type} task ===\n")
        count = 0
        
        for sample in dataset:
            if count >= 5:  # Test 2 examples per task
                break
                
            image_id = sample['id']
            if image_id.split('_')[1] != task_type:
                continue
                
            # Get question from conversations
            question = next((msg['value'] for msg in sample['conversations'] 
                           if msg['from'] == 'human'), None)
            if not question:
                continue
                
            # Generate prompt
            prompt = generator.generate_prompt(image_id, question, metadata)
            
            print(f"Image ID: {image_id}")
            print("-" * 50)
            print(prompt)
            print("-" * 50)
            print()
            
            count += 1

if __name__ == "__main__":
    main()