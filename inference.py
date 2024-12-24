import os
import json 
import torch
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset
import transformers
from prompt_processor import RAGDataHandler, CODAPromptGenerator
import argparse

def load_model_and_processor(base_model_id, ckpt_path: str):
    """Load LLaVA model and processor"""
    processor = transformers.AutoProcessor.from_pretrained(base_model_id, revision='a272c74')
    model = transformers.LlavaForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, processor

def clean_response(response: str) -> str:
    """Clean model output by removing prefixes and extra whitespace"""
    # Split on common assistant markers and take the last part
    markers = ["Assistant:", "ASSISTANT:", "assistant:"]
    for marker in markers:
        if marker in response:
            response = response.split(marker)[-1]
    return response.strip()

def main():
    # Configurations
    parser = argparse.ArgumentParser(description="Inference Configuration")
    parser.add_argument('--base_model_id', type=str, default="llava-hf/llava-1.5-7b-hf", help='Base model ID')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/r128a256', help='Path to the model checkpoint')
    parser.add_argument('--rag_file', type=str, default="processed_outputs_v2/match_results.json", help='Path to the RAG file')
    parser.add_argument('--train_data', type=str, default="storage/conversations.json", help='Path to the training data')
    parser.add_argument('--metadata_file', type=str, default="processed_outputs_v2/cleaned_test_metadata.json", help='Path to the metadata file')
    parser.add_argument('--output_path', type=str, default="result/", help='Path to save the output file')
    
    args = parser.parse_args()  # Parse the arguments

    print("Loading model and processor...")
    gen_path = os.path.join(args.ckpt_path, 'llava-v1.5-7b-task-lora-general-hf')
    reg_path = os.path.join(args.ckpt_path, 'llava-v1.5-7b-task-lora-regional-hf')
    sug_path = os.path.join(args.ckpt_path, 'llava-v1.5-7b-task-lora-suggestion-hf')
    gen_model, processor = load_model_and_processor(args.base_model_id, gen_path)  # Use args instead of hardcoded values
    reg_model, _ = load_model_and_processor(args.base_model_id, reg_path) 
    sug_model, _ = load_model_and_processor(args.base_model_id, sug_path)     
    # Load metadata if exists
    metadata = {}
    if os.path.exists(args.metadata_file):  # Use args
        with open(args.metadata_file, 'r') as f:  # Use args
            metadata = json.load(f)
    
    print("Initializing prompt generator...")
    rag_handler = RAGDataHandler(args.rag_file, args.train_data)
    prompt_generator = CODAPromptGenerator(rag_handler)
    
    print("Loading test dataset...")
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test")
    dataset= dataset.take(5)
    predictions = {}
    print("Starting inference...")
    with torch.inference_mode():
        for item in tqdm(dataset):
            image = item['image']
            image_id = item['id']
            task = image_id.split('_')[1]
            if task == 'general':
                model = gen_model
            if task == 'regional':
                model = reg_model
            if task == 'suggestion':
                model = sug_model
            
            # Generate prompt using prompt processor
            prompt = prompt_generator.generate_prompt(
                image_id=image_id,
                metadata=metadata
            )
            # print(prompt)
            # Format conversation for LLaVA
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            input_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            # Prepare inputs
            inputs = processor(
                images=image,
                text=input_prompt,
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1000,
                    min_new_tokens=10,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # Decode and clean response
            response = processor.decode(output[0], skip_special_tokens=True)
            cleaned_response = clean_response(response)
            
            # Store prediction
            predictions[image_id] = cleaned_response
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save predictions
 # Create directory if it doesn't exist
    save_path = os.path.join(args.output_path, 'submission.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    print(f"Saving predictions to {save_path}")  
    with open(save_path, 'w') as f:  # Use args
        json.dump(predictions, f, indent=2)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()