import os
import json 
import torch
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset
import transformers
from prompt_processor import RAGDataHandler, CODAPromptGenerator, RefinedPromptGenerator

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
    base_model_id = "llava-hf/llava-1.5-7b-hf"
    # ckpt_path = './checkpoints/llava-v1.5-7b-task-lora-exp1-hf'
    ckpt_path = base_model_id  # model name or path to your model
    rag_file = "processed_outputs_v2/match_results.json"
    train_data = "storage/conversations.json"
    metadata_file = "processed_outputs_v2/cleaned_test_metadata.json"
    output_file = "submission.json"
    unrefined_input_path = "submission.json"
    
    print("Loading model and processor...")
    model, processor = load_model_and_processor(base_model_id, ckpt_path)
    
    # Load metadata if exists
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    print("Initializing prompt generator...")
    rag_handler = RAGDataHandler(rag_file, train_data)
    prompt_generator = RefinedPromptGenerator(rag_handler, unrefined_input_path)
    
    print("Loading test dataset...")
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming = True)
    
    predictions = {}
    print("Starting inference...")
    
    with torch.inference_mode():
        for item in tqdm(dataset):
            image = item['image']
            image_id = item['id']
            print(image_id)
                
            # Generate prompt using prompt processor
            prompt = prompt_generator.generate_prompt(
                image_id=image_id,
                metadata=metadata,
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
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()