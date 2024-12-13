import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from datasets import load_dataset
import torch
from tqdm import tqdm
import transformers
import tokenizers
from peft import PeftModel

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import argparse
import os
import copy
import json
import torch
from tqdm import tqdm
from itertools import islice
from typing import List, Dict

def load_model_with_weights(base_model_id, checkpoint_dir):
    # Load the base model
    config = transformers.AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    # config.attn_config['attn_impl'] = "triton"
    model = transformers.LlavaForConditionalGeneration.from_pretrained(
        base_model_id,  
        device_map="auto",  # Automatically map to GPU
        low_cpu_mem_usage=True,
    )
    
    # Load non-LoRA weights if they exist
    non_lora_path = os.path.join(checkpoint_dir, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print("Loading non-LoRA weights...")
        non_lora_state = torch.load(non_lora_path, map_location="cpu")
        model.load_state_dict(non_lora_state, strict=False)
    
    # Check if adapter weights exist
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.bin")
    if os.path.exists(adapter_path):
        print("Loading LoRA adapter weights...")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
        )
    
    return model

def get_task_type_from_id(image_id):
    """Extract task type from image_id (e.g., 'test_general_01' -> 'general')"""
    parts = image_id.split('_')
    if len(parts) >= 2:
        return parts[1]  # 'general', 'suggestion', etc.
    return None

def get_system_prompt(task_type):
    """Get the system prompt in LLaVA conversation format"""
    system_messages = {
        "general": "You are an autonomous driving perception expert specializing in comprehensive scene analysis.",
        "suggestion": "You are an autonomous driving expert specializing in providing safe driving recommendations.",
        "regional": "You are an autonomous driving agent that can capture details of specific objects in the red boxes ."
    }
    
    return {
        "from": "system",
        "value": system_messages.get(task_type, system_messages[task_type])
    }

def format_conversation(system_prompt, data_prompt):
    """Format the complete conversation in LLaVA format"""
    conversation = [
        system_prompt,
        data_prompt
    ]
    return conversation



def batch_iterator(iterable, batch_size):
    """Create batches from an iterable"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

def process_batch(batch: List[Dict], processor, model, rag_prompt_processor, device="cuda"):
    """Process a batch of images and prompts"""
    images = []
    prompts = []
    image_ids = []
    
    for item in batch:
        image = item['image']
        image_id = item['id']
        original_conversations = item['conversations']
        
        # Get task type and corresponding system prompt
        task_type = get_task_type_from_id(image_id)
        system_prompt = get_system_prompt(task_type)
        
        if original_conversations and len(original_conversations) > 0:
            conversation = format_conversation(system_prompt, original_conversations[0])
            
            # Convert conversation to string format
            prompt_str = ""
            for message in conversation:
                role = "USER: " if message["from"] in ["human", "system"] else "ASSISTANT: "
                prompt_str += f"{role}{message['value']} "
            question_message = prompt_str.strip()

            full_prompt = rag_prompt_processor.get_prompts(image_id, question_message) + " ASSISTANT: "

            
            images.append(image)
            prompts.append(full_prompt)
            image_ids.append(image_id)
    
    # Process batch
    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors='pt', 
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # Process outputs
    responses = []
    for output in outputs:
        response = processor.decode(output[2:], skip_special_tokens=True)
        if "ASSISTANT: " in response:
            response = response.split("ASSISTANT: ", 1)[1].strip()
        responses.append(response)
    
    return dict(zip(image_ids, responses))

def main():
    parser = argparse.ArgumentParser(description="LLaVA Batch Inference on HF Dataset")
    parser.add_argument('--output_path', type=str, default='submission.json', help='Output file for predictions')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/llava-v1.5-13b-task-lora', help='Path to checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--rag_results', type=str, required=True, help = 'the json file of rag_result')
    parser.add_argument('--convdata', type=str, default='', help = 'the json file about conversation and image_id of training datset')
    args = parser.parse_args()

    # Model and processor setup
    base_model_id = "llava-hf/llava-1.5-7b-hf"
    processor = transformers.AutoProcessor.from_pretrained(base_model_id)
    model = load_model_with_weights(base_model_id, args.checkpoint_dir)
    model.to(args.device)

    #RAG and promptprocessor setup
    with open(args.rag_results, 'r') as file:
        rag_results = json.load(file)
    with open(args.convdata, 'r') as file:
        convdata = json.load(file)
    prompt_processor = PromptProcessor(convdata, rag_results)
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming=True)
    
    predictions = {}
    total_items = 900  # number of test datasets
    
    # Process batches with progress bar
    total_batches = (total_items + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=total_items, desc="Processing batches") as pbar:
        for batch in batch_iterator(dataset, args.batch_size):
            batch_predictions = process_batch(batch, processor, model, args.device)
            predictions.update(batch_predictions)
            pbar.update(len(batch))
            
            # Optional: Add memory management
            if args.device == "cuda":
                torch.cuda.empty_cache()
    
    # Write predictions to JSON file
    print(f"\nSaving predictions to {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    main()