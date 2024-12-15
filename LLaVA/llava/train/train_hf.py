import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List,Any,Union
from datasets import load_dataset
import torch
from tqdm import tqdm
import transformers
from transformers import AutoProcessor

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava.model import *
from PIL import Image
from prompt_processor import PromptProcessor
local_rank = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                          metadata={"help": "Path to the training data."})
    convdata: str = field(default=None,
                          metadata={"help": "Path to the conversation data."})
    rag_results: str = field(default=None,
                            metadata={"help": "Path to the RAG results."})
    metadata_path: str = field(default=None,
                              metadata={"help": "Path to the metadata."})
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded."
        },
    )
    task: str = field(default="general")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    deepspeed: Optional[str] = field(default=None)  # Add this line
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress quantization statistics through double quantization."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # LoRA specific arguments
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

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

class HuggingfaceSupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 processor: AutoProcessor,
                 data_args: DataArguments,task='general'):
        super(HuggingfaceSupervisedDataset, self).__init__()
        
        self.dataset = load_dataset(data_path)
        self.train_data = self.dataset['train']
        if task == 'general':
            self.train_data=dataset.filter(lambda example: example["id"].startswith("Train_general"))
        elif task == 'regional':
            dataset=dataset.filter(lambda example: example["id"].startswith("Train_regional"))
        elif task ==  'suggestion':
            dataset=dataset.filter(lambda example: example["id"].startswith("Train_suggestion"))
        self.processor = processor
        self.data_args = data_args
        self.task=task

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.train_data[i]
        sources = [item] if isinstance(i, int) else item
        
        # Extract and process text
        conversations = sources[0]['conversations']
        if isinstance(conversations, list):
            # Create input text from human messages and target text from assistant messages
            input_texts = []
            target_texts = []
            
            for j, conv in enumerate(conversations):
                if conv['from'] == 'human':
                    input_texts.append(conv['value'])
                elif conv['from'] == 'gpt':
                    target_texts.append(conv['value'])
                    
            # Combine texts
            input_text = " ".join(input_texts)
            target_text = " ".join(target_texts)
        else:
            input_text = conversations
            target_text = conversations
        
        if 'image' in sources[0]:
            image = sources[0]['image']
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert('RGB')
            #add extra infos for input text
            image_id=sources[0]['id']
            assert image_id.split('_')[1] == self.task         
            prompt_processor=PromptProcessor(self.data_args.convdata, self.data_args.rag_results, self.data_args.metadata_path)
            system_text=get_system_prompt(image_id.split('_')[1])
            input_text=format_conversation(system_text, input_text)
            input_text= prompt_processor.get_prompts(image_id, input_text, 'vit_similar_images')
            print("[DEBUG@HuggingfaceSupervisedDataset] input_text:\n", input_text)
            # Process inputs
            inputs = self.processor(
                images=image,
                text=input_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.data_args.model_max_length
            )
            
            # Process targets/labels
            with self.processor.tokenizer.as_target_tokenizer():
                labels = self.processor.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.data_args.model_max_length
                )["input_ids"]
            
            # Remove the batch dimension
            for k in inputs.keys():
                if torch.is_tensor(inputs[k]) and inputs[k].ndim > 0:
                    inputs[k] = inputs[k].squeeze(0)
            
            inputs['labels'] = labels.squeeze(0)
            
        else:
            # Text-only processing
            inputs = self.processor(
                text=input_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.data_args.model_max_length
            )
            
            # Process labels
            with self.processor.tokenizer.as_target_tokenizer():
                labels = self.processor.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.data_args.model_max_length
                )["input_ids"]
            
            # Remove batch dimension
            for k in inputs.keys():
                if torch.is_tensor(inputs[k]) and inputs[k].ndim > 0:
                    inputs[k] = inputs[k].squeeze(0)
                    
            inputs['labels'] = labels.squeeze(0)
            
            # Add empty image tensor for consistency if needed
            if self.data_args.is_multimodal:
                inputs['pixel_values'] = torch.zeros(
                    3,
                    self.processor.image_processor.size['height'],
                    self.processor.image_processor.size['width']
                )
                
        return inputs

@dataclass
class DataCollatorForLLaVA:
    """Data collator for LLaVA that handles labels."""
    processor: Any
    max_length: int = 512
    ignore_pad_token_for_loss: bool = True
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        
        # Handle input_ids and attention_mask (these are integer tensors, should not require grad)
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
                
        # Handle pixel_values if present (these are float tensors, can require grad)
        if 'pixel_values' in features[0]:
            batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])
            # Only set requires_grad for floating point tensors
            if batch['pixel_values'].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                batch['pixel_values'].requires_grad_(True)
            
        # Process labels if present
        if 'labels' in batch:
            padding_side = self.processor.tokenizer.padding_side
            pad_token_id = self.processor.tokenizer.pad_token_id
            
            if self.ignore_pad_token_for_loss:
                labels = batch['labels'].clone()
                if padding_side == 'right':
                    labels[labels == pad_token_id] = -100
                else:
                    mask = labels.eq(pad_token_id)
                    labels = labels.masked_fill(mask, -100)
                batch['labels'] = labels
                
        return batch
def make_supervised_data_module(processor: AutoProcessor,
                              data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = HuggingfaceSupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        task=data_args.task
    )
    print(len(train_dataset))
    data_collator = DataCollatorForLLaVA(
        processor=processor,
        max_length=data_args.model_max_length
    )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

def find_all_linear_names(model):
    """Helper function to find all linear layer names for LoRA"""
    linear_layers = []
    
    # Target language model layers
    language_model_targets = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # attention
        'gate_proj', 'up_proj', 'down_proj'       # mlp
    ]
    
    # Target projector layers (if needed)
    projector_targets = [
        'linear_1', 'linear_2'  # multi-modal projector
    ]
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Extract the layer name
            layer_name = name.split('.')[-1]
            
            # Skip lm_head
            if layer_name == 'lm_head':
                continue
                
            # Language model layers
            if layer_name in language_model_targets:
                linear_layers.append(layer_name)
                continue
                
            # Multi-modal projector layers (optional)
            if layer_name in projector_targets:
                linear_layers.append(layer_name)
                
    # Remove duplicates while preserving order
    return list(dict.fromkeys(linear_layers))
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
def print_trainable_parameters(model):
    """
    Prints the names and shapes of all trainable parameters in the model.
    Also prints the total number of trainable parameters.
    
    Args:
        model: PyTorch model
    """
    trainable_params = []
    all_param = 0
    trainable_param = 0
    
    # Iterate through all parameters
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        
        if param.requires_grad:
            trainable_params.append(f"Module name: {name}, Shape: {param.shape}")
            trainable_param += num_params
    
    print("Trainable modules:")
    for param in trainable_params:
        print(param)
    
    print(f"\nTotal Parameters: {all_param:,}")
    print(f"Trainable Parameters: {trainable_param:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_param / all_param:.2f}%")

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize the processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path
    )

    # Compute dtype based on training args
    compute_dtype = (torch.float16 if training_args.fp16 else 
                    (torch.bfloat16 if training_args.bf16 else torch.float32))

    # Initialize the model
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type='nf4'
        )
        model_kwargs = {
            "device_map": "auto",
            "quantization_config": quantization_config
        }
    else:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": compute_dtype if training_args.bits == 16 else None
        }

    model = transformers.LlavaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        use_flash_attention_2=True,
        **model_kwargs
    )
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):  
            model.gradient_checkpointing_enable()
        # Enable input gradients
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
                if isinstance(input, tuple):
                    for i in input:
                        if torch.is_tensor(i):
                            i.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # Prepare model for k-bit training if needed
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
                
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        # After model initialization
        for param in model.parameters():
            if not param.requires_grad:
                param.requires_grad = True

        # If using LoRA, only set requires_grad for LoRA parameters
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Prepare the dataset
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args
    )
    print(training_args)

    trainer = LLaVATrainer(
        model=model,
        args=training_args,
        **data_module
    )

    # Train the model
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    

    # Save the model
    if training_args.local_rank in [-1, 0]:
        if training_args.lora_enable:
            # Save LoRA weights and configuration
            model.save_pretrained(training_args.output_dir)
        else:
            trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()