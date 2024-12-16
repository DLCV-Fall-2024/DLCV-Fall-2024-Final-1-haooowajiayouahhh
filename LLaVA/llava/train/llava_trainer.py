from transformers import Trainer
from typing import Dict, Optional, Union
import os
import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import PeftModel
import logging

def get_peft_state_maybe_zero_3(named_params, bias):
    """Extract LoRA state dict, handling DeepSpeed ZeRO-3 params."""
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
        
    # Handle DeepSpeed ZeRO-3 params
    for k, v in to_return.items():
        if hasattr(v, "ds_id"):
            with zero.GatheredParameters([v]):
                to_return[k] = v.data.detach().cpu().clone()
        else:
            to_return[k] = v.detach().cpu().clone()
            
    return to_return

class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        """
        Save a checkpoint of the training state, focusing on LoRA parameters.
        
        Args:
            model: The model to save
            trial: A possible trial object for hyperparameter search
            metrics (optional): The current training metrics
        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training state
        self.save_state()
        
        # Handle LoRA model saving
        print("Model type:", type(self.model))
        if isinstance(self.model, PeftModel):
            print("Saving LoRA parameters...")
            self.model.save_pretrained(output_dir)

            # model.peft_config.save_pretrained(output_dir)
            logging.info(f"Saved LoRA parameters to {output_dir}")
        else:
            logging.warning("Model is not a PeftModel. Saving entire model state.")
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(output_dir)
            else:
                torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save optimizer and scheduler states
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        # # Save metrics if provided
        # if metrics is not None:
        #     torch.save(metrics, os.path.join(output_dir, "metrics.pt"))
        
        if self.args.save_total_limit is not None:
            self._rotate_checkpoints(use_mtime=True,output_dir=output_dir)
            
    def _load_from_checkpoint(self, resume_from_checkpoint):
        """
        Load training state from checkpoint, focusing on LoRA parameters.
        
        Args:
            resume_from_checkpoint: Path to the checkpoint directory
        """
        if isinstance(self.model, PeftModel):
            # Load LoRA parameters
            adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            if os.path.exists(adapter_path):
                lora_state_dict = torch.load(adapter_path, map_location="cpu")
                self.model.load_state_dict(lora_state_dict, strict=False)
                logging.info(f"Loaded LoRA parameters from {adapter_path}")
            else:
                logging.warning(f"No LoRA parameters found at {adapter_path}")
        
        # Load optimizer and scheduler states
        optimizer_path = os.path.join(resume_from_checkpoint, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu")
            )
            
        scheduler_path = os.path.join(resume_from_checkpoint, "scheduler.pt")
        if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu")
            )

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Override train to handle LoRA-specific checkpoint loading.
        """
        # Handle checkpoint loading before training
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
            
        # Proceed with normal training
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)