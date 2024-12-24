#!/bin/bash
lora_r=${1:-128}
lora_alpha=${2:-256} 
echo general
bash scripts/v1_5/finetune_general_lora.sh $lora_r $lora_alpha
echo regional 
bash scripts/v1_5/finetune_regional_lora.sh $lora_r $lora_alpha
echo suggestion
bash scripts/v1_5/finetune_suggestion_lora.sh $lora_r $lora_alpha