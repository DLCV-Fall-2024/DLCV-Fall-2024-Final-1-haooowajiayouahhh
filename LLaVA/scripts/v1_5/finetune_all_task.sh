#!/bin/bash 
# echo general
# bash scripts/v1_5/finetune_general_lora.sh
echo regional 
bash scripts/v1_5/finetune_regional_lora.sh
echo suggestion
bash scripts/v1_5/finetune_suggestion_lora.sh