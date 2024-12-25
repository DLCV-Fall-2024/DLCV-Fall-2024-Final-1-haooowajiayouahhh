# experiment 1 : Naive finetune
echo "experiment 1: general task"
bash scripts/v1_5/finetune_general_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json
echo "experiment 1: regional task"
bash scripts/v1_5/finetune_regional_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json
echo "experiment 1: suggestion task"
bash scripts/v1_5/finetune_suggestion_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json

# experiment 2 : RAG + cleaned objects
echo "experiment 2: general task"
bash scripts/v1_5/finetune_general_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json
echo "experiment 2: regional task"
bash scripts/v1_5/finetune_regional_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json
echo "experiment 2: suggestion task"
bash scripts/v1_5/finetune_suggestion_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json

# experiment 3 : RAG + uncleaned objects
echo "experiment 3: general task"
bash scripts/v1_5/finetune_general_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_rag.json
echo "experiment 3: regional task"
bash scripts/v1_5/finetune_regional_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_rag.json
echo "experiment 3: suggestion task"
bash scripts/v1_5/finetune_suggestion_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_rag.json

# experiment 4 : cleaned objects
echo "experiment 4: general task"
bash scripts/v1_5/finetune_general_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json
echo "experiment 4: regional task"
bash scripts/v1_5/finetune_regional_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json
echo "experiment 4: suggestion task"
bash scripts/v1_5/finetune_suggestion_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/cleaned_train_rag.json

# experiment 5 : uncleaned objects
echo "experiment 3: general task"
bash scripts/v1_5/finetune_general_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_rag.json
echo "experiment 3: regional task"
bash scripts/v1_5/finetune_regional_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_rag.json
echo "experiment 3: suggestion task"
bash scripts/v1_5/finetune_suggestion_lora.sh 16 32 /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_metadata.json /workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v3/uncleaned_train_rag.json

# note: 
# for experiment 1, prompt processor only task descprition
# for experiment 2 3, all
# for experiment 4 5, prompt processor only task descprition and objects_section