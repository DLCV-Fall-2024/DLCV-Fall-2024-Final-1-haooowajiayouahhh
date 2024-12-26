file_id_1=""
file_id_2=""
file_id_3=""

# Download the files
gdown --id $file_id_1 --O ./checkpoints/llava-v1.5-7b-task-lora-general-hf.zip
gdown --id $file_id_2 --O ./checkpoints/llava-v1.5-7b-task-lora-suggestion-hf.zip
gdown --id $file_id_3 --O ./checkpoints/llava-v1.5-7b-task-lora-regional-hf.zip

# Unzip the files
unzip ./checkpoints/llava-v1.5-7b-task-lora-general-hf -d ./checkpoints/
unzip ./checkpoints/llava-v1.5-7b-task-lora-suggestion-hf -d ./checkpoints/
unzip ./checkpoints/llava-v1.5-7b-task-lora-regional-hf-d ./checkpoints/

# Remove the zip files

