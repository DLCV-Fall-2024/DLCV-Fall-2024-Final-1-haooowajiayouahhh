general_id="1n6ZWSvfmTVKD1AF737VA9ge6ChisJRuN"
suggestion_id="1XMuhCplnn82pyyun_GupxT5Q1jdKqE_Y"
regional_id="1S-DNglTZShUyjYtISREiacP_KntOvOC_"

# general:
# https://drive.google.com/file/d/1n6ZWSvfmTVKD1AF737VA9ge6ChisJRuN/view?usp=sharing
# suggestion:
# https://drive.google.com/file/d/1XMuhCplnn82pyyun_GupxT5Q1jdKqE_Y/view?usp=sharing
# regional:
# https://drive.google.com/file/d/1S-DNglTZShUyjYtISREiacP_KntOvOC_/view?usp=sharing

# Download the files
gdown $general_id -O ./checkpoints/llava-v1.5-7b-task-lora-general-hf.zip
gdown $suggestion_id -O ./checkpoints/llava-v1.5-7b-task-lora-suggestion-hf.zip
gdown $regional_id -O ./checkpoints/llava-v1.5-7b-task-lora-regional-hf.zip

# Unzip the files
unzip ./checkpoints/llava-v1.5-7b-task-lora-general-hf.zip -d ./checkpoints/
unzip ./checkpoints/llava-v1.5-7b-task-lora-suggestion-hf.zip -d ./checkpoints/
unzip ./checkpoints/llava-v1.5-7b-task-lora-regional-hf-d.zip ./checkpoints/

# Remove the zip files

