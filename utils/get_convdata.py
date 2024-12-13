import json
from datasets import load_dataset

def save_conversations_to_json(dataset_name, split, output_file):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Prepare the data to save
    data_to_save = {}
    for item in dataset:
        data_to_save[item["id"]] = item["conversations"]

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(data_to_save, f, indent=4)

if __name__ == "__main__":
    save_conversations_to_json("ntudlcv/dlcv_2024_final1", "train", "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/storage/conversations.json")
