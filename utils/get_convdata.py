import json
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ConversationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]['id'], self.dataset[idx]['conversations'][1]['value']
    
    def __len__(self):
        return len(self.dataset)    

def save_conversations_to_json(dataset_name, split, output_file):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    dataset = ConversationDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=32)

    # Prepare the data to save
    data_to_save = {}
    for batch_ids, batch_convs in tqdm(dataloader):
        batch_ids = list(batch_ids)
        for ids, conv in zip(batch_ids, batch_convs):
            data_to_save[ids] = conv

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(data_to_save, f, indent=4)

if __name__ == "__main__":
    save_conversations_to_json("ntudlcv/dlcv_2024_final1", "train", "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/storage/conversations.json")
