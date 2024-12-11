from datasets import load_dataset, Dataset

def create_small_subset(dataset_name, split, num_samples):
    # Load the dataset in streaming mode
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    # Create a small subset by iterating and taking a limited number of samples
    small_subset = []
    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break
        small_subset.append(sample)

    # Convert the list of samples to a Hugging Face dataset
    small_dataset = Dataset.from_list(small_subset)

    # Save the small dataset locally for reuse
    small_dataset.save_to_disk(f"./{dataset_name}_small_subset")
    print(f"Small subset saved to ./{dataset_name}_small_subset")

    return small_dataset
