import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader

from classifier import CLIPClassifier, YOLOClassifier
from rec_finder import find_red_rectangle

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        pil_image = sample["image"]
        image_id = sample["id"]
        image_np = np.array(pil_image)
        return image_np, image_id

if __name__ == "__main__":
    # Initialize CLIP model
    print("Available CLIP models:", clip.available_models())
    classifier = CLIPClassifier()
    classifier.set_labels([
        "car", "truck", "bus", "motorcycle", "bicycle", "tricycle", "van", "suv",
        "trailer", "construction vehicle", "moped", "recreational vehicle",
        "pedestrian", "cyclist", "wheelchair", "traffic light",
        "traffic sign", "traffic cone", "traffic island", "traffic box",
        "barrier", "bollard", "warning sign", "machinery", "dustbin",
        "concrete block", "cart", "dog"
    ])
    # classifier  = YOLOClassifier()

    # Load the dataset (non-streaming)
    hf_dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test")
    dataset = ImageDataset(hf_dataset)

    # Wrap the dataset with a DataLoader
    def my_collate_fn(batch):
        # batch is a list of (image, image_id) tuples
        # just return them as a list instead of stacking
        images, image_ids = zip(*batch)
        return list(images), list(image_ids)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=my_collate_fn
    )
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    results = {}
    count = 0
    for batch in dataloader:
        images, image_ids = batch
        # if count  > 3:
            # break
        for i in range(len(images)):
            image = images[i]
            image_id = image_ids[i]

            # Process "regional" images
            if "regional" in image_id:
                print(f"processing {image_id}", end=" ")
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cropped_image, position = find_red_rectangle(image_np, image_id.replace("Test_regional_", "tr"))
                if cropped_image is not None:
                    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                    count += 1
                    print("yay!")
                    #save cropped image to path
                    # cropped_pil.save(f"/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/red_debug/cropped_outputs/{image_id}_cropped.jpg")
                    label = classifier.classify(cropped_pil)
                    if image_id not in results:
                        results[image_id] = []
                    results[image_id].append({
                        "predicted_label": label,
                        "position": int(position)
                    })
                else:
                    print("")
                    label = "no_red_rectangle_found"
                    if image_id not in results:
                        results[image_id] = []
                # results.append({"image_id": image_id, "predicted_label": label})

    print(f"rectangle count : {count}")
    # Write results to JSON
    outputpath = "/workspace/DLCV-Fall-2024-Final-1-haooowajiayouahhh/processed_outputs_v2/regional_test.json"
    with open(outputpath, "w") as f:
        json.dump(results, f, indent=1)

    print(f"Processing complete! Results saved to '{outputpath}'")
