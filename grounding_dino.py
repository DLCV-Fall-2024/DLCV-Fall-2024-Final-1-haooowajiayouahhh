import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def draw_boxes(image, results):
    # Create a copy of the image to draw on
    draw = ImageDraw.Draw(image)
    
    # Get colors for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Dark Red
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Blue
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]
    
    for score, label, box in zip(results[0]['scores'], results[0]['labels'], results[0]['boxes']):
        # Convert box coordinates to int
        box = box.cpu().numpy()
        box = [int(x) for x in box]
        
        # Get color for this class
        color = colors[results[0]['labels'].index(label)]
        
        # Draw rectangle
        draw.rectangle(box, outline=color, width=3)
        
        # Draw label with score
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1]-20), label_text, fill=color)
    
    return image

def detect(image, model, processor, device = "cuda"):
    
    text = "car . truck . bus . motorcycle . bicycle . tricycle . van . suv . trailer . construction vehicle . moped . recreational vehicle . pedestrian . cyclist . wheelchair . stroller . traffic light . traffic sign . traffic cone . traffic island . traffic box . barrier . bollard . warning sign . debris . machinery . dustbin . concrete block . cart . chair . basket . suitcase . dog . phone booth ."
    
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    
    return results
    
if __name__ == "__main__":
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    image_path = "sample_3.jpg"
    output_path = "sample_3_detected.jpg"
   
    # Load image
    image = Image.open(image_path)
   
    # Get detections
    results = detect(image, model, processor, device)
    print(results)
   
    # Draw boxes and save
    annotated_image = draw_boxes(image, results)
    annotated_image.save(output_path)