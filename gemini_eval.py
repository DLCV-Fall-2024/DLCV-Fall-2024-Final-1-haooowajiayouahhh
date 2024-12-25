import os
import time
import json
import random
import string
import argparse
import numpy as np
from collections import defaultdict
from datasets import load_dataset
import google.generativeai as genai
from scorer import Scorer


def format_prompt(few_shot_path, message, sample_type="general"):
    template = "Question:\n[The Start of Reference Text]\n{}\n[The End of Reference Text]\n\n[The Start of Prediction Text]\n{}\n[The End of Prediction Text]"
    messages = ""
    if sample_type in ["general", "suggestion"]:
        folder = "scene_few_shot" if sample_type == "general" else "suggestion_few_shot"
        for sample in os.listdir(os.path.join(few_shot_path, folder)):
            data = json.load(open(os.path.join(few_shot_path, folder, sample)))
            messages += template.format(data["reference"], data["prediction"]) + "\n"
            messages += "Answer:\n" + data["response"] + "\n\n"     
    messages += template.format(message["reference"], message["prediction"]) + "\nAnswer:\n"
    
    return messages

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_output_tokens", type=int, default=300)
    parser.add_argument("--few_shot", type=str, default="few_shot")
    parser.add_argument("--prediction", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="ntudlcv/dlcv_2024_final1")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--save", type=str, default="gemini_eval.json")
    parser.add_argument("--api_key", type=str, default="AIzaSyBKgbVZR8Uv3GkLSkL4nAOLLbz53KG9uAs")
    return parser.parse_args()


if __name__ == "__main__":

    args = arguments()

    # Load model
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        system_instruction="You are an impartial judge tasked with evaluating text similarity and relevance of the reference text and autonomous driving AI assistant's predicted text. Be as objective as possible. Do not allow the length of the predicted text to influence your evaluation. After providing your short explanation, you must rate on a scale from 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[10]]\"."
    )
    generation_config = genai.GenerationConfig(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens
    )

    # Load dataset  
    reference = load_dataset(args.dataset_name, split=args.split, streaming=True)
    prediction = json.load(open(args.prediction, "r"))
    NLP_HYPOTHESIS = {key: [value.strip()] for key, value in prediction.items()}
    NLP_REFERENCE = {}
    
    # Evaluate
    result = defaultdict(list)
    save = {}
    # fail_cases = []
    for data in reference:
        sample_id = data["id"]
        score = 0

        if sample_id not in prediction:
            continue

        message = {"reference": "", "prediction": prediction[sample_id]}
        message["reference"] = data["conversations"][1]["value"]
        sample_type = (sample_id.split("_")[1]).lower()
        messages = format_prompt(args.few_shot, message, sample_type)
        try:
            response = model.generate_content(messages, generation_config=generation_config)
            try:
                score = int(response.text.split("[[")[-1].split("]]")[0])
            except:
                print("Missing score for sample: ", sample_id)
                # fail_cases.append(sample_id)
        except:
            print("Error for sample: ", sample_id)
            # fail_cases.append(sample_id)
        result[sample_type].append(score)

        save[sample_id] = {
            "prediction": response.text,
            "score": score
        }
        with open(args.save, "w") as f:
            json.dump(save, f, indent=4)
        time.sleep(3)
        NLP_REFERENCE[sample_id] = [data["conversations"][1]["value"]]

    coco_eval = Scorer(NLP_HYPOTHESIS, NLP_REFERENCE)
    total_scores = coco_eval.evaluate()

    print("\n\n")
    # LLM judge
    total = []
    for sample_type, scores in result.items():
        score = np.mean(scores)
        print(f"{sample_type.capitalize()} score: {score:.3f}")
        total.append(score)
    print(f"LLM judges: {np.mean(total):.3f}")
    # print("Number of fail cases: ", len(fail_cases))

    # NLP metric
    for key, value in total_scores.items():
        print(f'{key}: {value:.3f}')

    total_score = np.mean(total) * 0.8 + total_scores["Bleu_3"] * 0.2
    print(f"Total score: {total_score:.3f}") 
