import os
import csv
from config import MODEL_ID, DEVICE, DATASET_EVAL_CSV, DATASET_IMAGES_DIR
from multimodal_model import load_model, answer_question

def evaluate_model():
    """
    Evaluates the loaded VQA model against the provided dataset (data_eval.csv).
    Computes Exact Match (EM) accuracy on the evaluation set.
    """
    if not os.path.exists(DATASET_EVAL_CSV):
        print(f"Error: Evaluation CSV not found at {DATASET_EVAL_CSV}")
        return
        
    if not os.path.exists(DATASET_IMAGES_DIR):
        print(f"Error: Evaluation Images DIR not found at {DATASET_IMAGES_DIR}")
        return

    print("Loading AI Model for Dataset Evaluation...")
    model, processor = load_model(MODEL_ID, DEVICE)
    print("Model loaded successfully. Starting evaluation...\n")
    
    total_samples = 0
    correct_matches = 0
    
    with open(DATASET_EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # To avoid incredibly long eval times on CPU demo, limit if desired
            if idx > 100:  # Comment out this line to evaluate the entire dataset
                break 
                
            question = row.get("question", "").strip()
            ground_truth = row.get("answer", "").strip().lower()
            image_id = row.get("image_id", "").strip()
            
            # Form image path based on dataset structure
            image_path = os.path.join(DATASET_IMAGES_DIR, f"{image_id}.png")
            if not os.path.exists(image_path):
                # Fallback to jpg if png not found
                image_path = os.path.join(DATASET_IMAGES_DIR, f"{image_id}.jpg")
                
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_id} not found. Skipping...")
                continue
                
            total_samples += 1
            result = answer_question(image_path, question, model, processor, DEVICE)
            model_answer = result["answer"].strip().lower()
            
            # Exact Match accuracy check
            if model_answer == ground_truth:
                correct_matches += 1
                match_status = "✅ MATCH"
            else:
                match_status = f"❌ MISS (Expected: '{ground_truth}', Got: '{model_answer}')"
                
            print(f"[{total_samples}] Q: {question}")
            print(f"    {match_status}")
            
    if total_samples > 0:
        accuracy = (correct_matches / total_samples) * 100
        print(f"\n--- Evaluation Complete ---")
        print(f"Total Evaluated: {total_samples}")
        print(f"Exact Matches: {correct_matches}")
        print(f"Overall Accuracy (EM): {accuracy:.2f}%")
    else:
        print("No valid samples evaluated.")

if __name__ == "__main__":
    evaluate_model()
