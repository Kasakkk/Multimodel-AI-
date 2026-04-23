import json
import datetime
import os
from config import MODEL_ID, DEVICE, LOG_FILE_PATH, IMAGE_SAVE_PATH
from camera import capture_image
from multimodal_model import load_model, answer_question

# Pre-load model and processor during startup so interactive REPL remains fast
print("Initializing AI Assistant...")
model, processor = load_model(MODEL_ID, DEVICE)

def log_interaction(image_path: str, question: str, answer: str, reasoning: str, model_used: str):
    """
    Logs every interaction into a JSON file for local audit or debugging.
    """
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "model_used": model_used
    }
    
    # Read existing logs if file exists
    logs = []
    if os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            pass # Start fresh if file is corrupted
            
    logs.append(entry)
    
    with open(LOG_FILE_PATH, "w") as f:
        json.dump(logs, f, indent=4)

def run_assistant_simple():
    """
    A simple command-line REPL for the assistant prioritizing iterative prompting.
    """
    print("\nAssistant started. Type 'quit' or 'exit' to stop.")
    while True:
        question = input("\nYou: ")
        if question.lower() in ["quit", "exit"]:
            break
            
        print("Assistant: Processing your request...")
        
        # Decide if we need to take a fresh picture based on natural language clues
        intent_keywords = ["in front of me", "what do you see", "describe", "look at"]
        takes_picture = any(kw in question.lower() for kw in intent_keywords)
        
        try:
            if takes_picture:
                img_path = capture_image(IMAGE_SAVE_PATH)
                print(f"(System: Captured image to {img_path})")
            else:
                # Use previously captured picture if a follow-up question is asked
                img_path = IMAGE_SAVE_PATH
                if not os.path.exists(img_path):
                    print("AI: I don't have an image to look at. Please ask me to look at what's in front of you first.")
                    continue
        except RuntimeError as e:
            print(f"AI: Camera Error - {str(e)}")
            continue
            
        # Core dispatch call to underlying model weights
        result = answer_question(img_path, question, model, processor, DEVICE)
        
        print(f"\nAI: {result['answer']}")
        print(f"[Reasoning: {result['reasoning']}]")
        
        log_interaction(img_path, question, result["answer"], result["reasoning"], result["model_used"])

if __name__ == "__main__":
    run_assistant_simple()
