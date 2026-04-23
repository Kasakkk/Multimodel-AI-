# Multi-Modal AI Assistant (VQA)

## Project Goal
A multi-modal AI assistant explicitly built to natively answer questions about live visual surroundings as well as systematically evaluate against evaluation sets. A user can specifically ask prompts resembling "What is in front of me?", and the assistant captures an image, reasoning physically over the environment.

## Datasets and Backing Philosophy
This system leverages pre-trained Vision-Language models derived from visual understanding datasets. It includes a direct pipeline to test and visualize the **Kaggle Visual Question Answering - Computer Vision & NLP** competition dataset physically evaluating against local `/dataset` CSVs zero-shot context using Hugging Face pipelines (`Salesforce/blip-vqa-base`).

## Directory Structure
- `config.py`: Primary environment vars, system targets, and dataset pathing mappings.
- `camera.py`: Webcam access layer via `OpenCV`.
- `multimodal_model.py`: Core logic for Hugging Face BLIP-VQA loading, prep, and pipeline execution.
- `evaluate.py`: Standalone CLI utility simulating dataset runs over `data_eval.csv` automatically computing EM accuracy.
- `assistant.py`: CLI-based interface for interacting naturally capturing webcam pictures dynamically.
- `streamlit_app.py`: Main Streamlit website with Live Assistant and Dataset Evaluator tabs.
- `app.py`: Compatibility launcher that starts the Streamlit website.
- `logs.json`: Auto-generated upon run; safely persists interaction history.

## Setup and Run Instructions

### Step 1: Install Dependencies
Ensure you are using Python 3.10+, and make a virtual environment before installing:
```bash
python -m venv venv
# On Windows use: venv\Scripts\activate
# On Mac/Linux use: source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Running the System
Choose the interface option you prefer:

**1. Web UI Dashboard (Recommended):**
```bash
streamlit run streamlit_app.py
```
*Wait for model load, then navigate to the local URL shown in the terminal (usually `http://localhost:8501`). Includes dataset explorer and optional voice input.*

Alternative launcher:
```bash
python app.py
```
*This command launches the same Streamlit app for backward compatibility.*

**2. Terminal Live Assistant Version:**
```bash
python assistant.py
```
*Wait for model initialization, then interact within the terminal window natively typing "What do you see?"*

**3. Dataset Evaluation Script:**
```bash
python evaluate.py
```
*Computes validation loss/accuracy across the `data_eval.csv` ground truth lines directly reporting metrics iteratively to console.*

> **IMPORTANT Note on Resources:** The default model `Salesforce/blip-vqa-base` is highly compatible with general RAM constraints while being specifically optimized towards answering standard VQA context seamlessly.
