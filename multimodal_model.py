from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, AutoModelForCausalLM
from PIL import Image
import torch

EXHAUSTIVE_PROMPT_TEMPLATE = """Please describe every object you see in this image. List all objects, including small ones and background objects. Mention the color of each object if it is visible. Organize the answer as:
1. A main descriptive sentence listing all objects.
2. Group by location (e.g., table, wall, background).
3. Include the color of each object right after its name.
Do not limit the answer to one or two objects."""
FAST_DESCRIPTION_PROMPT = (
    "Briefly describe the main subject and key visible details in one short sentence."
)

FALLBACK_MODEL_ID = "Salesforce/blip-vqa-base"
DESCRIPTION_TRIGGERS = [
    "in front of me",
    "what do you see",
    "what is in this image",
    "what is in this picture",
    "what is in this photo",
    "what is in this clip",
    "describe this image",
    "describe this picture",
    "describe this photo",
    "describe this clip",
]


def _load_blip_fallback(device: str):
    processor = BlipProcessor.from_pretrained(FALLBACK_MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(FALLBACK_MODEL_ID).to(device)
    return model, processor


def should_use_descriptive_prompt(question: str) -> bool:
    text_lower = (question or "").lower()
    return any(trigger in text_lower for trigger in DESCRIPTION_TRIGGERS)


def load_model(model_name: str, device: str):
    """
    Loads LLaVA, BLIP, or Moondream models dynamically.
    """
    print(f"Loading {model_name} on {device}...")
    if "llava" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
    elif "moondream" in model_name.lower():
        try:
            revision = "2024-08-26"
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, revision=revision
            )
            # Some PhiConfig variants miss pad_token_id; provide it before model init.
            if not hasattr(config, "pad_token_id"):
                setattr(config, "pad_token_id", getattr(config, "eos_token_id", 0))
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                revision=revision,
                config=config,
            ).to(device)
            # Some Moondream/Phi builds miss pad_token_id with newer Transformers.
            eos_token_id = getattr(model.config, "eos_token_id", 0)
            try:
                current_pad = getattr(model.config, "pad_token_id", None)
            except Exception:
                current_pad = None
            if current_pad is None:
                setattr(model.config, "pad_token_id", eos_token_id)
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.pad_token_id = eos_token_id
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception as err:
            if "pad_token_id" in str(err):
                raise RuntimeError(
                    "Moondream failed due to a Transformers compatibility issue "
                    "(missing pad_token_id). Reinstall dependencies from requirements.txt "
                    "to use Moondream quality mode."
                ) from err
            raise
        
    else:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
        
    return model, processor

def answer_question(
    image_path: str,
    question: str,
    model,
    processor,
    device: str,
    response_mode: str = "fast",
) -> dict:
    """
    Processes the image and text question through the multimodal architecture.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        text_lower = question.lower()
        model_name = model.config.name_or_path.lower() if hasattr(model.config, 'name_or_path') else "model"
        use_descriptive_prompt = should_use_descriptive_prompt(question)
        
        if "llava" in model_name:
            if use_descriptive_prompt:
                user_prompt = (
                    EXHAUSTIVE_PROMPT_TEMPLATE
                    if response_mode == "detailed"
                    else FAST_DESCRIPTION_PROMPT
                )
            else:
                user_prompt = question
                
            prompt_text = f"USER: <image>\n{user_prompt}\nASSISTANT:"
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = {k: v.to(device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            max_tokens = 220 if response_mode == "detailed" else 90
            generate_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            input_length = inputs['input_ids'].shape[1]
            output_tokens = generate_ids[0][input_length:]
            answer = processor.decode(output_tokens, skip_special_tokens=True).strip()
            reasoning = f"Processed via {model_name} 7-Billion Parameter Visual-Text Decoder."
            
        elif "moondream" in model_name:
            if use_descriptive_prompt:
                user_prompt = (
                    EXHAUSTIVE_PROMPT_TEMPLATE
                    if response_mode == "detailed"
                    else FAST_DESCRIPTION_PROMPT
                )
            else:
                user_prompt = question
                
            enc_image = model.encode_image(image)
            answer = model.answer_question(enc_image, user_prompt, processor)
            reasoning = f"Processed via {model_name} 1.8-Billion Parameter Small-Vision-Language Model."

        else:
            # BLIP handling
            if use_descriptive_prompt:
                user_prompt = (
                    "describe the image with visible people and clothing"
                    if response_mode == "detailed"
                    else "what is shown in this image?"
                )
            else:
                user_prompt = question

            inputs = processor(image, user_prompt, return_tensors="pt").to(device)
            max_tokens = 60 if response_mode == "detailed" else 28
            out = model.generate(**inputs, max_new_tokens=max_tokens, num_beams=3)
            answer = processor.decode(out[0], skip_special_tokens=True).strip()
            reasoning = (
                f"Processed via {model_name} base VQA fallback. "
                "For richer clothing/color details, use Moondream."
            )

        return {
            "answer": answer,
            "reasoning": reasoning,
            "model_used": model_name
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": "I'm sorry, I encountered an error analyzing the image.",
            "reasoning": f"Exception raised: {str(e)}",
            "model_used": str(model.__class__.__name__)
        }

if __name__ == "__main__":
    print("multimodal_model.py loaded successfully.")
