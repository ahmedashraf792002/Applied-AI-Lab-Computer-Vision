# meme_app/app.py
import diffusers
import torch
LORA_WEIGHTS ="onstage3890/maya_model_v1_lora"
# Set device as a string: "cuda" or "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set dtype based on GPU availability
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

def load_model():
    
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4",torch_dtype= dtype
    )
    pipeline.load_lora_weights(


    weight_name="pytorch_lora_weights.10_epochs.safetensors",
    )
    pipeline.to(device)
    return pipeline
