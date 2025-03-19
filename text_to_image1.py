import torch
import os
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline

# Load API Key
load_dotenv()
access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_text_to_image_model():
    """ Load and optimize the text-to-image model. """
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",  # Smaller & Faster model
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=access_token
    )
    pipe.to(device)
    
    # Optimizations
    pipe.enable_attention_slicing()  # Reduce memory usage
    if device == "cuda":
        pipe.unet = torch.compile(pipe.unet)  # Speed-up with torch.compile()

    return pipe

text_to_image_pipeline = load_text_to_image_model()

def generate_image_from_text(prompt):
    """ Generate an image from text prompt. """
    return text_to_image_pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
