import torch
import os
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline

# Load API Key
load_dotenv()
access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_text_to_image_model():
    '''
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=access_token
    )
    '''
    
    pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=access_token  # Replace use_auth_token with token
)

    
    pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

text_to_image_pipeline = load_text_to_image_model()

def generate_image_from_text(prompt):
    image = text_to_image_pipeline(prompt).images[0]
    return image
