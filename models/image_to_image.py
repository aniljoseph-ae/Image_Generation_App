import torch
import os
from dotenv import load_dotenv
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Load API Key
# load_dotenv()
# access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "").strip()
if not access_token:
    raise ValueError("HUGGINGFACE_ACCESS_TOKEN is missing! Please set it as an environment variable.")


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_inpainting_model():
    '''
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=access_token
    )
    
    '''
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=access_token  # Replace use_auth_token with token
)

    
    pipe.to(device)
    return pipe

inpainting_pipeline = load_inpainting_model()

def generate_image_from_image(image, mask, prompt):
    image = inpainting_pipeline(prompt=prompt, image=image, mask_image=mask).images[0]
    return image
