import torch
import os
from dotenv import load_dotenv
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Load API Key
load_dotenv()
access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_inpainting_model():
    """ Load and optimize the image-to-image model. """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",  # Faster & optimized inpainting model
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=access_token
    )
    pipe.to(device)

    # Optimizations
    pipe.enable_attention_slicing()
    if device == "cuda":
        pipe.unet = torch.compile(pipe.unet)

    return pipe

inpainting_pipeline = load_inpainting_model()

def generate_image_from_image(image, mask, prompt):
    """ Generate a modified image using a mask and prompt. """
    return inpainting_pipeline(prompt=prompt, image=image, mask_image=mask, num_inference_steps=25).images[0]
