import torch
import os
import streamlit as st
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
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=access_token
    )
    pipe.to(device)
    return pipe

inpainting_pipeline = load_inpainting_model()

def generate_image_from_image(image, mask, prompt):
    return inpainting_pipeline(prompt=prompt, image=image, mask_image=mask).images[0]
