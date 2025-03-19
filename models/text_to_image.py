import torch
import os
import streamlit as st
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline

# Load API Key
load_dotenv()
access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_text_to_image_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=access_token  # Fix auth token usage
    )
    pipe.to(device)
    pipe.enable_attention_slicing()  # Optimizes VRAM usage
    return pipe

text_to_image_pipeline = load_text_to_image_model()

def generate_image_from_text(prompt):
    return text_to_image_pipeline(prompt).images[0]
