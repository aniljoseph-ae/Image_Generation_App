import streamlit as st
from models.text_to_image import generate_image_from_text
from models.image_to_image import generate_image_from_image
from PIL import Image
import torch
import asyncio

# Set Streamlit page configuration
st.set_page_config(page_title="AI Image Generator", layout="wide")

# Sidebar for Navigation
st.sidebar.title("AI Image Generator")
choice = st.sidebar.selectbox("Select Image Generation Method", ["Home", "Text-to-Image", "Image+Text→Image"])

if choice == "Home":
    st.title("AI Image Generator App")
    st.write("Generate AI-generated images from text prompts or modify images using AI.")

elif choice == "Text-to-Image":
    st.subheader("Generate Images from Text")
    prompt = st.text_input("Enter your text prompt")
    if prompt and st.button("Generate Image"):
        with st.spinner("Generating image... Please wait."):
            image = generate_image_from_text(prompt)
            st.image(image, caption="Generated Image", use_column_width=True)

elif choice == "Image+Text→Image":
    st.subheader("Modify Image using Text")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    uploaded_mask = st.file_uploader("Upload a mask (white for areas to modify)", type=["png", "jpg", "jpeg"])
    prompt = st.text_input("Enter your modification text")

    if uploaded_image and uploaded_mask and prompt and st.button("Generate Modified Image"):
        with st.spinner("Processing... Please wait."):
            image = Image.open(uploaded_image).convert("RGB")
            mask = Image.open(uploaded_mask).convert("L")
            modified_image = generate_image_from_image(image, mask, prompt)
            st.image(modified_image, caption="Modified Image", use_column_width=True)
