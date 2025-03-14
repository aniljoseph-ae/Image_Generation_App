# Multi-Modal AI Image Generator

## Overview
The **Multi-Modal AI Image Generator** is a Streamlit-based application that utilizes the **Stable Diffusion** model from Hugging Face to generate and modify images using AI. The app supports two types of input:

1. **Text-to-Image**: Generates images based on textual descriptions.
2. **Image+Text→Image (Inpainting)**: Modifies an existing image using a textual description and a mask.

## Features
- Generate AI-generated images from text prompts.
- Modify existing images using textual input and a mask.
- Utilizes **Stable Diffusion v2** for high-quality image generation.
- GPU acceleration support (if available).
- Secure API authentication using Hugging Face tokens.

---

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip
- Virtual environment (optional but recommended)

### Clone the Repository
```sh
git clone https://github.com/aniljoseph-ae/Image_Generation_App.git
cd Image_Generation_App
```

### Create and Activate Virtual Environment (Optional)
```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the project root and add your Hugging Face API key:
```sh
HUGGINGFACE_ACCESS_TOKEN=your_huggingface_api_token
```

---

## Usage
### Running the Application
```sh
streamlit run app.py
```

### Interface Overview
- **Home Page**: Displays an introduction to the application.
- **Text-to-Image**: Enter a text prompt to generate an image.
- **Image+Text→Image**: Upload an image and a mask, then enter a text prompt to modify the image accordingly.

---

## Project Structure
```
├── models
│   ├── text_to_image.py      # Handles text-to-image generation using Stable Diffusion
│   ├── image_to_image.py     # Handles image inpainting using Stable Diffusion Inpainting
│
├── .env                      # Stores Hugging Face API token
├── app.py                    # Streamlit application
├── requirements.txt          # Required dependencies
├── README.md                 # Documentation
```

---

## Code Explanation
### `app.py`
This is the main application file built with **Streamlit**. It provides:
- A sidebar for selecting the image generation method.
- A user interface for text-to-image and image+text→image functionalities.
- Calls to model functions for generating images.

### `text_to_image.py`
- Loads the **Stable Diffusion v2** model from Hugging Face.
- Uses a **text prompt** to generate a high-quality image.
- Supports GPU acceleration.

#### Function:
```python
def generate_image_from_text(prompt):
    image = text_to_image_pipeline(prompt).images[0]
    return image
```

### `image_to_image.py`
- Loads the **Stable Diffusion Inpainting** model.
- Uses an **input image, mask, and text prompt** to modify an image.
- Supports GPU acceleration.

#### Function:
```python
def generate_image_from_image(image, mask, prompt):
    image = inpainting_pipeline(prompt=prompt, image=image, mask_image=mask).images[0]
    return image
```

---

## Dependencies
The required packages are listed in `requirements.txt`. To install them, run:
```sh
pip install -r requirements.txt
```
### Key Libraries Used:
- `streamlit`: Web UI framework.
- `diffusers`: Hugging Face library for Stable Diffusion.
- `torch`: PyTorch for deep learning.
- `PIL`: Image processing.
- `dotenv`: Managing API keys securely.

---

## Future Enhancements
- Support for additional models like **SDXL** or **ControlNet**.
- Advanced image editing features (e.g., object removal, background replacement).
- User authentication for managing image history.

---

## License
This project is open-source under the **MIT License**.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

### How to Contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Added new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Submit a pull request.

---

## Contact
For queries or suggestions, feel free to reach out at:
📧 Email: aniljoseph.ae@gmail.com  
