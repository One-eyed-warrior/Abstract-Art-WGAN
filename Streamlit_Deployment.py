import streamlit as st
import torch
import numpy as np
from PIL import Image
import torch.nn as nn

# Define the Generator model architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Load the trained generator model
generator = Generator()
generator.load_state_dict(torch.load('G.pth'))
generator.eval()

st.set_page_config(page_title="Image Generator", layout="wide")

st.title("Image Generator")
st.write("Press the button below to generate an image.")

if st.button("Generate Image"):
    # Generate a random noise vector and reshape it to (1, 100, 1, 1)
    noise = torch.randn(1, 100, 1, 1)  # Batch size of 1, 100 channels, height and width of 1

    # Generate an image using the loaded generator model
    with torch.no_grad():  # Disable gradient calculation for inference
        generated_image = generator(noise)

    # Convert the generated image tensor to a displayable format
    generated_image = generated_image.squeeze().permute(1, 2, 0)  # Change shape to (H, W, C)
    generated_image = (generated_image + 1) / 2.0  # Normalize to [0, 1]
    generated_image = (generated_image * 255).numpy().astype(np.uint8)  # Scale to [0, 255]

    # Create a PIL image
    pil_image = Image.fromarray(generated_image)

    # Display the generated image
    st.image(pil_image, use_column_width=True)