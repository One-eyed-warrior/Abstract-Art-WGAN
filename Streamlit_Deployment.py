import streamlit as st
import torch
import numpy as np
from PIL import Image
import torch.nn as nn

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


device = torch.device('cpu')  # because cuda gpu apparently doesn't work with streamlit, eh what can you do
generator = Generator()
generator.load_state_dict(torch.load('G.pth', map_location=device))
generator.eval()

st.set_page_config(page_title="Image Generator", layout="wide")

st.title("Abstract Art Generator")
st.write("Press the button below to generate an image.")

if st.button("Generate Image"):
   
    noise = torch.randn(1, 100, 1, 1) 

    
    with torch.no_grad():  
        generated_image = generator(noise)

   
    generated_image = generated_image.squeeze().permute(1, 2, 0)  
    generated_image = (generated_image + 1) / 2.0  
    generated_image = (generated_image * 255).numpy().astype(np.uint8)  

    
    pil_image = Image.fromarray(generated_image)
    display_width = 150 
    st.image(pil_image, width=display_width)