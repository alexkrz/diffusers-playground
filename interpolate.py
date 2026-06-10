# %%
# Imports
from pprint import pprint

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

# %%
# Load pipeline
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
print(pipe.device)
pprint(dict(pipe.config), sort_dicts=False)

# %%
# Test pipeline
# Set up a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
assert isinstance(pipe.scheduler, DDIMScheduler)

# Sample an image to make sure it is all working
prompt = "Beautiful DSLR Photograph of a penguin on the beach, golden hour"
negative_prompt = "blurry, ugly, stock photo"
img: Image.Image = pipe(prompt, negative_prompt=negative_prompt).images[0]
# print(img)  # Returned image is of size 512x512 px
img.resize((256, 256))  # Resize for convenient viewing

# %%
