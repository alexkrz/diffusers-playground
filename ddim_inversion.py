# %%
#  Setup
from io import BytesIO
from pprint import pprint
from typing import cast

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tfmF
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from src.gaussian_diffusion import invert, sample


# Useful function for later
def load_image(url, size=None):
    response = requests.get(url, timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %%
# Load pipeline
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
# DDIM Sampling
# Test our sampling function by generating an image
sample(
    pipe,
    "Watercolor painting of a beach sunset",
    negative_prompt=negative_prompt,
    num_inference_steps=50,
)[0].resize((256, 256))

# %%
# DDIM Inversion

# %%
# Show input image
input_image = load_image("https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg", size=(512, 512))
input_image_prompt = "Photograph of a puppy on the grass"
input_image

# %%
# Encode with vae
with torch.no_grad():
    latent = pipe.vae.encode(tfmF.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
latent_vec = 0.18215 * latent.latent_dist.sample()


# %%
## Inversion
inverted_latents = invert(
    pipe,
    latent_vec,
    input_image_prompt,
    num_inference_steps=50,
)
print(inverted_latents.shape)

# Decode the final inverted latents
with torch.no_grad():
    img = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
print("Noisy image:")
print(img.shape)
pipe.numpy_to_pil(img)[0]

# %%
# Generate image from noisy latents
pipe(input_image_prompt, latents=inverted_latents[-1][None], num_inference_steps=50, guidance_scale=3.5).images[0]

# %%
# Generate image with custom start step
# The reason we want to be able to specify start step
start_step = 20
sample(
    pipe,
    input_image_prompt,
    start_latents=inverted_latents[-(start_step + 1)][None],
    start_step=start_step,
    num_inference_steps=50,
)[0]

# %%
# Sampling with a new prompt
start_step = 10
new_prompt = input_image_prompt.replace("puppy", "cat")
sample(
    pipe,
    new_prompt,
    start_latents=inverted_latents[-(start_step + 1)][None],
    start_step=start_step,
    num_inference_steps=50,
)[0]

# %%
