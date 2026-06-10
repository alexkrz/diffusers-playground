# %%
# Imports
from pprint import pprint

import torch
import torchvision.transforms.functional as tfmF
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from src.gaussian_diffusion import invert, sample

# %%
# Load pipeline
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Set up a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
assert isinstance(pipe.scheduler, DDIMScheduler)

# Print pipe setup
print(pipe.device)
pprint(dict(pipe.config), sort_dicts=False)

# %%
# Encode and decode image
input_image = Image.open("data/1_a.png").resize((512, 512))
input_image_prompt = "a photo of a person"
with torch.no_grad():
    latent = pipe.vae.encode(tfmF.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
latent_vec = 0.18215 * latent.latent_dist.sample()
print("Encoding image:")
inverted_latents = invert(
    pipe,
    latent_vec,
    input_image_prompt,
    num_inference_steps=50,
)
print(inverted_latents.shape)
print("Decoding image:")
start_step = 20
sample(
    pipe,
    input_image_prompt,
    start_latents=inverted_latents[-(start_step + 1)][None],
    start_step=start_step,
    num_inference_steps=50,
)[0]

# %%
