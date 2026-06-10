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
# Encode images
# Encode image_a
img_a = Image.open("data/1_a.png").resize((512, 512))
img_a_prompt = "a photo of a person"
with torch.no_grad():
    latent = pipe.vae.encode(tfmF.to_tensor(img_a).unsqueeze(0).to(device) * 2 - 1)
latent_a = 0.18215 * latent.latent_dist.sample()
print("Encoding img_a:")
inverted_latents_a = invert(
    pipe,
    latent_a,
    img_a_prompt,
    num_inference_steps=50,
)

# Encode image_b
img_b = Image.open("data/1_b.png").resize((512, 512))
img_b_prompt = "a photo of a person"
with torch.no_grad():
    latent = pipe.vae.encode(tfmF.to_tensor(img_b).unsqueeze(0).to(device) * 2 - 1)
latent_b = 0.18215 * latent.latent_dist.sample()
print("Encoding img_b:")
inverted_latents_b = invert(
    pipe,
    latent_b,
    img_b_prompt,
    num_inference_steps=50,
)

# %%
# Decode interpolated latent_vec
start_step = 25
latent_a = inverted_latents_a[-(start_step + 1)][None]
latent_b = inverted_latents_b[-(start_step + 1)][None]
latent_interp = (latent_a + latent_b) / 2

print("Decoding image:")
sample(
    pipe,
    img_a_prompt,
    start_latents=latent_interp,
    start_step=start_step,
    num_inference_steps=50,
)[0]

# %%
