# %%
# Imports
from pprint import pprint

import torch
import torchvision.transforms.functional as tfmF
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from src.gaussian_diffusion import invert, sample


def slerp(latent_a: torch.Tensor, latent_b: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical interpolation in latent space with linear fallback for near-colinear vectors."""
    a_flat = latent_a.flatten(start_dim=1)
    b_flat = latent_b.flatten(start_dim=1)

    a_norm = torch.norm(a_flat, dim=1, keepdim=True).clamp_min(eps)
    b_norm = torch.norm(b_flat, dim=1, keepdim=True).clamp_min(eps)

    a_unit = a_flat / a_norm
    b_unit = b_flat / b_norm

    dot = (a_unit * b_unit).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # If angle is tiny, SLERP is numerically unstable and equivalent to LERP.
    near_linear = sin_omega.abs() < eps
    t_tensor = torch.full_like(dot, float(t))
    s0 = torch.sin((1 - t_tensor) * omega) / sin_omega
    s1 = torch.sin(t_tensor * omega) / sin_omega
    interp_flat = s0 * a_flat + s1 * b_flat

    lerp_flat = (1 - t_tensor) * a_flat + t_tensor * b_flat
    interp_flat = torch.where(near_linear, lerp_flat, interp_flat)

    return interp_flat.view_as(latent_a)


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
mix = 0.5
latent_interp = slerp(latent_a, latent_b, t=mix)

print("Decoding image:")
sample(
    pipe,
    img_a_prompt,
    start_latents=latent_interp,
    start_step=start_step,
    num_inference_steps=50,
)[0]

# %%
