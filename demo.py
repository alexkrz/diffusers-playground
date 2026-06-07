import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image: Image.Image = pipe(prompt).images[0]

image.save("results/astronaut_rides_horse.png")
