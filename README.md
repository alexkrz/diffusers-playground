# diffusers-playground

This repository uses the latest version of diffusers to try different tutorials from the diffusers homepage.

## Setup

We recommend [miniforge](https://conda-forge.org/download/) to set up your python environment. \
Then [uv](https://docs.astral.sh/uv/) can be used to install the project dependencies:

```bash
conda create -n $YOUR_ENV_NAME python=3.12
conda activate $YOUR_ENV_NAME
uv pip install -r requirements.txt
pre-commit install
```

## Todos

- [ ] Add interpolate script that uses a ddpm model instead of an ldm model
