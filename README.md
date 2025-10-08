# xpc-autodiff-recon

Automatic differentiation (AD) reconstruction for X-ray phase-contrast (XPC) imaging. Supports both tomographic and projection geometries and a variety of forward models, enabled by JAX and Chromatix.

## Installation

Clone the git repo to your workspace: 

```bash
git clone https://github.com/gjadick/xpc-autodiff-recon.git
```

## Environment Setup

I generally advise using `uv` to speed up package installation, so these instructions are based on that. Create a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

For JAX, install individually first to ensure you get the accelerated version compatible with your hardware. For example, if you have NVIDIA GPUs and CUDA12:

```bash
uv pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
(If you lack GPUs, you can still use the CPU-only version of JAX with `uv pip install -U "jax[cpu]"` instead.)


Then install project dependencies. For scientific projects, so you can use the Jupyter notebooks and plotting, that is:

```bash
uv pip install -e ".[sci]"
```

Finally, install chromatix for AD-compatible XPC forward modeling:

```bash
pip install git+https://github.com/chromatix-team/chromatix.git
```

## Usage

In your own projects, just `import xpc`. A variety of example projects and conference/paper data are in the base directory.

