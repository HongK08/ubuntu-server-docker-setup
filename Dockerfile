FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git curl wget ca-certificates \
    build-essential pkg-config \
    cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

WORKDIR /workspace

# PyTorch (CUDA 13)
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Core LLM stack
RUN python3 -m pip install --no-cache-dir \
    transformers datasets accelerate tokenizers safetensors \
    peft trl bitsandbytes optimum \
    fastapi uvicorn[standard] \
    wandb mlflow hydra-core omegaconf optuna \
    einops torchmetrics

CMD ["/bin/bash"]
