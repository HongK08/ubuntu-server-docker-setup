FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    git curl wget ca-certificates \
    build-essential pkg-config \
    cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# venv 생성 + pip 최신화 (이제 PEP668 안 걸림)
RUN python3 -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install -U pip setuptools wheel

WORKDIR /workspace

# PyTorch (CUDA 13)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Core LLM stack
RUN pip install --no-cache-dir \
    transformers datasets accelerate tokenizers safetensors \
    peft trl bitsandbytes optimum \
    fastapi uvicorn[standard] \
    wandb mlflow hydra-core omegaconf optuna \
    einops torchmetrics

CMD ["/bin/bash"]
