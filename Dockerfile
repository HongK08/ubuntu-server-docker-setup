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

# venv 안에서 pip 사용
RUN python3 -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install -U pip setuptools wheel

WORKDIR /workspace

# PyTorch (CUDA 13)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu130

# LLM/ML stack (필요한거 더 추가 가능함.)
RUN pip install --no-cache-dir \
    transformers datasets accelerate tokenizers safetensors sentencepiece \
    peft trl bitsandbytes optimum \
    fastapi uvicorn[standard] gradio \
    wandb mlflow hydra-core omegaconf optuna \
    einops torchmetrics

CMD ["/bin/bash"]
