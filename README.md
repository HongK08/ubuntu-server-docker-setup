# ubuntu-server-docker-setup

## Overview | 개요

This repository provides a clean and reproducible setup for running:

- Ollama on the host system for inference and serving
- PyTorch + LLM training environment inside Docker containers

본 레포지토리는 다음 구조로 GPU 서버를 운영하기 위한 설정을 제공합니다:

- Host 시스템에서 Ollama를 이용한 추론 및 서빙
- Docker 컨테이너 내부에서 PyTorch 기반 학습 및 파인튜닝 환경

This design prevents environment conflicts when multiple users share the same GPU server.

이 구조는 여러 사용자가 동일한 GPU 서버를 사용할 때 환경 충돌을 방지하는 것을 목표로 합니다.

---

## Tested Environment | 테스트 환경

- Ubuntu 24.04
- NVIDIA Driver 590
- CUDA 13.1
- cuDNN 9.x
- RTX 4090 + RTX 5000 Ada

---

# 0. Host Requirements | 호스트 요구사항

## Check GPU | GPU 확인

```bash
nvidia-smi
```

## Verify Docker GPU Runtime | Docker GPU 동작 확인

```bash
sudo docker run --rm --gpus all nvidia/cuda:13.1.0-base nvidia-smi
```

If GPUs are visible, Docker and NVIDIA runtime are properly configured.

GPU가 정상적으로 보이면 Docker와 NVIDIA runtime이 올바르게 설정된 상태입니다.

---

# 1. Ollama (Host) – Inference / Serving

## Installation | 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl status ollama --no-pager
```

## Allow External Access (Optional) | 외부 접근 허용 (선택)

```bash
sudo systemctl edit ollama
```

Add the following:

다음 내용을 추가합니다:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Apply changes:

적용:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

---

# 2. Clone Repository | 레포지토리 클론

```bash
git clone https://github.com/YOUR_USERNAME/ubuntu-server-docker-setup.git
cd ubuntu-server-docker-setup
```

---

# 3. Build Training Image | 학습용 이미지 빌드

```bash
docker build -t hai-train:cu13.1 .
```

Note: Ubuntu 24.04 with Python 3.12 enforces PEP 668.  
The Dockerfile uses a virtual environment inside the container to avoid system package conflicts.

참고: Ubuntu 24.04 (Python 3.12)에서는 PEP 668 정책이 적용됩니다.  
Dockerfile은 컨테이너 내부에서 virtual environment를 사용하여 시스템 패키지 충돌을 방지합니다.

---

# 4. Run Container | 컨테이너 실행

## Standard Execution | 기본 실행

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace -w /workspace \
  -v "$PWD/.cache":/workspace/.cache \
  -p 8000:8000 -p 7860:7860 \
  hai-train:cu13.1
```

## Access Ollama from Container | 컨테이너에서 Ollama 접근

```bash
docker run --rm -it --gpus all --network host \
  -v "$PWD":/workspace -w /workspace \
  hai-train:cu13.1
```

Inside the container, Ollama API is available at:

컨테이너 내부에서 Ollama API는 다음 주소로 접근 가능합니다:

```
http://127.0.0.1:11434
```

---

# 5. Sanity Check | 환경 확인

Run inside the container:

컨테이너 내부에서 실행:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
print("cudnn:", torch.backends.cudnn.version())
PY
```

---

# 6. Multi-GPU Usage Recommendation | 멀티 GPU 사용 권장 구조

Recommended separation:

권장 구성:

- GPU 0 → Training
- GPU 1 → Ollama serving

Example:

예시:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

---

# 7. Docker Cleanup | Docker 정리

Remove exited containers:

종료된 컨테이너 정리:

```bash
docker container prune -f
```

Remove unused images:

사용하지 않는 이미지 정리:

```bash
docker image prune -f
```

---

# 8. Public Repository Safety | 퍼블릭 레포지토리 보안 주의사항

This repository is public.

본 레포지토리는 Public입니다.

Do NOT upload:

업로드 금지 항목:

- .env files
- API keys
- HuggingFace tokens
- Model weights
- Private datasets
- SSH keys

Recommended `.gitignore`:

권장 `.gitignore`:

```
.env
.cache/
__pycache__/
*.pt
*.bin
*.safetensors
wandb/
mlruns/
```

---

# License | 라이선스

MIT
