# ubuntu-server-docker-setup

GPU 서버(우분투 24.04)에서 **Ollama(호스트)** 로 추론/서빙을 하고,  
**Docker(컨테이너)** 로 학습/파인튜닝 환경을 재현 가능하게 관리하는 레포입니다.

- Host: Ollama (inference / serving)
- Container: PyTorch + LLM training stack (reproducible)

> Tested: Ubuntu 24.04, NVIDIA Driver 590, CUDA 13.1, cuDNN 9.x, RTX 4090 + RTX 5000 Ada

---

## 0) Prerequisites (Host)

- NVIDIA driver installed (`nvidia-smi` 정상)
- Docker installed
- NVIDIA Container Toolkit installed

GPU passthrough 테스트:

```bash
sudo docker run --rm --gpus all nvidia/cuda:13.1.0-base nvidia-smi


<img width="909" height="1035" alt="image" src="https://github.com/user-attachments/assets/c02d55c2-9303-4d6d-8905-e4e771f9092b" />
