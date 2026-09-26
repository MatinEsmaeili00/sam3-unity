# SAM 3 live-stream segmentation server for Unity / Meta Quest 3
#
# This image contains ONLY: (1) this repo's own server code (MIT-licensed,
# see LICENSE), and (2) Meta's official SAM 3 inference code, fetched fresh
# at build time from https://github.com/facebookresearch/sam3 (pinned to a
# specific commit for reproducibility).
#
# The model CHECKPOINT is NOT included. SAM 3's weights are gated on Hugging
# Face under Meta's "SAM License" - you accept that license with your own
# account, and the container downloads the checkpoint at *runtime* using your
# mounted Hugging Face credentials/cache (see docker-compose.yml, README.md).
FROM python:3.12-slim

LABEL org.opencontainers.image.title="sam3-unity" \
      org.opencontainers.image.description="SAM 3 WebSocket segmentation server for streaming from Unity / Meta Quest 3" \
      org.opencontainers.image.version="1.0.0"

# git: to fetch the official SAM3 source; build-essential/python3-dev: SAM3's
# NMS uses a Triton kernel that JIT-compiles a small C extension at runtime;
# libgl1/libglib2.0-0: required by opencv-python-headless at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 13.0. Chosen for an NVIDIA GB10 (Blackwell, sm_121):
# torch's cu126/cu128 wheels have no sm_121 kernels and fail with "no kernel
# image is available for execution on the device". On another GPU, pick the
# matching index from https://pytorch.org/get-started/locally/.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Meta's official SAM3 inference code (pinned). Not pip-installed - only its
# declared dependencies are installed below, and it is imported via PYTHONPATH.
ARG SAM3_COMMIT=660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && cd /opt/sam3 && git checkout ${SAM3_COMMIT}
ENV PYTHONPATH="/opt/sam3"

# SAM3's own dependencies (pyproject.toml [project.dependencies] + "notebooks"
# extra), plus scikit-image for the test client's bundled sample image.
RUN pip install --no-cache-dir \
    "timm>=1.0.17" "numpy>=1.26,<2" tqdm "ftfy==6.1.1" regex "iopath>=0.1.10" \
    typing_extensions huggingface_hub einops pycocotools psutil pandas \
    opencv-python-headless matplotlib pillow imageio scikit-image scikit-learn

# WebSocket server.
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websockets

COPY src/ ./src/
ENV MPLBACKEND=Agg
EXPOSE 8765

ENTRYPOINT ["python", "src/stream_server.py"]
CMD ["--port", "8765"]
