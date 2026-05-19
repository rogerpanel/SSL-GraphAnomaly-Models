# Reproducible CUDA / PyTorch image for SSL-GraphAnomaly.
# Build:  docker build -t ssl-graphanomaly:latest -f docker/Dockerfile .
# Run  :  docker run --rm --gpus all -it -p 8014:8014 \
#             -v $PWD/data:/workspace/data -v $PWD/outputs:/workspace/outputs \
#             ssl-graphanomaly:latest

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# System-level deps (build tools needed by torch-scatter / torch-sparse wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache friendly: copy requirements first, install, then copy code.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip wheel \
    && pip install -r /workspace/requirements.txt

# Copy the rest of the source tree and install in editable mode.
COPY . /workspace
RUN pip install -e .

EXPOSE 8014

CMD ["python", "-m", "ssl_graph_anomaly.inference.serve"]
