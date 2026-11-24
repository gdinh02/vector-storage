# CUDA 12 + cuDNN 9 runtime image (compatible with onnxruntime-gpu >= 1.19)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
    QDRANT_PATH=/data/qdrant_local \
    MODELS_CACHE=/data/models \
    STORAGE_DIR=/data/uploaded_files

# Install Python and common image/PDF system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    # Ensure GPU build of onnxruntime is active (not CPU-only)
    pip3 uninstall -y onnxruntime onnxruntime-gpu || true && \
    pip3 install --no-cache-dir onnxruntime-gpu[cuda,cudnn]

# Copy application code
COPY . .

# Create persistent data directories
RUN mkdir -p /data/qdrant_local /data/models /data/uploaded_files

EXPOSE 5000

# Run as a module so package imports resolve cleanly
CMD ["python3", "server.py"]