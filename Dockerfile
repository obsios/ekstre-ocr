FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex-cache \
    OCR_DEVICE=gpu:0 \
    OCR_RECOGNITION_BATCH_SIZE=8 \
    OCR_PAGE_BATCH_SIZE=8 \
    OCR_DPI=300 \
    MAX_DOCUMENT_MB=35 \
    MAX_PAGES=100

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3-pip \
        python-is-python3 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --ignore-installed cryptography -r requirements.txt

# Keep the heavy model-cache layer independent from normal handler edits.
COPY ocr_engine.py download_models.py ./
RUN python download_models.py \
    && find /opt/paddlex-cache -type f | sort | head -50

COPY input_loader.py handler.py ./

CMD ["python", "-u", "handler.py"]
