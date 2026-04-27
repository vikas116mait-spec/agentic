# GPU-enabled container for the FastAPI + Ollama + local-QLoRA trainer.
# Used for Fly.io deploys (see README "Deploying to Vercel" section); the
# slimmer CPU-only ../Dockerfile.api is kept for docker-compose dev.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AGENTIC_DATA_DIR=/data/python_api \
    AGENTIC_UPLOADS_DIR=/data/uploads_python \
    OLLAMA_MODELS=/data/ollama \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_AUTO_MANAGE=0 \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        tini \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        python3-pip \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt requirements-local-training.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r requirements-local-training.txt

COPY python_api ./python_api
COPY prisma ./prisma

COPY python_api/start.sh /usr/local/bin/start-agentic.sh
RUN chmod +x /usr/local/bin/start-agentic.sh

EXPOSE 8080
VOLUME ["/data"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start-agentic.sh"]
