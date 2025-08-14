FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app/handler.py .

# Pre-download model
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-gemma')"

# Start the RunPod serverless handler
CMD ["python", "handler.py"]
