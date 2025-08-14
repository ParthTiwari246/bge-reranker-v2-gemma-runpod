
FROM python:3.10-slim

# Prevents Debconf warnings
ENV DEBIAN_FRONTEND=noninteractive
# Hugging Face cache to keep layers tidy
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code
COPY app/handler.py /app/handler.py

# Pre-download model + tokenizer during build to reduce cold starts
RUN python - <<'PY'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = "BAAI/bge-reranker-v2-gemma"
AutoTokenizer.from_pretrained(m)
AutoModelForSequenceClassification.from_pretrained(m)
print("Model cached successfully.")
PY

# Expose the runpod serverless port (informational)
EXPOSE 8000

# Launch handler
CMD ["python", "handler.py"]
