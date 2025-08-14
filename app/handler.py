import runpod
from sentence_transformers import CrossEncoder
import torch

MODEL_NAME = "BAAI/bge-reranker-v2-gemma"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder(MODEL_NAME, device=device)

def handler(event):
    """
    OpenWebUI Reranker-compatible.
    Input: { "query": "...", "documents": ["...", "..."] }
    Output: { "scores": [0.9, 0.1, ...] }
    """
    data = event.get("input", {}) or {}
    query = data.get("query")
    docs = data.get("documents", [])

    if not query or not isinstance(docs, list) or len(docs) == 0:
        return {"error": "Missing 'query' or 'documents'"}

    with torch.no_grad():
        scores = model.predict([[query, doc] for doc in docs])

    return {"scores": [float(s) for s in scores]}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})