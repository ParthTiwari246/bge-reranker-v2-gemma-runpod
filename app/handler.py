import runpod
from sentence_transformers import CrossEncoder
import torch

MODEL_NAME = "BAAI/bge-reranker-v2-gemma"

# Load model at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder(MODEL_NAME, device=device)

def rerank(query, docs, top_k=None):
    """
    Reranks a list of documents given a query.
    Optionally returns only the top_k results.
    """
    pairs = [[query, doc] for doc in docs]
    with torch.no_grad():
        scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    results = [{"document": d, "score": float(s)} for d, s in ranked]
    if top_k is not None:
        results = results[:top_k]
    return results

def handler(event):
    """
    Expects JSON input:
    {
        "query": "What is AI?",
        "documents": ["AI is artificial intelligence.", "Bananas are yellow."],
        "top_k": 2  # optional
    }
    """
    try:
        data = event.get("input", {}) or {}
        query = data.get("query")
        docs = data.get("documents", [])
        top_k = data.get("top_k", None)

        if not query or not isinstance(docs, list) or len(docs) == 0:
            return {"error": "Provide 'query' (str) and non-empty 'documents' (list[str])."}

        results = rerank(query, docs, top_k=top_k)
        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
