import runpod
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-gemma"

# Load model once at container start
model = CrossEncoder(MODEL_NAME)

def rerank(query, docs):
    # Create query-doc pairs
    pairs = [[query, doc] for doc in docs]
    scores = model.predict(pairs)
    # Sort by score descending
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [{"document": d, "score": float(s)} for d, s in ranked]

def handler(event):
    """
    Expects input like:
    {
        "query": "What is AI?",
        "documents": ["AI is artificial intelligence.", "Bananas are yellow."]
    }
    """
    data = event.get("input", {}) or {}
    query = data.get("query")
    docs = data.get("documents", [])

    if not query or not isinstance(docs, list) or len(docs) == 0:
        return {"error": "Provide 'query' (str) and non-empty 'documents' (list[str])."}

    results = rerank(query, docs)
    return {"results": results}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
