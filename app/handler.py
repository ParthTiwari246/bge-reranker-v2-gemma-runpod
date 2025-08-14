
import runpod
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "BAAI/bge-reranker-v2-gemma"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

def rerank(query, docs):
    pairs = [[query, d] for d in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs, return_dict=True).logits.view(-1).float()
    # Sort high-to-low
    ranked = sorted(zip(docs, logits.tolist()), key=lambda x: x[1], reverse=True)
    # return tidy structure
    return [{"document": d, "score": float(s)} for d, s in ranked]

def handler(event):
    '''
    Input:
    {
        "query": "string",
        "documents": ["doc1", "doc2", ...]
    }
    '''
    data = event.get("input", {}) or {}
    query = data.get("query")
    docs = data.get("documents", [])
    if not query or not isinstance(docs, list) or len(docs) == 0:
        return {"error": "Provide 'query' (str) and non-empty 'documents' (list[str])."}

    results = rerank(query, docs)
    return {"results": results}

