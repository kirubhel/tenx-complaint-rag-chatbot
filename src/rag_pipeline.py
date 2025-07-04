# src/rag_pipeline.py

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load vector store & metadata
index = faiss.read_index("vector_store/faiss_index.index")

with open("vector_store/documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(question, k=5):
    # Embed the query
    question_embedding = model.encode([question])
    
    # Search vector store
    distances, indices = index.search(np.array(question_embedding), k)
    
    retrieved_chunks = []
    for i in indices[0]:
        retrieved_chunks.append({
            "text": documents[i],
            "meta": metadata[i]
        })
    
    return retrieved_chunks
