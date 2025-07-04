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


def build_prompt(question, context_chunks):
    context = "\n---\n".join([chunk["text"] for chunk in context_chunks])
    prompt = f"""You are a financial analyst assistant for CrediTrust.
Your task is to answer questions based on real customer complaints.
Use only the context provided below. If unsure, say so.

Context:
{context}

Question:
{question}

Answer:"""
    return prompt

# step 3
from transformers import pipeline

# Load local model pipeline
qa_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=200)

def generate_answer(prompt):
    return qa_pipeline(prompt)[0]["generated_text"]

#step 4

test_questions = [
    "Why are customers unhappy with Buy Now, Pay Later?",
    "What are the major issues in credit card services?",
    "What complaints are common in money transfers?"
]

for question in test_questions:
    chunks = retrieve_relevant_chunks(question)
    prompt = build_prompt(question, chunks)
    answer = generate_answer(prompt)

    print("Q:", question)
    print("Answer:", answer)
    print("Sources:")
    for c in chunks[:2]:
        print("-", c["text"][:150])
    print("-" * 80)
