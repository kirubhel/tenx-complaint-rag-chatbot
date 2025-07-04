# src/embedding_indexing.py

import os
import pandas as pd
from tqdm import tqdm
import faiss
import pickle

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load cleaned complaints
df = pd.read_csv('../data/filtered_complaints.csv')

# 2. Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# 3. Chunk the narratives
documents = []
metadatas = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    chunks = text_splitter.split_text(str(row["cleaned_narrative"]))
    for chunk in chunks:
        documents.append(chunk)
        metadatas.append({
            "product": row["Product"],
            "original_index": idx
        })

# 4. Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents, show_progress_bar=True)

# 5. Build FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. Persist vector store and metadata
os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, "vector_store/faiss_index.index")

with open("vector_store/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print("✅ Vector store created and saved to vector_store/")
