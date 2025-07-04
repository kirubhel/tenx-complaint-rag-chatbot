# Tenx Complaint Chatbot

This project implements a Retrieval-Augmented Generation (RAG) based chatbot that helps analyze and summarize customer complaints from a dataset using LLMs. It is part of the 10 Academy Tenx Challenge.

## 💡 Project Objective

To build an intelligent chatbot that:
- Accepts natural language queries related to financial complaints.
- Retrieves relevant complaint snippets using vector search.
- Generates coherent answers using a large language model (LLM).

## 📁 Project Structure

```
tenx-complaint-rag-chatbot/
│
├── data/
│   └── complaints.csv            # Raw complaints dataset
│
├── src/
│   ├── rag_pipeline.py          # Main RAG pipeline logic
│   └── utils.py                 # Helper functions for retrieval and preprocessing
│
├── app.py                       # Gradio frontend
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## ⚙️ How It Works

1. **Preprocessing:** Complaints from `complaints.csv` are cleaned and chunked.
2. **Embedding & Indexing:** Complaints are converted to embeddings and stored in a FAISS index.
3. **Retrieval:** On receiving a user query, top-k similar complaint chunks are retrieved.
4. **Generation:** A locally loaded LLM (e.g., TinyLlama) generates an answer based on the context.
5. **Frontend:** Gradio UI allows users to interact with the chatbot.

## 🧪 Sample Questions
- What are common issues in money transfers?
- What do users complain about credit cards?
- Why are customers unhappy with Buy Now, Pay Later?

## 🚀 Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

## 📷 Screenshots

See the final report PDF for full screenshots and example queries.

## 🧠 Model Info

The chatbot uses the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model hosted on HuggingFace. Ensure your system supports MPS or CUDA for better performance.

## 📌 Notes

- Replace restricted models with accessible ones if needed (e.g., replace Mistral-7B with TinyLlama).
- If running into gated repo errors, ensure you have appropriate Hugging Face permissions or use alternatives.

## 📄 License

MIT License
