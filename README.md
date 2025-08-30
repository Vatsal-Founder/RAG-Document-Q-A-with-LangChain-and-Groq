# RAG Document Q\&A — LangChain + Groq + FAISS

End‑to‑end **RAG** application that answers questions over your **internal PDFs** using:

* **Groq** for fast, low‑latency LLM inference (chat generation)
* **OpenAI embeddings** for vectorization
* **FAISS** for vector search
* **LangSmith** for tracing and run analytics

> Repo: `Vatsal-Founder/RAG-Document-Q-A-with-LangChain-and-Groq`


Link: 

---

## Features

* 📄 **Bring your own PDFs** — drop files into a folder and index them.
* 🧭 **Accurate retrieval** with FAISS and chunking.
* ⚡ **Groq LLM** (e.g., Llama‑3 family) for fast responses.
* 🔎 **OpenAI embeddings** (swappable) for high‑quality retrieval.
* 🧩 **LangChain** orchestration with optional LangSmith traces.

---

## Architecture

```
[User] → [Chat UI / CLI]
            ↓
       [LangChain App]
   ├─ Retriever (FAISS over PDF chunks)
   ├─ Embeddings (OpenAI)
   └─ Generator (Groq LLM)
            ↓
        [Answer + Sources]
```

---

## Project Structure (typical)

```
.
├── app.py                 # entrypoint (chat UI or server)
├── requirements.txt
├── data/
│   └── pdfs/              # place your PDFs here
└── README.md
```

---

## Requirements

* Python 3.10+
* OpenAI API key (embeddings)
* Groq API key (LLM)

---

## Setup

```bash
git clone https://github.com/Vatsal-Founder/RAG-Document-Q-A-with-LangChain-and-Groq.git
cd RAG-Document-Q-A-with-LangChain-and-Groq
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# create .env with your secrets
cp .env.example .env  # if the file exists; otherwise create it
```

### Environment variables (minimum)

```ini
# LLM (Groq)
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b  # example; set to your preferred Groq model

# Embeddings (OpenAI)
OPENAI_API_KEY=your_openai_key
EMBEDDINGS_MODEL=text-embedding-3-small


# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=rag-docs
```

---


## Run locally

Choose the mode that matches `app.py` (UI or server). If unsure, try Streamlit first.

### Streamlit UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and start asking questions about your PDFs.


---



## How to Use (at a glance)

1. **Ingest** your PDFs.
2. **Run** the app (Streamlit or server).
3. **Ask** natural language questions.
4. See **answers + cited snippets/sources** (if implemented in your UI).

---

## Configuration Tips

* Start with `text-embedding-3-small` for speed/cost; upgrade to `text-embedding-3-large` for best recall.
* Tune `chunk-size`/`overlap` for your PDFs (technical docs often benefit from \~1000/150).
* Keep your FAISS index folder outside of transient build directories so it survives container rebuilds.

---

