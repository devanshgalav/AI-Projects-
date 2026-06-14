
# Agentic RAG (Single File)

## Features
- PDF-first retrieval
- FAISS vector search
- OpenAI embeddings
- Tavily search fallback
- Wikipedia tool
- arXiv tool
- Streamlit UI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy:

```bash
cp .env.example .env
```

Run:

```bash
streamlit run app.py
```

## Example Questions

- Summarize this document
- What are the key findings?
- Who is Alan Turing?
- Latest OpenAI model?
- Recent papers on Agentic RAG?
