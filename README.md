# EbookRAG

EbookRAG is a small Retrieval-Augmented Generation (RAG) toolkit for EPUB libraries. It indexes local ebooks and answers questions about them by combining open-source embeddings with either LangChain or LlamaIndex pipelines.

## Requirements
- Python 3.10 or newer
- [uv](https://github.com/astral-sh/uv) for environment and dependency management
- Access to an OpenAI-compatible chat model (hosted or self-hosted)
- Pandoc (installed automatically through `pypandoc` when missing)

## Setup with uv
```bash
uv venv
source .venv/bin/activate
# install core dependencies declared in pyproject.toml
uv sync
```
Choose one (or both) pipeline extras:
```bash
# LangChain toolchain
uv sync --extra langchain

# LlamaIndex toolchain
uv sync --extra llamaindex
```
If you prefer ad-hoc installation, `uv pip install '.[langchain]'` or `uv pip install '.[llamaindex]'` works as well.

## Configuration
1. Copy the sample environment file and adjust the values:
   ```bash
   cp .env.example .env
   ```
2. Place your EPUB files inside the directory configured by `EPUB_DIR` (defaults to `epubs/`).
3. Update the variables in `.env` as needed:
   - `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `OPENAI_API_MODEL`, `OPENAI_TEMPERATURE`
   - `EMBED_PROVIDER` and related settings to choose HuggingFace, OpenAI-compatible, or Ollama embeddings
   - Chunking parameters such as `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_SEPARATORS`
   - Optional `QA_PROMPT_TEMPLATE` or `QA_PROMPT_TEMPLATE_PATH` for custom prompts

## Usage
- **LangChain pipeline**
  ```bash
  python rag_langchain.py
  ```
  The script builds (or updates) a FAISS vector store on first run, then drops you into an interactive question prompt. Embeddings are saved under `vector_store_langchain/` with a manifest for incremental updates.

- **LlamaIndex pipeline**
  ```bash
  python rag_llamaindex.py
  ```
  This variant maintains its own FAISS index inside `vector_store_llamaindex/` and offers the same interactive question loop.

Both scripts log retrieval progress and print answer references so you can trace responses back to source files.

## Tips
- The toolkit supports multilingual embeddings out of the box via `intfloat/multilingual-e5-base`.
- If you switch to Ollama for embeddings, ensure the Ollama server is running and reachable.
- Re-run the chosen script whenever you add or update EPUB files; manifests detect changes and refresh only the affected books.
