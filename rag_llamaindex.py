import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from types import MethodType

try:
    import pypandoc
except ImportError:
    pypandoc = None

from dotenv import load_dotenv

# ---- LlamaIndex core components
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# ---- Reader: parse EPUB files with Unstructured
from llama_index.readers.file import UnstructuredReader

# ---- Embedding & LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# ---- Vector store: FAISS
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def ensure_pandoc():
    if pypandoc is None:
        raise ImportError("pypandoc is required for automatic Pandoc installation.")
    try:
        pypandoc.get_pandoc_version()
    except (OSError, RuntimeError):
        logger.info("Pandoc not found. Downloading with pypandoc...")
        pypandoc.download_pandoc()
        logger.info("Pandoc download finished")

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ========== Configuration ==========
EPUB_DIR = os.getenv("EPUB_DIR", "epubs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")  # Suitable for multilingual medical retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMAINDEX_VECTOR_STORE_DIR = Path(os.getenv("LLAMAINDEX_VECTOR_STORE_DIR", "vector_store_llamaindex"))
EMBED_BATCH_SIZE = max(1, int(os.getenv("VECTOR_EMBED_BATCH_SIZE", "64")))

# LLM: OpenAI compatible (swap in your own OpenAI-compatible service if needed)
_llm_kwargs = {
    "model": OPENAI_API_MODEL,
    "temperature": OPENAI_TEMPERATURE,
}
if OPENAI_API_BASE_URL:
    _llm_kwargs["api_base"] = OPENAI_API_BASE_URL
    # `OpenAI` currently accepts both `api_base` and `base_url`; populate both for forward compatibility
    _llm_kwargs["base_url"] = OPENAI_API_BASE_URL
if OPENAI_API_KEY:
    _llm_kwargs["api_key"] = OPENAI_API_KEY
llm = OpenAI(**_llm_kwargs)

# Embedding
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def _enable_llamaindex_embedding_progress(embedding_model):
    if getattr(embedding_model, "_progress_logging_enabled", False):
        return

    if hasattr(embedding_model, "get_text_embedding_batch"):
        original_batch = embedding_model.get_text_embedding_batch

        def get_text_embedding_batch_with_progress(self, texts, *args, **kwargs):
            texts_list = list(texts) if not isinstance(texts, list) else texts
            total = len(texts_list)
            if total == 0:
                return original_batch(texts_list, *args, **kwargs)

            outputs = []
            for start in range(0, total, EMBED_BATCH_SIZE):
                end = min(start + EMBED_BATCH_SIZE, total)
                batch = texts_list[start:end]
                logger.info(
                    "LlamaIndex embedding progress: %d/%d document nodes",
                    end,
                    total,
                )
                outputs.extend(original_batch(batch, *args, **kwargs))
            return outputs

        object.__setattr__(
            embedding_model,
            "get_text_embedding_batch",
            MethodType(get_text_embedding_batch_with_progress, embedding_model),
        )

        if hasattr(embedding_model, "aget_text_embedding_batch"):
            original_async_batch = embedding_model.aget_text_embedding_batch

            async def aget_text_embedding_batch_with_progress(self, texts, *args, **kwargs):
                texts_list = list(texts) if not isinstance(texts, list) else texts
                total = len(texts_list)
                if total == 0:
                    return await original_async_batch(texts_list, *args, **kwargs)

                outputs = []
                for start in range(0, total, EMBED_BATCH_SIZE):
                    end = min(start + EMBED_BATCH_SIZE, total)
                    batch = texts_list[start:end]
                    logger.info(
                        "LlamaIndex async embedding progress: %d/%d document nodes",
                        end,
                        total,
                    )
                    outputs.extend(await original_async_batch(batch, *args, **kwargs))
                return outputs

            object.__setattr__(
                embedding_model,
                "aget_text_embedding_batch",
                MethodType(aget_text_embedding_batch_with_progress, embedding_model),
            )
    else:
        original_single = embedding_model.get_text_embedding

        def get_text_embedding_with_progress(self, text, *args, **kwargs):
            logger.info("LlamaIndex embedding progress: processed another document node")
            return original_single(text, *args, **kwargs)

        object.__setattr__(
            embedding_model,
            "get_text_embedding",
            MethodType(get_text_embedding_with_progress, embedding_model),
        )

    object.__setattr__(embedding_model, "_progress_logging_enabled", True)


_enable_llamaindex_embedding_progress(embed_model)

# Node splitting: keep small chunks for precise citations
node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = node_parser
Settings.callback_manager = CallbackManager([])

# ========== EPUB helpers ==========
def get_epub_paths(epub_dir: str) -> list[Path]:
    target_dir = Path(epub_dir)
    if not target_dir.exists():
        logger.error("EPUB directory not found at %s - exiting.", target_dir.resolve())
        sys.exit(1)
    epub_paths = sorted(target_dir.glob("**/*.epub"))
    if not epub_paths:
        logger.error("No EPUB files found under %s - exiting.", target_dir.resolve())
        sys.exit(1)
    return epub_paths


def compute_epub_manifest(epub_paths: list[Path]) -> dict:
    files = []
    for path in epub_paths:
        stat = path.stat()
        with path.open("rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        files.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": digest,
            }
        )
    files.sort(key=lambda item: item["path"])
    return {"files": files}


def load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to read manifest at %s (%s); index will be rebuilt.",
            manifest_path,
            exc,
        )
        return None


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


# ========== Load EPUB files ==========
EPUB_DIR_PATH = Path(EPUB_DIR)
epub_paths = get_epub_paths(EPUB_DIR)
manifest_path = LLAMAINDEX_VECTOR_STORE_DIR / "manifest.json"
current_manifest = compute_epub_manifest(epub_paths)
stored_manifest = load_manifest(manifest_path)

vector_store_valid = (
    LLAMAINDEX_VECTOR_STORE_DIR.exists()
    and stored_manifest is not None
    and stored_manifest == current_manifest
)

index = None
if vector_store_valid:
    try:
        logger.info(
            "EPUB fingerprints unchanged; loading LlamaIndex store from %s",
            LLAMAINDEX_VECTOR_STORE_DIR.resolve(),
        )
        vector_store = FaissVectorStore.from_persist_dir(
            str(LLAMAINDEX_VECTOR_STORE_DIR)
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(LLAMAINDEX_VECTOR_STORE_DIR),
        )
        index = load_index_from_storage(storage_context)
    except Exception as exc:
        logger.warning(
            "Failed to load existing LlamaIndex store (%s); rebuilding",
            exc,
        )
        vector_store_valid = False

if index is None:
    if stored_manifest is None:
        logger.info(
            "Manifest missing or unreadable for %s; rebuilding index",
            LLAMAINDEX_VECTOR_STORE_DIR.resolve(),
        )
    else:
        logger.info(
            "Detected EPUB changes; rebuilding index at %s",
            LLAMAINDEX_VECTOR_STORE_DIR.resolve(),
        )

    logger.info("Preparing to index %d EPUB file(s)", len(epub_paths))
    ensure_pandoc()

    file_extractor = {".epub": UnstructuredReader()}
    logger.info("Loading documents from %s", EPUB_DIR_PATH.resolve())
    docs = SimpleDirectoryReader(
        input_dir=EPUB_DIR,
        file_extractor=file_extractor,
        recursive=True,
    ).load_data()
    logger.info("Loaded %d document node(s)", len(docs))

    logger.info("Initializing FAISS index for LlamaIndex store")
    probe_embedding = embed_model.get_text_embedding("dimension_probe")
    if isinstance(probe_embedding, list):
        dim = len(probe_embedding)
    else:
        dim = len(probe_embedding.tolist())
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("Building LlamaIndex VectorStoreIndex")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )
    storage_context.persist(persist_dir=str(LLAMAINDEX_VECTOR_STORE_DIR))
    save_manifest(manifest_path, current_manifest)
    logger.info(
        "Persisted LlamaIndex store and manifest to %s",
        LLAMAINDEX_VECTOR_STORE_DIR.resolve(),
    )

# ========== Query engine (with citations) ==========
logger.info("Configuring LlamaIndex retriever and query engine")
retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
response_synthesizer = get_response_synthesizer(response_mode="compact")

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)


def ask(q: str):
    logger.info("Invoking LlamaIndex query engine for query: %s", q)
    resp = query_engine.query(q)
    logger.info("Response: %s", resp.response)
    if resp.source_nodes:
        lines = []
        for idx, src in enumerate(resp.source_nodes, 1):
            meta = src.node.metadata
            lines.append(
                (
                    f"{idx}. file={meta.get('file_path')}, "
                    f"section={meta.get('section')}, "
                    f"start_char={src.node.start_char_idx}, "
                    f"end_char={src.node.end_char_idx}"
                )
            )
        logger.info("Sources:\n%s", "\n".join(lines))
    else:
        logger.info("No sources returned for query")


def interactive_loop():
    logger.info("Entering interactive LlamaIndex QA loop. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("Enter question (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("Interactive session terminated by user")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            logger.info("Exiting interactive QA loop")
            break

        ask(user_input)


if __name__ == "__main__":
    interactive_loop()
