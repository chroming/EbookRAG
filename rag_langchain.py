import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import MethodType

try:
    import pypandoc
except ImportError:
    pypandoc = None

from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


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

EPUB_DIR = os.getenv("EPUB_DIR", "epubs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
TOP_K = int(os.getenv("TOP_K", "5"))
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "vector_store_langchain"))
EMBED_BATCH_SIZE = max(1, int(os.getenv("VECTOR_EMBED_BATCH_SIZE", "64")))
EMBED_PROVIDER = (os.getenv("EMBED_PROVIDER") or "huggingface").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_OPTIONS = os.getenv("OLLAMA_EMBED_OPTIONS")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
_DEFAULT_CHUNK_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
_chunk_separators_raw = os.getenv("CHUNK_SEPARATORS")
if _chunk_separators_raw:
    try:
        CHUNK_SEPARATORS = json.loads(_chunk_separators_raw)
        if not isinstance(CHUNK_SEPARATORS, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Invalid CHUNK_SEPARATORS value; expected JSON list. Using default separators. (%s)",
            exc,
        )
        CHUNK_SEPARATORS = _DEFAULT_CHUNK_SEPARATORS
else:
    CHUNK_SEPARATORS = _DEFAULT_CHUNK_SEPARATORS

QA_PROMPT_TEMPLATE = os.getenv("QA_PROMPT_TEMPLATE")
QA_PROMPT_TEMPLATE_PATH = os.getenv("QA_PROMPT_TEMPLATE_PATH")

# LLM: OpenAI compatible (you can swap in a local service)
_llm_kwargs = {
    "model": OPENAI_API_MODEL,
    "temperature": OPENAI_TEMPERATURE,
}
if OPENAI_API_BASE_URL:
    _llm_kwargs["base_url"] = OPENAI_API_BASE_URL
if OPENAI_API_KEY:
    _llm_kwargs["api_key"] = OPENAI_API_KEY
llm = ChatOpenAI(**_llm_kwargs)

# Embedding


def create_embeddings():
    if EMBED_PROVIDER == "huggingface":
        logger.info("Using HuggingFace embeddings with model=%s", EMBED_MODEL_NAME)
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

    if EMBED_PROVIDER == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Ollama embeddings requested but langchain_community Ollama support is unavailable."
            ) from exc

        extra_options: dict[str, object] = {}
        if OLLAMA_EMBED_OPTIONS:
            try:
                extra_options = json.loads(OLLAMA_EMBED_OPTIONS)
                if not isinstance(extra_options, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Ignoring OLLAMA_EMBED_OPTIONS because it is not valid JSON object: %s",
                    exc,
                )
                extra_options = {}

        if not getattr(OllamaEmbeddings, "_ebookrag_sanitized", False):
            original_getter = OllamaEmbeddings._default_params.fget
            # Remove empty key to prevent warning in ollama api
            def sanitized_default_params(self, _orig=original_getter):
                params = dict(_orig(self))
                options = dict(params.get("options") or {})
                options = {k: v for k, v in options.items() if v is not None}
                custom = getattr(self, "_ebookrag_embed_options", None)
                if custom:
                    options.update({k: v for k, v in custom.items() if v is not None})
                if options:
                    params["options"] = options
                else:
                    params.pop("options", None)
                return params

            OllamaEmbeddings._default_params = property(sanitized_default_params)
            OllamaEmbeddings._ebookrag_sanitized = True

        model_name = (OLLAMA_EMBED_MODEL or DEFAULT_OLLAMA_EMBED_MODEL).strip()
        kwargs = {}
        if OLLAMA_BASE_URL:
            kwargs["base_url"] = OLLAMA_BASE_URL
        logger.info("Using Ollama embeddings with model=%s", model_name)
        embedding = OllamaEmbeddings(model=model_name, **kwargs)
        embedding._ebookrag_embed_options = extra_options
        return embedding

    raise ValueError(
        "Unsupported EMBED_PROVIDER '%s'. Expected 'huggingface' or 'ollama'."
        % EMBED_PROVIDER
    )


embeddings = create_embeddings()


def load_prompt_template() -> str:
    if QA_PROMPT_TEMPLATE_PATH:
        template_path = Path(QA_PROMPT_TEMPLATE_PATH)
        try:
            content = template_path.read_text(encoding="utf-8")
            logger.info("Using prompt template loaded from %s", template_path.resolve())
            return content
        except OSError as exc:
            logger.warning(
                "Failed to read QA prompt template from %s (%s); falling back to default template.",
                template_path,
                exc,
            )

    if QA_PROMPT_TEMPLATE:
        logger.info("Using prompt template provided via QA_PROMPT_TEMPLATE environment variable")
        return QA_PROMPT_TEMPLATE

    return (
        "You are a knowledge assistant. Answer the question using the provided excerpts.\n"
        "Context:\n{context}\n\n"
        "Guidelines:\n"
        "1) Base the answer strictly on the context; if unsure, respond with \"No clear evidence found in the provided context.\"\n"
        "2) Present key steps as a list.\n"
        "3) Context snippets are prefixed with \"Source Info: ...\" that list file names and sections—use them when writing the answer.\n"
        "4) Conclude with **References** and bullet(s) formatted exactly as \"- <section or label> (<source filename>; <location if available>)\".\n\n"
        "Question: {input}"
    )


_REFERENCE_CANDIDATE_KEYS = (
    "display_label",
    "title",
    "heading",
    "section",
    "category",
    "source_title",
    "source_section",
    "subtitle",
)


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _trim_snippet(text: str, max_length: int = 80) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def _metadata_label(meta: dict) -> str | None:
    for key in _REFERENCE_CANDIDATE_KEYS:
        value = meta.get(key)
        if isinstance(value, str):
            cleaned = _collapse_whitespace(value).strip()
            if cleaned:
                return cleaned
    return None


def _annotate_documents(documents: list[Document]) -> None:
    for doc in documents:
        meta = doc.metadata = doc.metadata or {}
        original_content = doc.page_content or ""
        meta.setdefault("_original_excerpt", original_content)

        label = _metadata_label(meta)
        if not label:
            for line in original_content.splitlines():
                cleaned = _collapse_whitespace(line).strip()
                if cleaned:
                    label = _trim_snippet(cleaned)
                    break
        if not label:
            label = "Excerpt"
        meta["display_label"] = label

        location_parts = []
        if "page_number" in meta:
            location_parts.append(f"page {meta['page_number']}")
        if "element_id" in meta:
            location_parts.append(f"id {meta['element_id']}")
        if "start_index" in meta and "end_index" in meta:
            location_parts.append(f"chars {meta['start_index']}-{meta['end_index']}")
        location = ", ".join(location_parts)
        if location:
            meta["reference_location"] = location

        source_path = meta.get("source")
        filename = Path(source_path).name if source_path else "unknown"
        header_parts = [f"File: {filename}"]
        if label:
            header_parts.append(f"Section: {label}")
        if location:
            header_parts.append(f"Location: {location}")
        header = "Source Info: " + "; ".join(header_parts)
        doc.page_content = f"{header}\n{original_content}"


def _enable_embedding_progress_logging(embedding_model):
    if getattr(embedding_model, "_progress_logging_enabled", False):
        return

    original_embed = embedding_model.embed_documents

    def embed_documents_with_progress(self, texts, *args, **kwargs):
        texts_list = list(texts) if not isinstance(texts, list) else texts
        total = len(texts_list)
        if total == 0:
            return original_embed(texts_list, *args, **kwargs)

        results = []
        for start in range(0, total, EMBED_BATCH_SIZE):
            end = min(start + EMBED_BATCH_SIZE, total)
            batch = texts_list[start:end]
            logger.info(
                "LangChain embedding progress: %d/%d document chunks",
                end,
                total,
            )
            results.extend(original_embed(batch, *args, **kwargs))
        return results

    object.__setattr__(
        embedding_model,
        "embed_documents",
        MethodType(embed_documents_with_progress, embedding_model),
    )
    object.__setattr__(embedding_model, "_progress_logging_enabled", True)


_enable_embedding_progress_logging(embeddings)


def get_epub_paths(epub_dir: str) -> list[Path]:
    target_dir = Path(epub_dir)
    if not target_dir.exists():
        logger.info(
            "EPUB directory not found at %s; creating directory.",
            target_dir.resolve(),
        )
        target_dir.mkdir(parents=True, exist_ok=True)
    epub_paths = sorted(target_dir.glob("**/*.epub"))
    if not epub_paths:
        logger.error("No EPUB files found under %s - exiting.", target_dir.resolve())
        sys.exit(1)
    return epub_paths


def compute_epub_manifest(epub_paths: list[Path]) -> dict:
    files: dict[str, dict[str, object]] = {}
    for path in epub_paths:
        stat = path.stat()
        with path.open("rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        files[str(path.resolve())] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": digest,
        }
    return {"files": files}


def load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to read manifest at %s (%s); vector store will be rebuilt.",
            manifest_path,
            exc,
        )
        return None

    files = data.get("files") if isinstance(data, dict) else None
    if isinstance(files, list):
        converted: dict[str, dict[str, object]] = {}
        for entry in files:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not path:
                continue
            converted[str(path)] = {
                k: v
                for k, v in entry.items()
                if k in {"size", "mtime", "sha256", "doc_ids"}
            }
        data["files"] = converted
        return data
    if isinstance(files, dict):
        return data
    logger.warning(
        "Manifest at %s is in an unexpected format; vector store will be rebuilt.",
        manifest_path,
    )
    return None


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def load_epubs(epub_paths: list[Path]):
    logger.info("Loading %d EPUB file(s)", len(epub_paths))
    docs = []
    for p in epub_paths:
        logger.debug("Loading EPUB file: %s", p)
        tmp_dir = None
        loader_path = str(p)
        if any(ch in loader_path for ch in "[]{}*?"):
            tmp_dir = tempfile.TemporaryDirectory()
            safe_path = Path(tmp_dir.name) / "ebook.epub"
            shutil.copy2(p, safe_path)
            loader_path = str(safe_path)
            logger.debug("Copied EPUB with special characters to temporary path %s", loader_path)

        try:
            loader = UnstructuredEPubLoader(loader_path)
            dlist = loader.load()
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()
        # Attach the file path as metadata for easier citation
        for d in dlist:
            d.metadata = d.metadata or {}
            d.metadata["source"] = str(p)
        docs.extend(dlist)
    logger.info("Loaded %d document(s) from EPUB files", len(docs))
    return docs

epub_paths = get_epub_paths(EPUB_DIR)
manifest_path = VECTOR_STORE_DIR / "manifest.json"
current_manifest = compute_epub_manifest(epub_paths)
stored_manifest = load_manifest(manifest_path)

manifest_data = stored_manifest if stored_manifest is not None else {"files": {}}
if "files" not in manifest_data or not isinstance(manifest_data["files"], dict):
    manifest_data = {"files": {}}

stored_files: dict[str, dict[str, object]] = manifest_data["files"]
current_files: dict[str, dict[str, object]] = current_manifest["files"]

vectordb = None
vector_store_loaded = False
if VECTOR_STORE_DIR.exists():
    try:
        vectordb = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store_loaded = True
        logger.info(
            "Loaded existing FAISS vector store from %s",
            VECTOR_STORE_DIR.resolve(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load existing vector store at %s (%s); it will be recreated as needed.",
            VECTOR_STORE_DIR.resolve(),
            exc,
        )
        vectordb = None
        stored_files.clear()

removed_paths = [path for path in list(stored_files.keys()) if path not in current_files]
if removed_paths:
    if vectordb is None and vector_store_loaded:
        logger.warning("Vector store was expected but could not be loaded; skipped deletions for removed EPUBs.")
    for path in removed_paths:
        entry = stored_files.pop(path, None)
        doc_ids = entry.get("doc_ids") if isinstance(entry, dict) else None
        if vectordb is not None and doc_ids:
            try:
                vectordb.delete(doc_ids)
                logger.info("Removed %d embedding vector(s) for deleted EPUB %s", len(doc_ids), path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove embeddings for %s (%s)", path, exc)
    if vectordb is not None:
        vectordb.save_local(str(VECTOR_STORE_DIR))
    save_manifest(manifest_path, manifest_data)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=CHUNK_SEPARATORS,
)

pending_updates: list[tuple[Path, str, dict[str, object], dict[str, object] | None]] = []
for path in epub_paths:
    resolved = str(path.resolve())
    file_info = current_files.get(resolved)
    if file_info is None:
        logger.warning("Missing fingerprint information for %s; skipping.", resolved)
        continue
    stored_entry = stored_files.get(resolved)
    if stored_entry and stored_entry.get("sha256") == file_info.get("sha256"):
        continue
    pending_updates.append((path, resolved, file_info, stored_entry))

if pending_updates:
    ensure_pandoc()
    names = ", ".join(str(item[0].name) for item in pending_updates)
    logger.info(
        "Detected %d EPUB file(s) pending embedding: %s",
        len(pending_updates),
        names,
    )
else:
    logger.info("All EPUB embeddings are up to date; no pending files detected.")

for path, resolved, file_info, stored_entry in pending_updates:
    if stored_entry and vectordb is not None:
        old_ids = stored_entry.get("doc_ids")
        if old_ids:
            try:
                vectordb.delete(old_ids)
                logger.info(
                    "Removed %d outdated embedding vector(s) for %s before re-embedding",
                    len(old_ids),
                    resolved,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove outdated embeddings for %s (%s)", resolved, exc)

    raw_docs = load_epubs([path])
    logger.info("Splitting %d raw document(s) for %s", len(raw_docs), resolved)
    docs = splitter.split_documents(raw_docs)
    _annotate_documents(docs)
    logger.info("Generated %d document chunk(s) for %s", len(docs), resolved)

    doc_ids = [f"{file_info['sha256']}:{i}" for i in range(len(docs))]

    if docs:
        if vectordb is None:
            logger.info("Creating new FAISS vector store with %s", resolved)
            vectordb = FAISS.from_documents(docs, embeddings, ids=doc_ids)
        else:
            vectordb.add_documents(docs, ids=doc_ids)
            logger.info("Appended %d chunk(s) from %s to FAISS index", len(docs), resolved)
        vectordb.save_local(str(VECTOR_STORE_DIR))
    else:
        logger.info("No chunks generated for %s; skipping vector store update", resolved)

    stored_files[resolved] = {
        **file_info,
        "doc_ids": doc_ids,
    }
    save_manifest(manifest_path, manifest_data)

if vectordb is None:
    if VECTOR_STORE_DIR.exists():
        try:
            vectordb = FAISS.load_local(
                str(VECTOR_STORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "Loaded FAISS vector store after incremental updates from %s",
                VECTOR_STORE_DIR.resolve(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unable to load FAISS vector store from %s after updates (%s)",
                VECTOR_STORE_DIR.resolve(),
                exc,
            )
            sys.exit(1)
    else:
        logger.error("Vector store directory %s does not exist; add EPUB files first.", VECTOR_STORE_DIR.resolve())
        sys.exit(1)

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
logger.info("Vector store ready; retriever configured with top_k=%d", TOP_K)

# ========== Prompt template (with citation placeholders) ==========
prompt_template_text = load_prompt_template()
prompt = ChatPromptTemplate.from_template(prompt_template_text)

# "stuff" chain: inject retrieved documents directly into the context
combine_chain = create_stuff_documents_chain(llm, prompt)

# QA chain: retriever -> LLM
qa_chain = create_retrieval_chain(retriever, combine_chain)


def _reference_label(document: Document) -> str:
    meta = document.metadata or {}
    label = meta.get("display_label")
    if isinstance(label, str):
        cleaned_label = _collapse_whitespace(label).strip()
        if cleaned_label:
            return cleaned_label

    metadata_label = _metadata_label(meta)
    if metadata_label:
        return metadata_label

    content = meta.get("_original_excerpt") or document.page_content or ""
    for line in content.splitlines():
        cleaned = _collapse_whitespace(line).strip()
        if cleaned:
            return _trim_snippet(cleaned)

    return "Excerpt"


def pretty_sources(source_docs):
    entries = []
    seen = set()
    for i, d in enumerate(source_docs, 1):
        src = d.metadata.get("source")
        # Estimate approximate location for context citations
        pos = []
        if "page_number" in d.metadata:
            pos.append(f"page={d.metadata['page_number']}")
        if "element_id" in d.metadata:
            pos.append(f"id={d.metadata['element_id']}")
        if "start_index" in d.metadata and "end_index" in d.metadata:
            pos.append(f"char={d.metadata['start_index']}-{d.metadata['end_index']}")
        key = (src, tuple(pos))
        if key in seen:
            continue
        seen.add(key)
        filename = Path(src).name if src else "unknown"
        location_meta = d.metadata.get("reference_location")
        location = location_meta or ", ".join(pos)
        pos_suffix = f"; {location}" if location else ""
        label = _reference_label(d)
        entries.append(f"- {label} ({filename}{pos_suffix})")
    return "\n".join(entries)

def ask(q: str):
    logger.info("Invoking QA chain for query: %s", q)
    res = qa_chain.invoke({"input": q})
    context_docs = res.get("context", [])
    docs_for_refs = [d for d in context_docs if isinstance(d, Document)] if isinstance(context_docs, list) else []
    answer = res.get("answer", "")
    logger.info("QA chain completed; received %d source document(s)", len(docs_for_refs))
    logger.info("Question: %s", q)
    logger.info("Answer: %s", answer)
    if docs_for_refs:
        logger.info("References:\n%s", pretty_sources(docs_for_refs))
    else:
        logger.info("References unavailable from retrieval context")


def interactive_loop():
    logger.info("Entering interactive QA loop. Type 'exit' or 'quit' to stop.")
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
