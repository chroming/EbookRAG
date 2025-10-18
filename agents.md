# EbookRAG Agent Brief

## 项目速览 / Project Snapshot
EbookRAG 是一个针对本地 EPUB 藏书的检索增强生成（RAG）工具包，通过 LangChain+LangGraph 与 LlamaIndex 两条脚本化问答路径提供同等能力。两套脚本共享同一 `.env` 配置、EPUB 目录和 Pandoc 自动安装流程，默认使用 `intfloat/multilingual-e5-base` 向量模型，并调用任意 OpenAI 兼容聊天模型生成答案。典型迭代只需改动 `rag_langchain.py` 或 `rag_llamaindex.py` 中的单文件流程。

## 资源索引 / Repo Atlas
- `epubs/`：默认 EPUB 输入目录，可在 `.env` 中使用 `EPUB_DIR` 覆盖。
- `vector_store_langchain/`、`vector_store_llamaindex/`：各自的 FAISS 索引与 `manifest.json` 指纹，用于增量更新。
- `.env` / `.env.example`：集中保存 LLM、嵌入、切分、LangGraph 提示等配置，脚本启动时通过 `python-dotenv` 自动加载。
- `rag_langchain.py`、`rag_llamaindex.py`：两个入口脚本封装全部索引与问答逻辑；需要共享逻辑时再拆分公共模块（别忘了同步 `pyproject.toml`）。

## 环境与运行手册 / Setup & Runbook
```bash
# 初始化虚拟环境与依赖
uv venv
source .venv/bin/activate
uv sync --extra langchain     # 或 uv sync --extra llamaindex

# 运行交互式问答
python rag_langchain.py       # LangChain + LangGraph
python rag_llamaindex.py      # LlamaIndex
```
确保 Pandoc 可用；两个脚本都会在检测缺失时通过 `pypandoc.download_pandoc()` 自动安装，需要网络访问以及 `pypandoc` 依赖。

## 关键配置旋钮 / Core Configuration Knobs
- **LLM**：`OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `OPENAI_API_MODEL`, `OPENAI_TEMPERATURE`（两条流水线共享）。
- **Embeddings**：`EMBED_PROVIDER`（`huggingface` 或 `ollama`）、`EMBED_MODEL_NAME`、`OLLAMA_BASE_URL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_EMBED_OPTIONS`（LangChain 路径会清洗空选项并缓存自定义参数）。
- **切分与检索**：`CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_SEPARATORS`, `TOP_K`, `VECTOR_EMBED_BATCH_SIZE`。
- **LangGraph 提示**：`QA_PROMPT_TEMPLATE`, `QA_PROMPT_TEMPLATE_PATH`, `LANGGRAPH_GRADER_SYSTEM_PROMPT`, `LANGGRAPH_GRADER_HUMAN_PROMPT`。
- **持久化目录**：`VECTOR_STORE_DIR`（LangChain）与 `LLAMAINDEX_VECTOR_STORE_DIR`（LlamaIndex）可通过 `.env` 覆写，用于自定义缓存位置。

## LangChain 流程 / LangChain Flow (rag_langchain.py)
1. 读取 `.env` 并初始化 LLM 与嵌入（含 Ollama 参数清洗和嵌入进度日志）。
2. 使用 `FAISS.load_local()` 加载或创建向量库，依据 EPUB `manifest.json` 指纹判定是否增量重建。
3. 调用 `UnstructuredEPubLoader` 抽取文本，`RecursiveCharacterTextSplitter` 切分片段并通过 `_annotate_documents()` 添加引用头。
4. 构建 LangGraph 状态机：`retrieve → grade_documents → answer`。检索器基于 `vectordb.as_retriever`, `_grade_documents_with_llm()` 用 LLM 过滤无关片段，`combine_chain` 生成最终回答并附带引用。
5. `_CONVERSATION_MEMORY` 维护最近若干轮问答，`pretty_sources()` 去重并格式化引用，`ask()`/`interactive_loop()` 提供调用入口。

## LlamaIndex 流程 / LlamaIndex Flow (rag_llamaindex.py)
1. 加载 `.env` 并创建 `Settings`（LLM、嵌入、`SentenceSplitter`、空回调管理器）；嵌入函数被包装以打印批量进度。
2. 通过 EPUB 指纹比较决定是复用现有 `FaissVectorStore` 还是重新构建；缺失或变化时会重新下载 Pandoc 并构建索引。
3. 如果需要重建：`SimpleDirectoryReader`+`UnstructuredReader` 解析 EPUB → `VectorStoreIndex.from_documents()` 写入新的 FAISS 向量库并持久化。
4. 通过 `VectorIndexRetriever` 与 `RetrieverQueryEngine` 暴露问答接口；`ask()` 打印答案与源节点范围，`interactive_loop()` 提供命令行会话。

## 协作提示 / Collaboration Notes
- 常见需求（新增清洗、改切分、换重排）可直接在对应脚本内调整；若演进到多脚本共享逻辑，建议抽取模块并在 `pyproject.toml` 中声明入口或 extras。
- LangChain pipeline 中引用格式逻辑集中在 `_annotate_documents()` 与 `pretty_sources()`；需要调整输出格式或引用策略时优先修改这里。
- 修改嵌入配置后脚本会自动重建索引；确保旧目录无残留文件，以免误判指纹或造成加载失败。
- 当前没有自动化测试；默认验证方式是运行脚本并确认日志（LangChain 为 DEBUG，LlamaIndex 为 INFO）。提交前尽量手动问答几轮观察引用与回答质量。
- 代码只需要兼容python3.10+, 如果有必要的话新增代码加上对应的类型注解。

> 本文件提供面向 AI 协作者的快速入门与改动指引。需要更详尽的环境说明时，请查阅 `README.md` 或目标脚本内注释。
