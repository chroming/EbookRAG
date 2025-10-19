import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from rag_langchain import (  # type: ignore[attr-defined]
    _grade_documents_with_llm,
    llm,
    pretty_sources,
    retriever,
)

logger = logging.getLogger(__name__)


def _format_context(docs: Sequence[Document]) -> str:
    if not docs:
        return (
            "Context:\n\n"
            "No relevant excerpts were found for this query.\n\n"
            "References:\n- None"
        )
    context_blocks = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    references = pretty_sources(docs)
    return f"Context:\n\n{context_blocks}\n\nReferences:\n{references}"


@tool
def search_epubs(question: str) -> str:
    """Retrieve relevant EPUB excerpts for the given question."""
    logger.debug("Retrieving docs for question: %s", question)
    retrieved_docs = retriever.invoke(question) or []
    filtered_docs = _grade_documents_with_llm(question, retrieved_docs)

    logger.info(
        "search_epubs retained %d/%d chunks",
        len(filtered_docs),
        len(retrieved_docs),
    )
    return _format_context(filtered_docs)


SYSTEM_PROMPT = (
    "You are a knowledgeable assistant answering questions about a curated EPUB library. "
    "You MUST call the `search_epubs` tool before you attempt an answer. "
    "Base all conclusions strictly on the tool output.\n\n"
    "After receiving tool results:\n"
    "1) Present the key steps or findings as a concise list.\n"
    "2) If the tool reports no context, reply with exactly "
    '"No clear evidence found in the provided context.".\n'
    "3) End every response with a section titled **References** and reuse the bullet list "
    "returned by the tool. Do not add new references or alter file names.\n"
)

rag_agent = create_agent(
    llm,
    tools=[search_epubs],
    system_prompt=SYSTEM_PROMPT,
    name="epub_rag_agent",
    checkpointer=InMemorySaver()
)


def answer_question(question: str) -> str:
    """Run the RAG agent for a single user query and return the assistant's reply."""
    result = rag_agent.invoke({"messages": [("user", question)]}, config={"configurable": {"thread_id": "1"}})
    messages: Sequence[BaseMessage] = result.get("messages", [])
    for message in reversed(messages):
        logger.info(f"Message from agent {message}")
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.content
    raise RuntimeError("Agent did not produce a final AI message.")


def interactive_loop() -> None:
    """Simple CLI loop for chatting with the agent."""
    logger.info("Entering agent chat loop. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("Enter question (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # ensure clean newline
            logger.info("Interactive session terminated by user.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            logger.info("Exiting agent chat loop.")
            break

        try:
            response = answer_question(user_input)
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent failed: %s", exc)
            continue
        print(response)


if __name__ == "__main__":
    interactive_loop()
