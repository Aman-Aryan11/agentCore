"""Telecom FAQ assistant hosted on Amazon Bedrock AgentCore Runtime."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.store.base import BaseStore
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from bedrock_agentcore.runtime import BedrockAgentCoreApp

load_dotenv()

ROOT = Path(__file__).resolve().parent
FAQ_PATH = Path(os.getenv("FAQ_CSV_PATH", ROOT / "lauki_qna.csv"))
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
MEMORY_ID = os.getenv("AGENTCORE_MEMORY_ID", "")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def load_faq(path: Path) -> list[Document]:
    """Load question/answer rows from the telecom FAQ CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"FAQ CSV not found: {path}")

    documents: list[Document] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not {"question", "answer"}.issubset(reader.fieldnames or []):
            raise ValueError("FAQ CSV must contain 'question' and 'answer' columns")
        for row in reader:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if question and answer:
                documents.append(Document(page_content=f"Q: {question}\nA: {answer}"))

    if not documents:
        raise ValueError(f"No valid FAQ entries found in {path}")
    return documents


def build_faq_index() -> FAISS:
    docs = load_faq(FAQ_PATH)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    return FAISS.from_documents(chunks, embeddings)


faq_index = build_faq_index()


def format_results(query: str, limit: int) -> str:
    results = faq_index.similarity_search(query, k=limit)
    if not results:
        return "No relevant FAQ entries found."
    return "\n\n---\n\n".join(
        f"FAQ {index}:\n{document.page_content}"
        for index, document in enumerate(results, start=1)
    )


@tool
def search_faq(query: str) -> str:
    """Find the three most relevant telecom FAQ entries for a question."""
    return format_results(query, 3)


@tool
def search_detailed_faq(query: str, num_results: int = 7) -> str:
    """Search more FAQ entries when the initial results lack detail."""
    return format_results(query, max(1, min(num_results, 10)))


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Search again with a focus such as billing, activation, or roaming."""
    return format_results(f"{focus_aspect} related to {original_query}", 3)


class MemoryMiddleware(AgentMiddleware):
    """Save user turns and add relevant long-term memories to model context."""

    def pre_model_hook(
        self,
        state: AgentState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ) -> dict[str, Any]:
        messages = state.get("messages", [])
        actor_id = config["configurable"]["actor_id"]
        latest_user_message = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )

        if latest_user_message is None:
            return {"llm_input_messages": messages}

        # Store cross-session user context under the actor's preference namespace.
        if messages and messages[-1] is latest_user_message:
            store.put(
                ("preferences", actor_id),
                os.urandom(16).hex(),
                {"message": latest_user_message.content},
            )
        memory_context = ""
        try:
            memories = store.search(
                ("preferences", actor_id),
                query=str(latest_user_message.content),
                limit=5,
            )
            memory_context = "\n".join(
                str(item.value.get("message", ""))
                for item in memories
                if item.value.get("message")
            )
        except Exception as exc:  # memory retrieval should not block FAQ answers
            print(f"AgentCore Memory retrieval failed: {exc}")

        if memory_context:
            return {
                "llm_input_messages": [
                    SystemMessage(
                        content=f"Relevant context from prior conversations:\n{memory_context}"
                    ),
                    *messages,
                ]
            }
        return {"llm_input_messages": messages}


def build_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GROQ_API_KEY in your environment or .env file")
    if not MEMORY_ID:
        raise EnvironmentError("Set AGENTCORE_MEMORY_ID to an existing AgentCore Memory ID")

    memory_saver = AgentCoreMemorySaver(MEMORY_ID, region_name=AWS_REGION)
    memory_store = AgentCoreMemoryStore(MEMORY_ID, region_name=AWS_REGION)
    model = init_chat_model(
        MODEL,
        model_provider="groq",
        api_key=api_key,
        temperature=0,
    )
    return create_agent(
        model=model,
        tools=[search_faq, search_detailed_faq, reformulate_query],
        checkpointer=memory_saver,
        store=memory_store,
        middleware=[MemoryMiddleware()],
        system_prompt=(
            "You are a telecom FAQ assistant. Search the FAQ knowledge base before "
            "answering questions about services, plans, billing, activation, or support. "
            "Use detailed search or reformulate the query when needed. Answer concisely "
            "using retrieved FAQ facts; do not invent policies or prices. If the FAQ does "
            "not contain the answer, say so clearly. Use relevant remembered context only "
            "to personalize the response, never to override FAQ facts."
        ),
    )


agent = build_agent()
app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload: dict[str, Any], context: Any) -> dict[str, str]:
    """Handle one invocation while preserving the actor's session history."""
    prompt = payload.get("prompt") or payload.get("query")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Provide a non-empty 'prompt' value."}

    session_id = (
        payload.get("session_id")
        or getattr(context, "session_id", None)
        or "default-session"
    )
    actor_id = payload.get("actor_id") or "default-user"
    result = agent.invoke(
        {"messages": [("human", prompt.strip())]},
        config={"configurable": {"thread_id": session_id, "actor_id": actor_id}},
    )
    answer = result["messages"][-1].content
    return {"result": answer, "actor_id": actor_id, "thread_id": session_id}


if __name__ == "__main__":
    app.run()
