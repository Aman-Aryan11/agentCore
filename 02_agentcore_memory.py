import csv
import os
import uuid
from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.store.base import BaseStore

# Import AgentCore runtime and memory integrations
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from dotenv import load_dotenv

_ = load_dotenv()

app = BedrockAgentCoreApp()
# AgentCore Memory Configuration
REGION = "ap-southeast-2"
MEMORY_ID = "lauki_agent_memory-Yrm3JrG0Vz"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize memory components
checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID)
store = AgentCoreMemoryStore(memory_id=MEMORY_ID)


def load_faq_csv(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))
    return docs


docs = load_faq_csv("./lauki_qna.csv")
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_documents(docs)
faq_store = FAISS.from_documents(chunks, emb)


@tool
def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for relevant information.
    Use this tool when the user asks questions about products, services, or policies.

    Args:
        query: The search query to find relevant FAQ entries

    Returns:
        Relevant FAQ entries that might answer the question
    """
    results = faq_store.similarity_search(query, k=3)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def search_detailed_faq(query: str, num_results: int = 5) -> str:
    """Search the FAQ knowledge base with more results for complex queries.
    Use this when the initial search doesn't provide enough information.

    Args:
        query: The search query
        num_results: Number of results to retrieve (default: 5)

    Returns:
        More comprehensive FAQ entries
    """
    results = faq_store.similarity_search(query, k=num_results)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Reformulate the query to focus on a specific aspect.
    Use this when you need to search for a different angle of the question.

    Args:
        original_query: The original user question
        focus_aspect: The specific aspect to focus on (e.g., "pricing", "activation", "troubleshooting")

    Returns:
        A reformulated query focused on the specified aspect
    """
    reformulated = f"{focus_aspect} related to {original_query}"
    results = faq_store.similarity_search(reformulated, k=3)

    if not results:
        return f"No results found for aspect: {focus_aspect}"

    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Results for '{focus_aspect}' aspect:\n\n{context}"


tools = [search_faq, search_detailed_faq, reformulate_query]


class MemoryMiddleware(AgentMiddleware):
    # Pre-model hook: saves messages and retrieves long-term memories
    def pre_model_hook(self, state: AgentState, config: RunnableConfig, *, store: BaseStore):
        """
        Hook that runs before LLM invocation to:
        1. Save the latest human message to long-term memory
        2. Retrieve relevant user preferences and memories
        3. Append memories to the context
        """
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]

        # Namespace for this specific session
        namespace = (actor_id, thread_id)
        messages = state.get("messages", [])

        # Save the last human message to long-term memory
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})

                # OPTIONAL: Retrieve user preferences from long-term memory
                # Search across all sessions for this actor
                user_preferences_namespace = ("preferences", actor_id)
                try:
                    preferences = store.search(
                        user_preferences_namespace,
                        query=msg.content,
                        limit=5
                    )

                    # If we found relevant memories, add them to the context
                    if preferences:
                        memory_context = "\n".join([
                            f"Memory: {item.value.get('message', '')}"
                            for item in preferences
                        ])
                        # You can append this to the messages or use it another way
                        print(f"Retrieved memories: {memory_context}")
                except Exception as e:
                    print(f"Memory retrieval error: {e}")
                break

        return {"messages": messages}


    # OPTIONAL: Post-model hook to save AI responses
    def post_model_hook(state, config: RunnableConfig, *, store: BaseStore):
        """
        Hook that runs after LLM invocation to save AI messages to long-term memory
        """
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]
        namespace = (actor_id, thread_id)

        messages = state.get("messages", [])

        # Save the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})
                break

        return state


# Initialize the LLM
llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    api_key=GROQ_API_KEY
)

system_prompt = """You are a helpful FAQ assistant with access to a knowledge base and user memory.

Your goal is to answer user questions accurately using the available tools while remembering user preferences.

Guidelines:
1. Check if you have relevant user preferences or history from previous conversations
2. Use the search_faq tool to find relevant information from the knowledge base
3. If the query is complex, use reformulate_query to search different aspects
4. Personalize responses based on user preferences when relevant
5. Always provide a clear, concise answer based on the retrieved information
6. If you cannot find relevant information, clearly state that

Think step-by-step and use tools strategically to provide the best answer."""

# Create the agent with memory configurations
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    store=store,
    middleware=[MemoryMiddleware()],
    system_prompt=system_prompt,
)


# AgentCore Entrypoint
@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation in AgentCore runtime with memory support"""
    print("Received payload:", payload)
    print("Context:", context)

    # Extract query from payload
    query = payload.get("prompt", "No prompt found in input")

    # Extract or generate actor_id and thread_id
    actor_id = payload.get("actor_id", "default-user")
    thread_id = payload.get("thread_id", payload.get("session_id", "default-session"))

    # Configure memory context
    config = {
        "configurable": {
            "thread_id": thread_id,  # Maps to AgentCore session_id
            "actor_id": actor_id     # Maps to AgentCore actor_id
        }
    }

    # Invoke the agent with memory
    result = agent.invoke(
        {"messages": [("human", query)]},
        config=config
    )

    print("Result:", result)

    # Extract the final answer from the result
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "No response generated"

    # Return the answer
    return {
        "result": answer,
        "actor_id": actor_id,
        "thread_id": thread_id
    }


if __name__ == "__main__":
    app.run()
