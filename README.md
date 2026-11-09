# Telecom FAQ Assistant with Amazon Bedrock AgentCore

A LangGraph-powered assistant that answers telecom questions from a CSV FAQ. It indexes FAQ entries with Hugging Face sentence embeddings and FAISS, provides semantic search, expanded search, and query reformulation tools, and runs behind Amazon Bedrock AgentCore Runtime. AgentCore Memory preserves conversation state by actor and session and stores user context for retrieval across sessions. The FAQ data file is excluded from this repository; provide your own data locally.

The language model provider in this implementation is Groq (`openai/gpt-oss-20b` by default). AgentCore provides the runtime and memory services. Configure Groq credentials and AWS credentials separately.

## Capabilities

- Semantic retrieval over a local FAQ CSV with FAISS and `sentence-transformers/all-MiniLM-L6-v2`.
- Three agent tools: `search_faq`, `search_detailed_faq`, and `reformulate_query`.
- Amazon Bedrock AgentCore Runtime HTTP entrypoint accepting a `prompt` and optional `actor_id` / `session_id`.
- AgentCore Memory checkpointing for session continuity and a memory store for cross-session context.
- Responses grounded in FAQ results, with an explicit fallback when the knowledge base lacks an answer.

## Requirements

- Python 3.13 (as declared in `pyproject.toml`) and [uv](https://docs.astral.sh/uv/).
- A Groq API key.
- AWS credentials with permission to access the configured AgentCore Memory and to deploy/invoke AgentCore Runtime.
- An existing AgentCore Memory resource in the AWS region configured by `AWS_REGION`.

## Setup

```bash
uv sync
cp .sample_env .env
```

Edit `.env` and set:

```dotenv
GROQ_API_KEY=your_groq_api_key
AGENTCORE_MEMORY_ID=your_agentcore_memory_id
AWS_REGION=ap-southeast-2
GROQ_MODEL=openai/gpt-oss-20b
```

AWS credentials should be available through the standard AWS SDK credential chain (for example `aws configure`, a profile, or an IAM role). The embedding model is downloaded the first time the application starts. `FAQ_CSV_PATH` and `EMBEDDING_MODEL` can optionally override the defaults.

Provide a UTF-8 CSV with `question` and `answer` columns at `lauki_qna.csv`, or set `FAQ_CSV_PATH` to another local CSV path. FAQ data files are ignored by Git and are not included in this repository.

## Run locally

Start the AgentCore-compatible local server:

```bash
uv run python main.py
```

In another terminal, invoke it:

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"How do I activate a new SIM?","actor_id":"demo-user","session_id":"demo-session"}'
```

The `actor_id` identifies a user. Keep `session_id` the same across turns in one conversation so AgentCore Memory can restore that session's conversation state. Use a new session ID to start another conversation for the same actor.

## Deploy to AgentCore Runtime

Install the [AgentCore CLI](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-code-deploy-python.html), configure AWS credentials and permissions, then prepare and deploy this code as an AgentCore Python project with `main.py` as its entrypoint. The runtime needs the environment variables above and outbound access to Groq and the Hugging Face model host. Set the secret `GROQ_API_KEY` using your deployment's supported secret configuration; do not commit it.

For direct-code deployment using the AgentCore CLI, the documented workflow is:

```bash
agentcore dev --no-browser
agentcore deploy
agentcore invoke "How do I activate a new SIM?"
```

The CLI project configuration may need to be initialized for your AWS account and region before deployment. AgentCore Memory must already exist, and the runtime role must be allowed to use it (`bedrock-agentcore:CreateEvent`, `bedrock-agentcore:ListEvents`, and `bedrock-agentcore:RetrieveMemories`) as well as invoke the configured Groq service over the network.

## Project files

- `main.py` - AgentCore Runtime entrypoint, FAQ index, retrieval tools, and memory integration.
- Local FAQ CSV - not stored in this repository; provide `question` and `answer` columns as described above.
- `00_langgraph_agent.py`, `01_agentcore_runtime.py`, `02_agentcore_memory.py` - earlier learning-stage examples retained for reference.
- `pyproject.toml` / `uv.lock` - Python project dependencies.

## Notes

- Supply FAQ data with approved company answers before using the assistant for real customer support.
- The assistant uses Groq for generation; it does not invoke an Amazon Bedrock foundation model. The hosting and memory services are Amazon Bedrock AgentCore.
- Session history and cross-session memory depend on AgentCore Memory being configured, reachable, and authorized in the selected AWS region.
