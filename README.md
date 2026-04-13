# langchain-neo4j-agent

Chat application using LangChain with Neo4j MCP for graph database retrieval and Neo4j Agent Memory for persistent conversation history and user preferences.

## Architecture

```
┌──────────────┐       ┌──────────────────┐       ┌─────────────────────┐
│  Streamlit   │──SSE──│  FastAPI Backend  │──MCP──│  Retrieval Neo4j    │
│  Frontend    │       │  (LangChain)      │       │  (external, yours)  │
│  :8501       │       │  :8000            │       │                     │
└──────────────┘       └────────┬──────────┘       └─────────────────────┘
                                │
                          neo4j-agent-memory
                                │
                       ┌────────┴──────────┐
                       │  Memory Neo4j     │
                       │  (Docker, :7688)  │
                       └───────────────────┘
```

- **Retrieval Neo4j** — your existing database, queried via Neo4j MCP (stdio). Not in Docker.
- **Memory Neo4j** — containerized, stores conversation history + user preferences.

## Quick Start

```bash
# 1. Configure env
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and your retrieval Neo4j connection

# 2. Run everything
docker compose up --build
```

Open http://localhost:8501 to chat.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `NEO4J_URI` | Retrieval Neo4j URI (your external DB) |
| `NEO4J_USERNAME` | Retrieval Neo4j username |
| `NEO4J_PASSWORD` | Retrieval Neo4j password |

Memory Neo4j is auto-configured in `docker-compose.yml` — no env vars needed.

## Local Development (without Docker)

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Terminal 1: backend
cd backend && uvicorn app.main:app --reload

# Terminal 2: frontend
streamlit run frontend/streamlit_app.py
```

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── config.py       # Environment configuration
│   │   ├── memory.py       # Neo4j Agent Memory ↔ LangChain adapter
│   │   ├── agent.py        # LangChain ReAct agent (MCP tools + memory)
│   │   └── main.py         # FastAPI endpoints
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml      # BE + FE + Memory Neo4j
└── .env.example
```
