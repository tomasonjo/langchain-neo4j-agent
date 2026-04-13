"""FastAPI server for the LangChain Neo4j agent."""

from datetime import datetime
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from neo4j_agent_memory import MemoryClient, MemorySettings

from app.agent import run_agent, run_agent_stream, get_memory_settings

app = FastAPI(title="LangChain Neo4j Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str = "default-user"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    message_count: int = 0
    first_message_preview: str | None = None
    last_message_preview: str | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    response = await run_agent(
        user_message=request.message,
        session_id=session_id,
        user_id=request.user_id,
    )
    return ChatResponse(response=response, session_id=session_id)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    async def event_generator():
        async for event_json in run_agent_stream(
            user_message=request.message,
            session_id=session_id,
            user_id=request.user_id,
        ):
            yield f"data: {event_json}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    order_by: Literal["created_at", "updated_at", "message_count"] = "updated_at",
):
    """List all sessions sorted by time descending."""
    memory_settings = get_memory_settings()
    memory_client = MemoryClient(memory_settings)
    await memory_client.connect()
    try:
        sessions = await memory_client.short_term.list_sessions(
            limit=limit,
            order_by=order_by,
            order_dir="desc",
        )
        return [
            SessionResponse(
                session_id=s.session_id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=s.message_count,
                first_message_preview=s.first_message_preview,
                last_message_preview=s.last_message_preview,
            )
            for s in sessions
        ]
    finally:
        await memory_client.close()


@app.get("/users", response_model=list[str])
async def list_users():
    """List distinct user IDs from message metadata."""
    memory_settings = get_memory_settings()
    memory_client = MemoryClient(memory_settings)
    await memory_client.connect()
    try:
        results = await memory_client.short_term._client.execute_read(
            """
            MATCH (c:Conversation)-[:HAS_MESSAGE]->(m:Message)
            WHERE m.metadata IS NOT NULL AND m.metadata CONTAINS 'user_id'
            WITH apoc.convert.fromJsonMap(m.metadata) AS meta
            RETURN DISTINCT meta.user_id AS user_id
            ORDER BY user_id
            """,
            {},
        )
        return [r["user_id"] for r in results if r.get("user_id")]
    except Exception:
        return []
    finally:
        await memory_client.close()


class ToolUse(BaseModel):
    name: str
    args: dict = {}
    result: str | None = None


class MessageResponse(BaseModel):
    role: str
    content: str
    tool_uses: list[ToolUse] = []


@app.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_session_messages(session_id: str):
    """Get conversation history for a session."""
    memory_settings = get_memory_settings()
    memory_client = MemoryClient(memory_settings)
    await memory_client.connect()
    try:
        conversation = await memory_client.short_term.get_conversation(
            session_id=session_id,
        )
        results = []
        for msg in conversation.messages:
            if msg.role.value not in ("user", "assistant"):
                continue
            tool_uses = []
            if msg.metadata and "tool_uses" in msg.metadata:
                tool_uses = msg.metadata["tool_uses"]
            results.append(
                MessageResponse(
                    role=msg.role.value,
                    content=msg.content,
                    tool_uses=tool_uses,
                )
            )
        return results
    except Exception:
        return []
    finally:
        await memory_client.close()


@app.get("/health")
async def health():
    return {"status": "ok"}
