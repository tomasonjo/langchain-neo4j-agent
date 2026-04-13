"""FastAPI server for the LangChain Neo4j agent."""

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import run_agent, run_agent_stream

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
        async for token in run_agent_stream(
            user_message=request.message,
            session_id=session_id,
            user_id=request.user_id,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
