"""LangChain agent with Neo4j MCP tools and Neo4j Agent Memory."""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.integrations.langchain import Neo4jAgentMemory

from app.memory_tools import create_memory_tools
from app.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    NEO4J_MEMORY_URI,
    NEO4J_MEMORY_USERNAME,
    NEO4J_MEMORY_PASSWORD,
    OPENAI_MODEL,
)


def get_mcp_config() -> dict:
    """Build MCP server config using env vars."""
    return {
        "neo4j": {
            "command": "uvx",
            "args": ["neo4j-mcp-server"],
            "env": {
                "NEO4J_URI": NEO4J_URI,
                "NEO4J_USERNAME": NEO4J_USERNAME,
                "NEO4J_PASSWORD": NEO4J_PASSWORD,
                "NEO4J_DATABASE": NEO4J_DATABASE,
            },
            "transport": "stdio",
        }
    }


def get_memory_settings() -> MemorySettings:
    return MemorySettings(
        neo4j={
            "uri": NEO4J_MEMORY_URI,
            "username": NEO4J_MEMORY_USERNAME,
            "password": NEO4J_MEMORY_PASSWORD,
        },
        extraction={
            "extractor_type": "gliner",
        },
    )


SYSTEM_PROMPT = """You are a helpful assistant with access to a Neo4j graph database via MCP tools \
and a persistent memory system.

## Neo4j Database Tools
Use the Neo4j MCP tools to read the schema, run Cypher queries, and retrieve data.

## Memory Tools
You have tools to interact with the user's memory/context graph:
- **search_memory_entities**: Search the knowledge graph for entities
- **get_user_preferences**: Retrieve stored user preferences to personalize responses
- **save_user_preference**: Save new preferences when the user expresses likes/dislikes
- **search_conversation_history**: Search past conversation messages for context

## Memory Context
{memory_context}

## Cypher Query Guidelines
- Use the Neo4j tools to explore the database schema first if you're unsure about the structure.
- Write efficient Cypher queries.
- Do NOT use SQL syntax in Cypher. Common mistakes to avoid:
  - No `NULLS LAST` or `NULLS FIRST` — these are not valid Cypher. If you need to handle nulls \
in ordering, use `CASE WHEN prop IS NULL THEN 1 ELSE 0 END` as a secondary sort key, or filter \
nulls with a `WHERE prop IS NOT NULL` clause before ordering.
  - No `GROUP BY` — use aggregation functions directly in RETURN (e.g., `RETURN label, count(*)`).
  - No `HAVING` — use `WITH` + `WHERE` after aggregation instead.
  - No `SELECT`, `FROM`, `JOIN` — use `MATCH` patterns instead.

## General Guidelines
- When users express preferences or important facts, use save_user_preference to remember them.
- Use get_user_preferences and search_memory_entities to personalize your responses.
- Be concise and helpful.
"""


def _create_agent_memory(
    memory_client: MemoryClient,
    session_id: str,
) -> Neo4jAgentMemory:
    """Create a Neo4jAgentMemory instance for the given session."""
    return Neo4jAgentMemory(
        memory_client=memory_client,
        session_id=session_id,
        include_short_term=True,
        include_long_term=True,
        include_reasoning=True,
        max_messages=10,
        max_preferences=10,
        max_traces=3,
    )


async def get_memory_context(
    memory: Neo4jAgentMemory,
    current_message: str,
) -> str:
    """Retrieve memory context using Neo4jAgentMemory integration."""
    try:
        variables = await memory._load_memory_variables_async(
            {"input": current_message}
        )
    except Exception:
        return "No prior context available."

    parts = []

    if variables.get("history"):
        parts.append("### Recent Conversation History\n" + variables["history"])

    if variables.get("context"):
        parts.append("### Memory Context\n" + variables["context"])

    if variables.get("preferences"):
        pref_lines = [
            f"  - [{p['category']}] {p['preference']}"
            for p in variables["preferences"]
        ]
        if pref_lines:
            parts.append("### User Preferences\n" + "\n".join(pref_lines))

    if variables.get("similar_tasks"):
        parts.append("### Similar Past Tasks\n" + variables["similar_tasks"])

    if not parts:
        return "No prior context available."

    return "\n\n".join(parts)


async def store_interaction(
    memory: Neo4jAgentMemory,
    user_id: str,
    user_message: str,
    assistant_message: str,
    tool_uses: list[dict] | None = None,
) -> None:
    """Store the interaction in memory (short-term + long-term extraction)."""
    # Store short-term messages directly so we can attach user_id metadata.
    # Neo4jAgentMemory._save_context_async does not support metadata, so we
    # call the underlying client to preserve user attribution.
    user_metadata = {"user_id": user_id}
    await memory.memory_client.short_term.add_message(
        session_id=memory.session_id,
        role="user",
        content=user_message,
        metadata=user_metadata,
    )

    assistant_metadata: dict = {"user_id": user_id}
    if tool_uses:
        assistant_metadata["tool_uses"] = tool_uses

    await memory.memory_client.short_term.add_message(
        session_id=memory.session_id,
        role="assistant",
        content=assistant_message,
        metadata=assistant_metadata,
    )

    # Long-term extraction (entities, preferences) is not handled by the
    # integration, so we call it directly on the memory client.
    try:
        await memory.memory_client.long_term.extract_and_store(
            text=f"User: {user_message}\nAssistant: {assistant_message}",
            user_id=user_id,
            session_id=memory.session_id,
        )
    except Exception:
        pass


def _collect_tool_uses(messages: list) -> list[dict]:
    """Extract tool call/result pairs from agent messages."""
    tool_uses = []
    tool_calls_by_id: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_by_id[tc.get("id", "")] = tc
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            tc = tool_calls_by_id.get(msg.tool_call_id)
            if tc:
                result_preview = msg.content
                if not isinstance(result_preview, str):
                    result_preview = json.dumps(result_preview, indent=2)
                if len(result_preview) > 500:
                    result_preview = result_preview[:500] + "..."
                tool_uses.append({
                    "name": tc["name"],
                    "args": tc["args"],
                    "result": result_preview,
                })
    return tool_uses


async def store_decision_trace(
    memory: Neo4jAgentMemory,
    user_id: str,
    user_message: str,
    assistant_message: str,
    tool_uses: list[dict],
) -> None:
    """Store a reasoning/decision trace when tools were called."""
    reasoning = memory.memory_client.reasoning

    trace = await reasoning.start_trace(
        session_id=memory.session_id,
        task=user_message,
        metadata={"user_id": user_id},
    )

    for tool_use in tool_uses:
        step = await reasoning.add_step(
            trace_id=trace.id,
            action=f"Called {tool_use['name']}",
            observation=tool_use.get("result", ""),
        )
        await reasoning.record_tool_call(
            step_id=step.id,
            tool_name=tool_use["name"],
            arguments=tool_use.get("args", {}),
            result=tool_use.get("result"),
        )

    await reasoning.complete_trace(
        trace_id=trace.id,
        outcome=assistant_message[:500] if assistant_message else None,
        success=True,
    )


async def run_agent(
    user_message: str,
    session_id: str,
    user_id: str = "default-user",
) -> str:
    """Run the LangChain agent with Neo4j MCP tools and memory."""
    memory_settings = get_memory_settings()
    memory_client = MemoryClient(memory_settings)
    await memory_client.connect()

    try:
        memory = _create_agent_memory(memory_client, session_id)
        memory_context = await get_memory_context(memory, user_message)

        mcp_client = MultiServerMCPClient(get_mcp_config())
        mcp_tools = await mcp_client.get_tools()
        memory_tools = create_memory_tools(memory_client, user_id=user_id)
        tools = mcp_tools + memory_tools

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        system_message = SYSTEM_PROMPT.format(memory_context=memory_context)

        agent = create_agent(llm, tools=tools, system_prompt=system_message)

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message)]}
        )

        ai_messages = [
            m for m in result["messages"] if isinstance(m, AIMessage)
        ]
        response = ai_messages[-1].content if ai_messages else "No response."

        tool_uses = _collect_tool_uses(result["messages"])
        await store_interaction(
            memory, user_id, user_message, response,
            tool_uses=tool_uses or None,
        )
        if tool_uses:
            try:
                await store_decision_trace(
                    memory, user_id, user_message, response, tool_uses
                )
            except Exception:
                pass

        return response
    finally:
        await memory_client.close()


async def run_agent_stream(
    user_message: str,
    session_id: str,
    user_id: str = "default-user",
):
    """Stream the agent response with tool usage events.

    Yields JSON-encoded SSE events:
      {"type": "token", "content": "..."}
      {"type": "tool_call", "name": "...", "args": {...}}
      {"type": "tool_result", "name": "...", "content": "..."}
      {"type": "done"}
    """
    memory_settings = get_memory_settings()
    memory_client = MemoryClient(memory_settings)
    await memory_client.connect()

    try:
        memory = _create_agent_memory(memory_client, session_id)
        memory_context = await get_memory_context(memory, user_message)

        mcp_client = MultiServerMCPClient(get_mcp_config())
        mcp_tools = await mcp_client.get_tools()
        memory_tools = create_memory_tools(memory_client, user_id=user_id)
        tools = mcp_tools + memory_tools

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        system_message = SYSTEM_PROMPT.format(memory_context=memory_context)

        agent = create_agent(llm, tools=tools, system_prompt=system_message)

        full_response = ""
        seen_tool_calls = set()
        seen_tool_results = set()
        chunk = None

        try:
            async for chunk in agent.astream(
                {"messages": [HumanMessage(content=user_message)]},
                stream_mode="values",
            ):
                messages = chunk["messages"]
                latest = messages[-1]

                # Emit tool calls from AIMessage
                if isinstance(latest, AIMessage) and latest.tool_calls:
                    for tc in latest.tool_calls:
                        tc_id = tc.get("id", "")
                        if tc_id not in seen_tool_calls:
                            seen_tool_calls.add(tc_id)
                            yield json.dumps({
                                "type": "tool_call",
                                "id": tc_id,
                                "name": tc["name"],
                                "args": tc["args"],
                            })

                # Emit tool results from ToolMessage
                if isinstance(latest, ToolMessage):
                    if latest.id not in seen_tool_results:
                        seen_tool_results.add(latest.id)
                        content = latest.content
                        if not isinstance(content, str):
                            content = json.dumps(content, indent=2)
                        if len(content) > 2000:
                            content = content[:2000] + "... (truncated)"
                        yield json.dumps({
                            "type": "tool_result",
                            "tool_call_id": latest.tool_call_id or "",
                            "name": latest.name or "",
                            "content": content,
                        })

                # Stream AI text tokens
                if isinstance(latest, AIMessage) and latest.content:
                    new_content = latest.content[len(full_response):]
                    if new_content:
                        full_response = latest.content
                        yield json.dumps({
                            "type": "token",
                            "content": new_content,
                        })
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": str(e),
            })
            return

        # Collect tool usage for metadata storage
        tool_uses = _collect_tool_uses(
            chunk["messages"] if chunk is not None else []
        )

        await store_interaction(
            memory, user_id, user_message, full_response,
            tool_uses=tool_uses or None,
        )
        if tool_uses:
            try:
                await store_decision_trace(
                    memory, user_id, user_message, full_response, tool_uses
                )
            except Exception:
                pass
    finally:
        await memory_client.close()
