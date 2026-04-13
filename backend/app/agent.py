"""LangChain agent with Neo4j MCP tools and Neo4j Agent Memory."""

import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from neo4j_agent_memory import MemoryClient, MemorySettings

from app.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    NEO4J_MEMORY_URI,
    NEO4J_MEMORY_USERNAME,
    NEO4J_MEMORY_PASSWORD,
)
from app.memory import Neo4jChatMessageHistory


def get_mcp_config() -> dict:
    """Build MCP server config using env vars."""
    return {
        "neo4j": {
            "command": "npx",
            "args": ["-y", "@neo4j/mcp-neo4j"],
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
        }
    )


SYSTEM_PROMPT = """You are a helpful assistant with access to a Neo4j graph database via MCP tools.

You can query the database to answer user questions. Use the available Neo4j MCP tools
to read the schema, run Cypher queries, and retrieve data.

## Memory Context
{memory_context}

## Guidelines
- Use the Neo4j tools to explore the database schema first if you're unsure about the structure.
- Write efficient Cypher queries.
- When users express preferences or important facts about themselves, acknowledge them.
- Reference information from past conversations when relevant.
- Be concise and helpful.
"""


async def get_memory_context(
    memory_client: MemoryClient,
    session_id: str,
    user_id: str,
    current_message: str,
) -> str:
    """Retrieve memory context: conversation history, user facts, and preferences."""
    parts = []

    # Get conversation history (short-term)
    try:
        conversation = await memory_client.short_term.get_conversation(
            session_id=session_id,
        )
        if conversation.messages:
            history_lines = []
            for msg in conversation.messages[-10:]:  # Last 10 messages
                history_lines.append(f"  {msg.role}: {msg.content}")
            parts.append(
                "### Recent Conversation History\n" + "\n".join(history_lines)
            )
    except Exception:
        pass

    # Get user preferences (long-term)
    try:
        preferences = await memory_client.long_term.get_preferences(
            user_id=user_id,
        )
        if preferences:
            pref_lines = [
                f"  - [{p.category}] {p.preference}" for p in preferences
            ]
            parts.append(
                "### User Preferences\n" + "\n".join(pref_lines)
            )
    except Exception:
        pass

    # Search for relevant entities (long-term)
    try:
        entities = await memory_client.long_term.search_entities(
            query=current_message,
            limit=5,
        )
        if entities:
            entity_lines = [
                f"  - {e.name} ({e.type}): {e.description or 'N/A'}"
                for e in entities
            ]
            parts.append(
                "### Related Entities from Memory\n" + "\n".join(entity_lines)
            )
    except Exception:
        pass

    if not parts:
        return "No prior context available."

    return "\n\n".join(parts)


async def store_interaction(
    memory_client: MemoryClient,
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Store the interaction in memory (short-term + long-term extraction)."""
    # Store messages in short-term memory
    await memory_client.short_term.add_message(
        session_id=session_id,
        role="user",
        content=user_message,
        metadata={"user_id": user_id},
    )
    await memory_client.short_term.add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        metadata={"user_id": user_id},
    )

    # Extract and store user preferences from the conversation
    try:
        await memory_client.long_term.extract_and_store(
            text=f"User: {user_message}\nAssistant: {assistant_message}",
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        # extract_and_store may not exist in all versions; silently skip
        pass


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
        # Get memory context
        memory_context = await get_memory_context(
            memory_client, session_id, user_id, user_message
        )

        # Build the agent with MCP tools (session must stay alive during agent execution)
        mcp_client = MultiServerMCPClient(get_mcp_config())
        async with mcp_client.session("neo4j") as session:
            tools = await load_mcp_tools(session)

            llm = ChatOpenAI(model="gpt-4o", temperature=0)

            system_message = SYSTEM_PROMPT.format(memory_context=memory_context)
            agent = create_react_agent(llm, tools, prompt=system_message)

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_message)]}
            )

            # Extract the final AI response
            ai_messages = [
                m for m in result["messages"] if isinstance(m, AIMessage)
            ]
            response = ai_messages[-1].content if ai_messages else "No response."

        # Store the interaction in memory
        await store_interaction(
            memory_client, session_id, user_id, user_message, response
        )

        return response
    finally:
        await memory_client.close()


async def run_agent_stream(
    user_message: str,
    session_id: str,
    user_id: str = "default-user",
):
    """Stream the agent response token by token."""
    memory_settings = get_memory_settings()
    memory_client = MemoryClient(memory_settings)
    await memory_client.connect()

    try:
        memory_context = await get_memory_context(
            memory_client, session_id, user_id, user_message
        )

        mcp_client = MultiServerMCPClient(get_mcp_config())
        async with mcp_client.session("neo4j") as session:
            tools = await load_mcp_tools(session)
            llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

            system_message = SYSTEM_PROMPT.format(memory_context=memory_context)
            agent = create_react_agent(llm, tools, prompt=system_message)

            full_response = ""
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                version="v2",
            ):
                if (
                    event["event"] == "on_chat_model_stream"
                    and event["data"]["chunk"].content
                ):
                    token = event["data"]["chunk"].content
                    full_response += token
                    yield token

        # Store after streaming completes
        await store_interaction(
            memory_client, session_id, user_id, user_message, full_response
        )
    finally:
        await memory_client.close()
