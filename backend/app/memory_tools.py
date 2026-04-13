"""LangChain tools for interacting with the neo4j-agent-memory context graph."""

import json
from typing import Optional

from langchain.tools import BaseTool
from neo4j_agent_memory import MemoryClient
from pydantic import BaseModel, Field


class SearchEntitiesInput(BaseModel):
    query: str = Field(description="Search query for entities")
    entity_types: Optional[list[str]] = Field(
        default=None,
        description="Entity type filters (e.g., ['PRODUCT', 'PERSON', 'ORGANIZATION'])",
    )
    limit: int = Field(default=10, description="Maximum results to return")


class SearchEntitesTool(BaseTool):
    name: str = "search_memory_entities"
    description: str = (
        "Search the knowledge graph for entities like products, people, or organizations. "
        "Use this to find information about things mentioned in past conversations."
    )
    args_schema: type[BaseModel] = SearchEntitiesInput
    memory_client: MemoryClient

    async def _arun(
        self,
        query: str,
        entity_types: Optional[list[str]] = None,
        limit: int = 10,
    ) -> str:
        entities = await self.memory_client.long_term.search_entities(
            query=query,
            entity_types=entity_types,
            limit=limit,
        )
        if not entities:
            return "No entities found."
        return json.dumps(
            [
                {
                    "name": e.name,
                    "type": str(e.type),
                    "description": e.description,
                    "attributes": e.attributes,
                }
                for e in entities
            ],
            indent=2,
        )

    def _run(self, **kwargs) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))


class GetPreferencesInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Category filter (e.g., 'brand', 'style', 'topic'). If not provided, searches broadly.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query to find specific preferences.",
    )


class GetPreferencesTool(BaseTool):
    name: str = "get_user_preferences"
    description: str = (
        "Get stored preferences for the current user. Use this to personalize "
        "recommendations based on what the user has previously stated they like or dislike."
    )
    args_schema: type[BaseModel] = GetPreferencesInput
    memory_client: MemoryClient
    user_id: str = "default-user"

    async def _arun(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:
        if query:
            preferences = await self.memory_client.long_term.search_preferences(
                query=query,
                category=category,
            )
        elif category:
            preferences = await self.memory_client.long_term.get_preferences_by_category(
                category=category,
            )
        else:
            # No filter — get all preferences via a broad low-threshold search
            preferences = await self.memory_client.long_term.search_preferences(
                query="preferences likes dislikes",
                limit=50,
                threshold=0.0,
            )
        # Filter to current user
        filtered = [
            p for p in preferences
            if p.metadata and p.metadata.get("user_id") == self.user_id
        ]
        if not filtered:
            return "No preferences found."
        return json.dumps(
            [{"category": p.category, "preference": p.preference} for p in filtered],
            indent=2,
        )

    def _run(self, **kwargs) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))


class SavePreferenceInput(BaseModel):
    preference: str = Field(description="The preference to save")
    category: str = Field(
        description="Category: brand, style, size, topic, etc."
    )


class SavePreferenceTool(BaseTool):
    name: str = "save_user_preference"
    description: str = (
        "Save a new user preference learned during the conversation. "
        "Use when the user expresses a like, dislike, or preference."
    )
    args_schema: type[BaseModel] = SavePreferenceInput
    memory_client: MemoryClient
    user_id: str = "default-user"

    async def _arun(self, preference: str, category: str) -> str:
        await self.memory_client.long_term.add_preference(
            preference=preference,
            category=category,
            confidence=0.85,
            metadata={"user_id": self.user_id},
        )
        return f"Saved preference: [{category}] {preference}"

    def _run(self, **kwargs) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))


class SearchConversationHistoryInput(BaseModel):
    query: str = Field(description="Search query for past messages")
    limit: int = Field(default=5, description="Maximum results to return")


class SearchConversationHistoryTool(BaseTool):
    name: str = "search_conversation_history"
    description: str = (
        "Search past conversation messages for relevant context. "
        "Use this to recall what was discussed in earlier conversations."
    )
    args_schema: type[BaseModel] = SearchConversationHistoryInput
    memory_client: MemoryClient
    user_id: str = "default-user"

    async def _arun(self, query: str, limit: int = 5) -> str:
        messages = await self.memory_client.short_term.search_messages(
            query=query,
            limit=limit,
            metadata_filters={"user_id": self.user_id},
        )
        if not messages:
            return "No relevant messages found."
        return json.dumps(
            [
                {"role": str(m.role), "content": m.content[:300]}
                for m in messages
            ],
            indent=2,
        )

    def _run(self, **kwargs) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))


def create_memory_tools(memory_client: MemoryClient, user_id: str = "default-user") -> list[BaseTool]:
    """Create all memory tools for the agent."""
    return [
        SearchEntitesTool(memory_client=memory_client),
        GetPreferencesTool(memory_client=memory_client, user_id=user_id),
        SavePreferenceTool(memory_client=memory_client, user_id=user_id),
        SearchConversationHistoryTool(memory_client=memory_client, user_id=user_id),
    ]
