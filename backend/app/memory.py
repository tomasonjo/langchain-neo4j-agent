"""Neo4j Agent Memory integration for LangChain."""

from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from neo4j_agent_memory import MemoryClient


class Neo4jChatMessageHistory(BaseChatMessageHistory):
    """LangChain message history backed by neo4j-agent-memory."""

    def __init__(
        self,
        memory_client: MemoryClient,
        session_id: str,
        user_id: str | None = None,
    ):
        self.memory_client = memory_client
        self.session_id = session_id
        self.user_id = user_id

    async def aget_messages(self) -> List[BaseMessage]:
        conversation = await self.memory_client.short_term.get_conversation(
            session_id=self.session_id,
        )
        result = []
        for msg in conversation.messages:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))
        return result

    @property
    def messages(self) -> List[BaseMessage]:
        import asyncio

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aget_messages())

    async def aadd_message(self, message: BaseMessage) -> None:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        await self.memory_client.short_term.add_message(
            session_id=self.session_id,
            role=role,
            content=message.content,
            metadata={"user_id": self.user_id} if self.user_id else None,
        )

    def add_message(self, message: BaseMessage) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.aadd_message(message))

    async def aclear(self) -> None:
        await self.memory_client.short_term.delete_session(
            session_id=self.session_id,
        )

    def clear(self) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.aclear())
