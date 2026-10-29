import os
import re
import hmac
import hashlib
from collections.abc import Sequence
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, MessagesState, StateGraph
from psycopg_pool import AsyncConnectionPool

from services.orchestrator.pii import anonymize_message

MAX_PERSISTED_MESSAGES = 20
SESSION_INACTIVITY_MINUTES = 15
EXPIRATION_BATCH_SIZE = 50
_SECRET_VALUE_PATTERN = re.compile(
    r"\b(password|passcode|pin|otp|one[-_ ]time code|cvv|cvc|security[_ ]code|"
    r"access[_ ]token|refresh[_ ]token|bearer[_ ]token)\b\s*(?:is|:|=)?\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_STRUCTURED_SECRET_VALUE_PATTERN = re.compile(
    r'("(?:password|passcode|pin|otp|access_token|refresh_token|bearer_token|'
    r'authorization)"\s*:\s*")[^"]*(")',
    re.IGNORECASE,
)
_BEARER_VALUE_PATTERN = re.compile(
    r"\bBearer\s+(?!token\b)[A-Za-z0-9._~+/-]+=*", re.IGNORECASE
)
_TOKEN_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{30,})\b"
)
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)
_CURRENCY_AMOUNT_PATTERN = re.compile(
    r"(?:\bUSD\s*|\$\s*)\d[\d,]*(?:\.\d{1,2})?", re.IGNORECASE
)
_FINANCIAL_FIELD_AMOUNT_PATTERN = re.compile(
    r"(\b(?:balance|amount)\b(?:\s+(?:is|of|was|currently))?\s*[:=]?\s*)"
    r"\d[\d,]*(?:\.\d{1,2})?",
    re.IGNORECASE,
)
_TRANSFER_AMOUNT_PATTERN = re.compile(
    r"(\b(?:transfer(?:red)?|send|move)\b[^\d]{0,30})"
    r"\d[\d,]*(?:\.\d{1,2})?",
    re.IGNORECASE,
)
_ACCOUNT_IDENTIFIER_PATTERN = re.compile(
    r"(\b(?:source_|destination_)?account[_ ](?:id|number)\b"
    r'"?\s*[:=]?\s*"?)([^,\s;}"\']+)(["\']?)',
    re.IGNORECASE,
)
_TRANSFER_ACCOUNT_ID_PATTERN = re.compile(
    r"(\b(?:from|to)\s+(?:the\s+)?(?:account\s+)?)#?\d[\d-]{3,}\b",
    re.IGNORECASE,
)
_STRUCTURED_FINANCIAL_VALUE_PATTERN = re.compile(
    r'("(?:balance|amount)"\s*:\s*")[^"]*(")', re.IGNORECASE
)


class ConversationNotFound(Exception):
    pass


def _owner_key(owner_id: str) -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret or len(secret.encode("utf-8")) < 32:
        raise RuntimeError("Conversation ownership is not configured")
    return hmac.new(
        secret.encode("utf-8"), owner_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Conversation storage is not configured")
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def sanitize_content(content: str) -> str:
    content = anonymize_message(content)
    content = _BEARER_VALUE_PATTERN.sub("Bearer [REDACTED]", content)
    content = _STRUCTURED_SECRET_VALUE_PATTERN.sub(
        r"\1[REDACTED]\2", content
    )
    content = _SECRET_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)} [REDACTED]", content
    )
    content = _TOKEN_PATTERN.sub("[REDACTED_TOKEN]", content)
    return _JWT_PATTERN.sub("[REDACTED_TOKEN]", content)


def safe_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    safe = []
    for message in messages:
        if not isinstance(message.content, str):
            continue
        content = sanitize_content(message.content)
        if isinstance(message, HumanMessage):
            safe.append(HumanMessage(content=content, id=message.id))
        elif isinstance(message, AIMessage):
            safe.append(AIMessage(content=content, id=message.id))
    return safe


def _expired_content(content: str) -> str:
    content = sanitize_content(content)
    content = _CURRENCY_AMOUNT_PATTERN.sub("[EXPIRED]", content)
    content = _FINANCIAL_FIELD_AMOUNT_PATTERN.sub(r"\1[EXPIRED]", content)
    content = _TRANSFER_AMOUNT_PATTERN.sub(r"\1[EXPIRED]", content)
    content = _TRANSFER_ACCOUNT_ID_PATTERN.sub(r"\1[EXPIRED]", content)
    content = _ACCOUNT_IDENTIFIER_PATTERN.sub(r"\1[EXPIRED]\3", content)
    content = _STRUCTURED_FINANCIAL_VALUE_PATTERN.sub(
        r"\1[EXPIRED]\2", content
    )
    return content


def _expired_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    expired = []
    for message in safe_messages(messages):
        content = _expired_content(message.content)
        if isinstance(message, HumanMessage):
            expired.append(HumanMessage(content=content, id=message.id))
        elif isinstance(message, AIMessage):
            expired.append(AIMessage(content=content, id=message.id))
    return expired


def _build_history_graph(checkpointer: AsyncPostgresSaver):
    def trim_history(state: MessagesState) -> dict:
        expired = state["messages"][:-MAX_PERSISTED_MESSAGES]
        if not expired:
            return {}
        return {
            "messages": [RemoveMessage(id=message.id) for message in expired]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("trim_history", trim_history)
    graph.set_entry_point("trim_history")
    graph.add_edge("trim_history", END)
    return graph.compile(checkpointer=checkpointer)


class ConversationStore:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        checkpointer: AsyncPostgresSaver,
    ) -> None:
        self._pool = pool
        self._checkpointer = checkpointer
        self._history_graph = _build_history_graph(checkpointer)

    @classmethod
    async def open(cls) -> "ConversationStore":
        _owner_key("configuration-check")
        pool = AsyncConnectionPool(
            conninfo=_database_url(),
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open()
        try:
            await pool.wait()
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            return cls(pool, checkpointer)
        except Exception:
            await pool.close()
            raise

    async def close(self) -> None:
        await self._pool.close()

    async def resolve_thread(
        self, owner_id: str, requested_thread_id: UUID | None
    ) -> UUID:
        owner_key = _owner_key(owner_id)
        if requested_thread_id is None:
            thread_id = uuid4()
            async with self._pool.connection() as connection:
                await connection.execute(
                    "INSERT INTO chat_conversations (thread_id, owner_key) "
                    "VALUES (%s, %s)",
                    (thread_id, owner_key),
                )
            return thread_id

        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT owner_key, expired_at, "
                    "last_activity_at <= now() - "
                    "(%s * INTERVAL '1 minute') AS inactive "
                    "FROM chat_conversations WHERE thread_id = %s FOR UPDATE",
                    (SESSION_INACTIVITY_MINUTES, requested_thread_id),
                )
                row = await cursor.fetchone()
                if row is None or not hmac.compare_digest(row[0], owner_key):
                    raise ConversationNotFound
                if row[1] is not None or row[2]:
                    await self._expire_sensitive_context(requested_thread_id)
                await connection.execute(
                    "UPDATE chat_conversations SET last_activity_at = now(), "
                    "expired_at = NULL WHERE thread_id = %s",
                    (requested_thread_id,),
                )
        return requested_thread_id

    async def load(self, thread_id: UUID) -> list[BaseMessage]:
        snapshot = await self._history_graph.aget_state(
            {"configurable": {"thread_id": str(thread_id)}}
        )
        return safe_messages(snapshot.values.get("messages", []))

    async def append(
        self, thread_id: UUID, messages: Sequence[BaseMessage]
    ) -> None:
        safe = safe_messages(messages)
        if not safe:
            return
        await self._history_graph.ainvoke(
            {"messages": safe},
            {"configurable": {"thread_id": str(thread_id)}},
        )
        async with self._pool.connection() as connection:
            await connection.execute(
                "UPDATE chat_conversations SET last_activity_at = now(), "
                "expired_at = NULL "
                "WHERE thread_id = %s",
                (thread_id,),
            )

    async def expire_inactive_sessions(self) -> int:
        expired_count = 0
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT thread_id FROM chat_conversations "
                    "WHERE expired_at IS NULL AND last_activity_at <= "
                    "now() - (%s * INTERVAL '1 minute') "
                    "ORDER BY last_activity_at LIMIT %s FOR UPDATE SKIP LOCKED",
                    (SESSION_INACTIVITY_MINUTES, EXPIRATION_BATCH_SIZE),
                )
                rows = await cursor.fetchall()
                for (thread_id,) in rows:
                    await self._expire_sensitive_context(thread_id)
                    await connection.execute(
                        "UPDATE chat_conversations SET expired_at = now() "
                        "WHERE thread_id = %s",
                        (thread_id,),
                    )
                    expired_count += 1
        return expired_count

    async def _expire_sensitive_context(self, thread_id: UUID) -> None:
        config = {"configurable": {"thread_id": str(thread_id)}}
        snapshot = await self._history_graph.aget_state(config)
        messages = _expired_messages(snapshot.values.get("messages", []))
        await self._checkpointer.adelete_thread(str(thread_id))
        if messages:
            await self._history_graph.ainvoke({"messages": messages}, config)