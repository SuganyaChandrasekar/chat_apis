import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from services.orchestrator import memory


def test_safe_messages_redacts_secrets_and_drops_message_metadata(monkeypatch):
    monkeypatch.setattr(memory, "anonymize_message", lambda content: content)
    raw_token = "sk-abcdefghijklmnopqrstuvwxyz123456"
    message = HumanMessage(
        content=f"My PIN is 1234 and this is my key {raw_token}",
        additional_kwargs={"access_token": "never-store-this"},
    )

    [safe_message] = memory.safe_messages([message])

    assert isinstance(safe_message, HumanMessage)
    assert "1234" not in safe_message.content
    assert raw_token not in safe_message.content
    assert safe_message.additional_kwargs == {}


def test_owner_key_is_keyed_and_does_not_store_subject(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")

    owner_key = memory._owner_key("customer-account-123")

    assert owner_key != "customer-account-123"
    assert len(owner_key) == 64


def test_input_sanitizer_redacts_credentials_before_model_use(monkeypatch):
    monkeypatch.setattr(memory, "anonymize_message", lambda content: content)

    sanitized = memory.sanitize_content(
        "My OTP is 123456 and my bearer token is abc.def.ghi "
        "and my API key is sk-abcdefghijklmnopqrstuvwxyz123456 "
        "and access_token=secret-token-value "
        "and this session token is "
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature12345678"
    )

    assert "123456" not in sanitized
    assert "abc.def.ghi" not in sanitized
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in sanitized
    assert "secret-token-value" not in sanitized
    assert "eyJhbGciOiJIUzI1NiJ9" not in sanitized
    structured = memory.sanitize_content(
        '{"access_token":"sensitive-token-value"}'
    )
    assert "sensitive-token-value" not in structured


def test_expiration_wipes_financial_values_but_preserves_turns_and_entities(
    monkeypatch,
):
    monkeypatch.setattr(memory, "anonymize_message", lambda content: content)
    messages = [
        HumanMessage(
            content=(
                "Transfer $125.00 from Checking account to Savings account; "
                "another transfer $20 from account 12345678 to 87654321; "
                "account_id: checking-123"
            )
        ),
        AIMessage(content="Your available balance is USD 500.00."),
        AIMessage(
            content="Earlier balance check for Checking account: USD 1,500.00."
        ),
    ]

    expired = memory._expired_messages(messages)

    assert len(expired) == 3
    assert "Checking account" in expired[0].content
    assert "Savings account" in expired[0].content
    assert "125.00" not in expired[0].content
    assert "12345678" not in expired[0].content
    assert "87654321" not in expired[0].content
    assert "checking-123" not in expired[0].content
    assert "500.00" not in expired[1].content
    assert "1,500.00" not in expired[2].content
    assert "Checking account" in expired[2].content
    structured = memory._expired_content(
        '{"balance":"500.00","amount":"125.00",'
        '"source_account_id":"checking-123"}'
    )
    assert "500.00" not in structured
    assert "125.00" not in structured
    assert "checking-123" not in structured


def test_expiration_rebuilds_checkpoint_with_scrubbed_history(monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(memory, "anonymize_message", lambda content: content)
        checkpointer = InMemorySaver()
        graph = memory._build_history_graph(checkpointer)
        thread_id = uuid4()
        config = {"configurable": {"thread_id": str(thread_id)}}
        await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="Transfer $125 from Checking to Savings."
                    ),
                    AIMessage(content="Transfer prepared for your review."),
                    HumanMessage(content="Can I use Checking again?"),
                    AIMessage(content="Yes, the prior conversation mentioned Checking."),
                ]
            },
            config,
        )
        store = memory.ConversationStore.__new__(memory.ConversationStore)
        store._checkpointer = checkpointer
        store._history_graph = graph

        await store._expire_sensitive_context(thread_id)
        restored = await graph.aget_state(config)
        restored_messages = restored.values["messages"]

        assert len(restored_messages) == 4
        assert "125" not in restored_messages[0].content
        assert "Checking" in restored_messages[0].content
        assert "Can I use Checking again?" == restored_messages[2].content

    asyncio.run(run())


def test_conversation_store_expires_inactive_rows_and_scrubs_checkpoints(
    monkeypatch,
):
    class Cursor:
        async def fetchall(self):
            return [(uuid4(),), (uuid4(),)]

    class Connection:
        def __init__(self):
            self.queries = []

        async def execute(self, query, params=()):
            self.queries.append((query, params))
            return Cursor()

        @asynccontextmanager
        async def transaction(self):
            yield

    class Pool:
        def __init__(self):
            self.connection_instance = Connection()

        @asynccontextmanager
        async def connection(self):
            yield self.connection_instance

    async def run() -> None:
        pool = Pool()
        store = memory.ConversationStore.__new__(memory.ConversationStore)
        store._pool = pool
        expired_threads = []

        async def expire_context(thread_id):
            expired_threads.append(thread_id)

        store._expire_sensitive_context = expire_context

        count = await store.expire_inactive_sessions()

        assert count == 2
        assert len(expired_threads) == 2
        select_query, select_params = pool.connection_instance.queries[0]
        assert "last_activity_at <= now()" in select_query
        assert "FOR UPDATE SKIP LOCKED" in select_query
        assert select_params == (
            memory.SESSION_INACTIVITY_MINUTES,
            memory.EXPIRATION_BATCH_SIZE,
        )
        assert len(pool.connection_instance.queries) == 3

    asyncio.run(run())


def test_resolving_expired_thread_scrubs_checkpoint_before_reactivating(
    monkeypatch,
):
    class Cursor:
        async def fetchone(self):
            return ("owner-key", object(), False)

    class Connection:
        def __init__(self):
            self.queries = []

        async def execute(self, query, params=()):
            self.queries.append((query, params))
            return Cursor()

        @asynccontextmanager
        async def transaction(self):
            yield

    class Pool:
        def __init__(self):
            self.connection_instance = Connection()

        @asynccontextmanager
        async def connection(self):
            yield self.connection_instance

    async def run() -> None:
        thread_id = uuid4()
        pool = Pool()
        store = memory.ConversationStore.__new__(memory.ConversationStore)
        store._pool = pool
        expired_threads = []

        async def expire_context(expired_thread_id):
            expired_threads.append(expired_thread_id)

        store._expire_sensitive_context = expire_context
        monkeypatch.setattr(memory, "_owner_key", lambda owner_id: "owner-key")

        resolved_thread_id = await store.resolve_thread("customer-123", thread_id)

        assert resolved_thread_id == thread_id
        assert expired_threads == [thread_id]
        assert len(pool.connection_instance.queries) == 2
        update_query, update_params = pool.connection_instance.queries[1]
        assert "last_activity_at = now()" in update_query
        assert "expired_at = NULL" in update_query
        assert update_params == (thread_id,)

    asyncio.run(run())


def test_message_checkpoint_restores_only_bounded_chat_history():
    async def run() -> None:
        checkpointer = InMemorySaver()
        graph = memory._build_history_graph(checkpointer)
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        messages = [
            HumanMessage(content=f"question {index}")
            if index % 2 == 0
            else AIMessage(content=f"answer {index}")
            for index in range(memory.MAX_PERSISTED_MESSAGES + 2)
        ]

        await graph.ainvoke({"messages": messages}, config)
        restored = await graph.aget_state(config)

        assert len(restored.values["messages"]) == memory.MAX_PERSISTED_MESSAGES
        assert restored.values["messages"][0].content == "question 2"

    asyncio.run(run())