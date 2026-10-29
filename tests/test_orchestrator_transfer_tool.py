import asyncio
import json

import httpx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from services.orchestrator import agent


def test_gemini_receives_prior_conversation_messages(monkeypatch) -> None:
    captured_messages = []

    class FakeModel:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            captured_messages.extend(messages)
            yield AIMessageChunk(content="The earlier topic was rent.")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda **kwargs: FakeModel(),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    account_id = "sandbox-checking-001"
    conversation: agent.ChatState = {
        "messages": [],
        "account_id": account_id,
        "tool_states": {},
    }
    agent.append_conversation_message(
        conversation, HumanMessage(content="I paid rent yesterday.")
    )
    agent.append_conversation_message(
        conversation, AIMessage(content="Understood.")
    )
    state = agent.initial_chat_state(
        "What was the payment for?", account_id, conversation["messages"]
    )
    graph = agent.build_agent_graph("validated-token", account_id)

    asyncio.run(graph.ainvoke(state))

    assert isinstance(captured_messages[0], SystemMessage)
    assert [message.content for message in captured_messages[1:]] == [
        "I paid rent yesterday.",
        "Understood.",
        "What was the payment for?",
    ]


def test_old_transfer_request_does_not_select_transfer_tool(monkeypatch) -> None:
    captured_tools = []

    class FakeModel:
        def bind_tools(self, tools):
            captured_tools.extend(registered_tool.name for registered_tool in tools)
            return self

        async def astream(self, messages):
            yield AIMessageChunk(content="A general follow-up.")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda **kwargs: FakeModel(),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    account_id = "sandbox-checking-001"
    state = agent.initial_chat_state(
        "What did we discuss?",
        account_id,
        [HumanMessage(content="I want to transfer money.")],
    )

    asyncio.run(agent.build_agent_graph("validated-token", account_id).ainvoke(state))

    assert "prepare_transfer" not in captured_tools


def test_past_transfer_statement_does_not_enable_transfer_tool(monkeypatch) -> None:
    captured_tools = []

    class FakeModel:
        def bind_tools(self, tools):
            captured_tools.extend(registered_tool.name for registered_tool in tools)
            return self

        async def astream(self, messages):
            yield AIMessageChunk(content="I can use that as conversation context.")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda **kwargs: FakeModel(),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    account_id = "sandbox-checking-001"
    state = agent.initial_chat_state(
        "I transferred $1,600 from Savings to Checking.", account_id
    )

    asyncio.run(agent.build_agent_graph("validated-token", account_id).ainvoke(state))

    assert captured_tools == []


def test_balance_tool_is_only_enabled_for_balance_questions(monkeypatch) -> None:
    captured_tools = []

    class FakeModel:
        def bind_tools(self, tools):
            captured_tools.extend(registered_tool.name for registered_tool in tools)
            return self

        async def astream(self, messages):
            yield AIMessageChunk(content="A general financial answer.")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda **kwargs: FakeModel(),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    account_id = "sandbox-checking-001"
    graph = agent.build_agent_graph("validated-token", account_id)

    for content, expected_tools in [
        (
            "I'm asking about my savings account. I plan to deposit $200 each "
            "month. What does an automatic deposit do?",
            [],
        ),
        ("What is my current balance?", ["check_balance"]),
    ]:
        captured_tools.clear()
        state = agent.initial_chat_state(content, account_id)
        asyncio.run(graph.ainvoke(state))
        assert captured_tools == expected_tools


def test_conversation_history_is_bounded() -> None:
    conversation: agent.ChatState = {
        "messages": [],
        "account_id": "sandbox-checking-001",
        "tool_states": {},
    }

    for index in range(agent.MAX_CONVERSATION_MESSAGES + 2):
        agent.append_conversation_message(
            conversation, HumanMessage(content=f"message {index}")
        )

    assert len(conversation["messages"]) == agent.MAX_CONVERSATION_MESSAGES
    assert conversation["messages"][0].content == "message 2"


def test_prepare_transfer_is_registered_and_streamed_as_confirmation(monkeypatch) -> None:
    token = "validated-test-token"
    account_id = "sandbox-checking-001"
    monkeypatch.setenv("BALANCE_SERVICE_URL", "http://balance.test")
    accounts = [
        {"account_id": "sandbox-checking-001", "label": "Checking account"},
        {"account_id": "sandbox-savings-001", "label": "Savings account"},
    ]

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            assert headers == {"Authorization": f"Bearer {token}"}
            request = httpx.Request("GET", url)
            if url == "http://balance.test/sandbox/accounts":
                return httpx.Response(200, request=request, json=accounts)
            assert url == "http://balance.test/sandbox/balances/sandbox-savings-001"
            return httpx.Response(
                200,
                request=request,
                json={
                    "account_id": "sandbox-savings-001",
                    "balance": "3200.00",
                    "currency": "USD",
                },
            )

    class FakeModel:
        def bind_tools(self, tools):
            assert [registered_tool.name for registered_tool in tools] == [
                "prepare_transfer",
            ]
            return self

        async def astream(self, messages):
            if any(isinstance(message, ToolMessage) for message in messages):
                yield AIMessageChunk(content="Review this transfer.")
            else:
                yield AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "prepare_transfer",
                            "args": {
                                "source_account": "savings",
                                "recipient": "checking",
                                "amount": "1000.00",
                            },
                            "id": "transfer-call-1",
                            "type": "tool_call",
                        }
                    ],
                )

    monkeypatch.setattr(agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda **kwargs: FakeModel(),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    graph = agent.build_agent_graph(token, account_id)
    state = agent.initial_chat_state(
        "Transfer $1,000 from savings to checking", account_id
    )

    async def collect_events():
        return [event async for event in agent.stream_agent_events(graph, state)]

    events = asyncio.run(collect_events())
    event = events[0]
    assert event["type"] == "transfer"
    assert event["transaction_id"]
    assert event["status"] == "PENDING_CONFIRMATION"
    assert event["mfa_required"] is True
    assert event["transfer"]["recipient"] == "Checking account"
    assert events[-1] == {"type": "completion"}


def test_prepare_transfer_refuses_frozen_source_before_preview(monkeypatch) -> None:
    accounts = [
        {"account_id": "sandbox-checking-001", "label": "Checking account", "status": "ACTIVE"},
        {"account_id": "sandbox-savings-001", "label": "Savings account", "status": "FROZEN"},
    ]

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request, json=accounts)

    monkeypatch.setattr(agent.httpx, "AsyncClient", FakeAsyncClient)
    prepare_transfer = agent.build_agent_tools("validated-token", "sandbox-checking-001")[1]
    result = asyncio.run(
        prepare_transfer.ainvoke(
            {
                "source_account": "savings",
                "recipient": "checking",
                "amount": "25.00",
            }
        )
    )

    assert result == "This source account is frozen and cannot send transfers."


def test_prepare_transfer_reports_insufficient_balance(monkeypatch) -> None:
    token = "validated-test-token"
    monkeypatch.setenv("BALANCE_SERVICE_URL", "http://balance.test")
    accounts = [
        {"account_id": "sandbox-checking-001", "label": "Checking account"},
        {"account_id": "sandbox-savings-001", "label": "Savings account"},
    ]

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            assert headers == {"Authorization": f"Bearer {token}"}
            request = httpx.Request("GET", url)
            if url == "http://balance.test/sandbox/accounts":
                return httpx.Response(200, request=request, json=accounts)
            assert url == "http://balance.test/sandbox/balances/sandbox-checking-001"
            return httpx.Response(
                200,
                request=request,
                json={
                    "account_id": "sandbox-checking-001",
                    "balance": "1500.00",
                    "currency": "USD",
                },
            )

    monkeypatch.setattr(agent.httpx, "AsyncClient", FakeAsyncClient)
    prepare_transfer = agent.build_agent_tools(token, "sandbox-checking-001")[1]
    result = asyncio.run(
        prepare_transfer.ainvoke(
            {
                "source_account": "checking",
                "recipient": "saving",
                "amount": "1600.00",
            }
        )
    )
    payload = json.loads(result)
    assert payload == {
        "kind": "transfer_error",
        "account_id": "sandbox-checking-001",
        "message": "Insufficient balance. Available balance is USD 1500.00.",
    }

    class FakeGraph:
        async def astream(self, state, stream_mode):
            yield (
                ToolMessage(
                    content=result,
                    name="prepare_transfer",
                    tool_call_id="transfer-call-1",
                ),
                {},
            )

    async def collect_events():
        state = agent.initial_chat_state(
            "Transfer $1600 from checking to saving", "sandbox-checking-001"
        )
        return [event async for event in agent.stream_agent_events(FakeGraph(), state)]

    events = asyncio.run(collect_events())
    assert events == [
        {"type": "content", "content": payload["message"]},
        {"type": "completion"},
    ]


def test_prepare_transfer_asks_for_missing_entities() -> None:
    prepare_transfer = agent.build_agent_tools(
        "validated-token", "sandbox-checking-001"
    )[1]

    source_question = asyncio.run(prepare_transfer.ainvoke({}))
    recipient_question = asyncio.run(
        prepare_transfer.ainvoke({"source_account": "Savings"})
    )

    assert source_question == "Which linked account would you like to transfer from?"
    assert recipient_question == "Which linked account should receive the transfer?"
