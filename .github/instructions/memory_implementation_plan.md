# Implementation Plan: Conversational State Memory in Enterprise Banking Chatbot

## Objective
Implement secure, enterprise-grade conversational state memory for the banking chatbot using Python, FastAPI, LangGraph, and Google Gemini LLM, ensuring multi-turn context retention, session isolation, and strict security compliance.

---

## 1. Codebase Analysis
- [x] Inspect current FastAPI endpoints, LangGraph graph, state schema, nodes, prompts, and Gemini integration.
- [x] Identify how messages currently flow through the application.
- [x] Reuse existing architecture and dependencies wherever possible without introducing unnecessary packages.
- [x] Document proposed changes briefly before implementation.

## 2. Short-Term Conversational Memory
- [x] Utilize LangGraph state management to maintain conversation message history.
- [x] Implement appropriate message state and reducers (`MessagesState` and `add_messages`) compatible with the existing graph.
- [x] Preserve user messages and assistant responses across conversational turns.
- [x] Ensure Gemini receives relevant conversation history when generating responses.
- [x] Support follow-up questions, topic changes, clarifications, and pronoun references ("that transaction", "the previous account").
- [x] Implement message limits or token trimming without losing essential context.

Implementation note: The existing `ChatState` extends `MessagesState`. Before PostgreSQL checkpointing was added, `/chat` kept a bounded connection-local history. That history is now replaced by owner-validated durable thread checkpoints described in Steps 3 and 4 below.

## 3. Persistent State & Checkpointing Across API Requests
- [x] Configure LangGraph checkpointing mechanism with a durable backend (e.g., PostgreSQL).
- [x] Enforce unique, validated conversation/thread identifiers (`thread_id`) per session.
- [x] Pass the correct `thread_id` through the FastAPI request lifecycle.
- [x] Ensure subsequent requests for the same authorized conversation restore previous state accurately.
- [x] Guarantee user isolation so different users cannot access unauthorized threads.

## 4. Banking Security and Privacy Controls
- [x] Treat conversation history as sensitive banking data.
- [x] Enforce authentication and authorization before retrieving or updating state.
- [x] Validate ownership of each conversation; never blindly trust client-provided thread IDs.
- [x] Strictly prohibit storing sensitive secrets (passwords, PINs, OTPs, CVVs, tokens) in conversation memory.
- [x] Avoid logging sensitive message content.
- [ ] Enforce data minimization, access controls, and retention/deletion policies. (Minimization and access controls are implemented; retention/deletion is pending a product policy.)
- [x] Prevent conversational memory from authorizing transactions or bypassing core banking APIs.
- [x] Treat retrieved historical messages as untrusted context to defend against prompt injection.

Implementation note: PostgreSQL stores an owner-keyed registry and LangGraph checkpoints containing only Presidio-anonymized human/assistant text. Owner keys are HMAC fingerprints based on `JWT_SECRET`; tokens, graph account fields, tool calls, tool outputs, and message metadata are not checkpointed. Thread IDs are UUIDs, generated server-side when omitted and returned in completion events. Foreign or unknown IDs are rejected before checkpoint reads. The latest 20 messages are retained. The required retention/deletion period is not defined and has not been invented.

## 5. FastAPI Integration
- [ ] Integrate state memory into existing chat endpoints and LangGraph invocation logic.
- [ ] Maintain backward-compatibility for API contracts.
- [ ] Handle edge cases: new conversations, continuing sessions, missing identifiers, invalid sessions, and storage errors.
- [ ] Return standard assistant responses and conversation IDs using existing response schemas.

## 6. Testing & Verification
- [ ] Write/update tests verifying:
  - [ ] Multi-turn message recall within the same conversation.
  - [ ] Contextual follow-up question handling.
  - [ ] Separation of histories across different conversations and users.
  - [ ] State persistence across API requests and application restarts (durable storage).
  - [ ] Message trimming boundaries.
  - [ ] Non-persistence of sensitive data.
  - [ ] Continued functionality of existing banking integrations and guardrails.

## 7. Implementation Constraints
- [ ] Follow existing coding conventions, folder structure, and type hints.
- [ ] Avoid hardcoding secrets or database credentials.
- [ ] Verify functionality and run tests, reporting any test failures or gaps.
