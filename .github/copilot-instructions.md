# Banking Chatbot Backend Instructions

## Service architecture
- Keep the chat orchestrator, balance service, and transfer service independently deployable. Do not put account or transfer business logic in the orchestrator.
- Use Python and FastAPI for service APIs, LangGraph for chat/workflow orchestration, and Microsoft Presidio for PII detection and anonymization.
- Keep routes, schemas, orchestration, PII handling, persistence, and configuration in focused modules.
- Read configuration from environment variables. Never commit secrets or production credentials.
- Preserve the frontend chat contract unless both sides are deliberately updated. Current message shape: `{ "type": "message", "content": "..." }`; responses stream token/content payloads followed by completion.

## Security and banking behavior
- Treat every request as untrusted. Authenticate users at the public API boundary and authorize balance access using verified server-side identity. Never trust client-supplied user or account identifiers.
- Keep internal services private and authenticate service-to-service calls. Do not return mock or fabricated balances outside tests; connect only to a configured, trusted account data source.
- Mask sensitive user content with Presidio before sending it to an LLM or writing it to logs. Fail closed if masking is unavailable.
- Never log prompts containing PII, credentials, OTPs, or access tokens.
- Keep transfer execution in deterministic backend services. Require server-side validation, authorization, explicit confirmation, applicable MFA, and idempotency before execution.
- Record each financial transaction as an immutable audit event with the required state `PENDING_VALIDATION`, `SUCCESS`, or `FAILED`; store only necessary non-sensitive metadata.
- Do not expose stack traces, prompts, credentials, or internal account data in API errors.

## API and validation
- Use typed Pydantic request and response models, explicit timeouts, bounded input, and consistent errors.
- Configure CORS from explicit environment-based origins. Do not allow every origin in production.
- Authenticate WebSocket connections, validate messages, handle disconnects, and avoid unbounded conversation history.
- Add focused tests for routes, validation, authorization, orchestration, PII handling, and downstream failures. Mock external systems in unit tests.
- Do not invent banking schemas, data sources, or authorization rules when requirements are missing; surface the missing decision before implementation.