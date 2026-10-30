# Banking Chatbot Services

This repository contains three independently deployable FastAPI services:

- `orchestrator`: public chat/API entry point, intended for LangGraph routing.
- `balance`: private balance-query service.
- `transfer`: private transfer workflow service.

## Run locally

Requires Python 3.11 or newer. Create an isolated environment and install the development dependencies:

```powershell
cd C:\chatbot\chat_apis
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[orchestrator,dev]"
python -m spacy download en_core_web_lg
```

Run each service in a separate terminal after activating the virtual environment:

```powershell
python -m uvicorn services.orchestrator.main:app --reload --port 8000
python -m uvicorn services.balance.main:app --reload --port 8001
python -m uvicorn services.transfer.main:app --reload --port 8002
python -m uvicorn services.mfa.main:app --reload --port 8003 --env-file .env
```

Each service exposes a health endpoint at its local port, for example `http://localhost:8001/health` for the balance service.

## JWT authentication

The backend owns JWT creation and verification. Production authentication should issue a token through a trusted identity provider or backend login endpoint; the chatbot services should only validate the token and authorize access using its verified `sub` claim.

Access tokens should use the following contract:

- `sub`: verified customer identifier
- `iat`: token issue time
- `exp`: expiration time
- `iss`: `banking-auth`
- `aud`: `banking-api`
- `typ`: `access`
- HS256 signature

The backend must keep `JWT_SECRET` private and use at least 32 UTF-8 bytes of randomly generated entropy. The current minimum is intentionally conservative; production deployments should use a managed secret store and rotate keys. Do not put the secret or private key in the frontend.

## Test chat locally

The React UI uses `.env.local` for its local endpoints. Start the balance service and orchestrator in separate PowerShell terminals. Use the same locally generated `JWT_SECRET` (at least 32 bytes) in both terminals:

```powershell
$env:SANDBOX_MODE = "true"
$env:JWT_SECRET = "<same local secret in both terminals>"
python -m uvicorn services.balance.main:app --reload --port 8001
```

In the orchestrator terminal, also enable the loopback-only test login and allow the Vite origin. Set `GEMINI_API_KEY` in this terminal:

```powershell
$env:SANDBOX_MODE = "true"
$env:LOCAL_TEST_AUTH_ENABLED = "true"
$env:CORS_ALLOWED_ORIGINS = "http://localhost:5173"
$env:JWT_SECRET = "<same local secret as the balance terminal>"
$env:GEMINI_API_KEY = "<your Gemini API key>"
python -m uvicorn services.orchestrator.main:app --reload --port 8000
```

Then run the UI from `C:\chatbot\bank-chatbot_ui` with `npm run dev` and open `http://localhost:5173`. Select **Sandbox login**, then ask “What is my balance?”. The test login issues a 30-minute HttpOnly cookie for the fixed `sandbox-checking-001` fixture account. It is available only when both sandbox and local test auth are explicitly enabled, and only to loopback clients and configured browser origins. Do not enable local test auth outside local sandbox development.

## Current scope and required integrations

The orchestrator chat API uses HS256 JWTs signed with `JWT_SECRET` (minimum 32 bytes), with issuer `banking-auth`, audience `banking-api`, and the authorized account ID in `sub`. Configure `GEMINI_API_KEY` and optionally `GEMINI_MODEL` (defaults to `gemini-3.5-flash-lite`) to enable chat. NeMo Guardrails runs Presidio input/output masking and a history-aware LLM self-check for retail banking scope, credential safety, and prompt injection before user content is passed to LangGraph. The orchestrator also uses a structured LLM classification of each masked message and recent turns to route balance checks, transfer questions, account-number requests, and transfer previews and extract relevant entities; classified account names are matched only against accounts returned for the authenticated customer. These checks fail closed if NeMo, Presidio, a model check, or audit persistence is unavailable. Guardrail decisions are written to `security_audit_events` without storing prompt or response text. Configure `DATABASE_URL` and run `alembic upgrade head` before enabling chat; startup initializes the LangGraph PostgreSQL checkpoint tables. `POST /chat/stream` streams SSE; `/chat` accepts authenticated WebSocket messages and emits content and completion events.

Chat messages may include an optional UUID `thread_id`. If omitted, the orchestrator creates an owner-bound conversation and returns its ID in the completion event. Send that ID on later HTTP requests or after reconnecting a WebSocket to restore that conversation. The orchestrator verifies ownership against the authenticated JWT subject before reading history. Checkpoints retain at most 20 anonymized human/assistant text messages; raw tool results, credentials, and message metadata are excluded. A recent displayed balance is retained as a text summary so the chatbot can answer follow-up questions during the active session. After 15 minutes without activity, a background sweep expires the session and removes stored currency amounts and explicit account identifiers from its checkpoint history while retaining prior turns and non-sensitive account entities. The sweep runs every 30 seconds; returning to an expired thread starts a new active session with the scrubbed history. Choose and document an application retention/deletion policy before relying on the checkpoint store for production conversations.

The LangGraph agent registers both `check_balance` and `prepare_transfer`. The balance tool calls `GET {BALANCE_SERVICE_URL}/sandbox/balances/{JWT sub}` and forwards the user's validated bearer token. The transfer tool resolves the source and destination against the authenticated customer's linked sandbox accounts and emits a preview; it never confirms or executes the transfer. Both tools use the existing model tool-calling flow, and financial responses are emitted as structured chat events. `BALANCE_SERVICE_URL` defaults to `http://localhost:8001`.

Set `SANDBOX_MODE=true` for the balance, transfer, and orchestrator services to enable their sandbox routes. After the user confirms the preview, the UI posts to the orchestrator's `POST /transfer`, which forwards the user's validated token and `Idempotency-Key` to `{TRANSFER_SERVICE_URL}/transfer` (default `http://localhost:8002`). The transfer service validates that both accounts belong to the authenticated customer, requires `confirmed: true`, and verifies `otp_code` with `MFA_VERIFICATION_URL` (default `http://localhost:8003/verify-otp`) only when the amount is at least `HIGH_VALUE_TRANSFER_THRESHOLD` (default `1000.00`). It rejects frozen source accounts and returns the available balance with an `INSUFFICIENT_FUNDS` refusal. Set `SANDBOX_FROZEN_ACCOUNT_IDS` to a comma-separated list to mark sandbox accounts frozen during local testing. It appends `PENDING_VALIDATION` then `SUCCESS` or `FAILED` rows to PostgreSQL. The authenticated `GET /transfers/latest` route returns account IDs from the most recent successful transfer; the orchestrator resolves labels from the authenticated balance service's authorized account list for contextual chat follow-ups. Configure `DATABASE_URL` and apply the schema with `alembic upgrade head` before accepting transfers.

The MFA service is a local sandbox test verifier only. Start it with `SANDBOX_MODE=true`; it accepts the demo OTP `123456` and only allows loopback clients. It does not generate or store OTPs and must not be used outside local sandbox testing.

The included account balances and ledger are still in-memory sandbox fixtures, not live core-banking data. This sandbox transfer tool supports transfers between the customer's linked accounts only; external payees such as Alice require a payee directory and transfer-service contract that are not present here. Confirmation and OTP are collected by the UI and are never generated by the model. Production use still requires a trusted account-data/transfer backend; sandbox mode must not be enabled in production.

## Public FAQ cache

The orchestrator can cache public banking FAQs in Redis or Valkey. Set `REDIS_URL` (or `VALKEY_URL`) to enable the cache; the cache is otherwise a safe no-op. Cached answers use a SHA-256 key derived from the normalized question, and `FAQ_CACHE_TTL_SECONDS` controls expiration (default: 300 seconds). Public questions are limited to general informational prompts; balance, transaction history, transfer, payment, card, and other authenticated financial queries bypass the cache and call the real-time microservices.

```powershell
$env:REDIS_URL = "redis://localhost:6379/0"
$env:FAQ_CACHE_TTL_SECONDS = "300"
$env:FAQ_CACHE_PREFIX = "bank-chatbot:faq"
```