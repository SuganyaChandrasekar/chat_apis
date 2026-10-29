from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from services.api_errors import register_exception_handlers
from services.authentication import authenticated_subject
from services.balance.contracts import (
    AuthenticatedCustomer,
    BalanceIntegrationUnavailable,
    BalanceProvider,
    UnconfiguredBalanceProvider,
    UnconfiguredCustomerAuthenticator,
)
from services.sandbox import (
    SANDBOX_ACCOUNT_LABELS,
    SANDBOX_CUSTOMER_ACCOUNTS,
    SANDBOX_CURRENCY,
    SANDBOX_INITIAL_BALANCES,
    SandboxAccount,
    SandboxBalance,
    require_sandbox_mode,
    sandbox_account_status,
)

app = FastAPI(title="Balance Service", version="0.1.0")
register_exception_handlers(app)
customer_authenticator = UnconfiguredCustomerAuthenticator()
balance_provider = UnconfiguredBalanceProvider()


async def get_authenticated_customer(request: Request) -> AuthenticatedCustomer:
    try:
        return await customer_authenticator.authenticate(request)
    except BalanceIntegrationUnavailable:
        raise HTTPException(
            status_code=503, detail="Customer authentication is not configured"
        ) from None


def get_balance_provider() -> BalanceProvider[Any]:
    return balance_provider


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"service": "balance", "status": "ok"}


@app.get("/balance", tags=["balance"])
async def get_balance(
    customer: AuthenticatedCustomer = Depends(get_authenticated_customer),
    provider: BalanceProvider[Any] = Depends(get_balance_provider),
) -> Any:
    try:
        return await provider.get_balance(customer)
    except BalanceIntegrationUnavailable:
        raise HTTPException(
            status_code=503, detail="Balance retrieval is not configured"
        ) from None


@app.get(
    "/sandbox/balances/{account_id}",
    response_model=SandboxBalance,
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def get_sandbox_balance(
    account_id: str,
    authenticated_account_id: str = Depends(authenticated_subject),
) -> SandboxBalance:
    authorized_accounts = SANDBOX_CUSTOMER_ACCOUNTS.get(authenticated_account_id, ())
    if account_id not in authorized_accounts:
        raise HTTPException(status_code=403, detail="Account access is not authorized")
    balance = SANDBOX_INITIAL_BALANCES.get(account_id)
    if balance is None:
        raise HTTPException(status_code=404, detail="Sandbox account not found")
    return SandboxBalance(
        account_id=account_id, balance=balance, currency=SANDBOX_CURRENCY
    )


@app.get(
    "/sandbox/accounts",
    response_model=list[SandboxAccount],
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def list_sandbox_accounts(
    authenticated_customer_id: str = Depends(authenticated_subject),
) -> list[SandboxAccount]:
    authorized_accounts = SANDBOX_CUSTOMER_ACCOUNTS.get(authenticated_customer_id)
    if authorized_accounts is None:
        raise HTTPException(status_code=403, detail="Account access is not authorized")
    return [
        SandboxAccount(
            account_id=account_id,
            label=SANDBOX_ACCOUNT_LABELS[account_id],
            status=sandbox_account_status(account_id),
        )
        for account_id in authorized_accounts
    ]