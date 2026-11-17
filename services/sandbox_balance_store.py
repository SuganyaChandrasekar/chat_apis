import os
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from uuid import uuid4

from sqlalchemy import Numeric, String, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from services.sandbox import SANDBOX_CURRENCY, SANDBOX_INITIAL_BALANCES


class SandboxBalanceBase(DeclarativeBase):
    pass


class SandboxAccountBalance(SandboxBalanceBase):
    __tablename__ = "sandbox_accounts"

    account_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    balance: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)


class SandboxBalanceStoreUnavailable(Exception):
    pass


class SandboxAccountNotFound(Exception):
    pass


class SandboxInsufficientFunds(Exception):
    def __init__(self, available_balance: Decimal):
        self.available_balance = available_balance
        super().__init__("Insufficient sandbox balance")


@dataclass(frozen=True)
class SandboxBalance:
    account_id: str
    balance: Decimal
    currency: str


@lru_cache(maxsize=1)
def _engine():
    from sqlalchemy import create_engine

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SandboxBalanceStoreUnavailable
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


def _seed_accounts(session: Session) -> None:
    rows = [
        {
            "account_id": account_id,
            "balance": balance,
            "currency": SANDBOX_CURRENCY,
        }
        for account_id, balance in SANDBOX_INITIAL_BALANCES.items()
    ]
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "postgresql":
        statement = postgresql_insert(SandboxAccountBalance).values(rows)
    elif dialect_name == "sqlite":
        statement = sqlite_insert(SandboxAccountBalance).values(rows)
    else:
        raise SandboxBalanceStoreUnavailable
    session.execute(
        statement.on_conflict_do_nothing(index_elements=["account_id"])
    )


def get_sandbox_balance(account_id: str) -> SandboxBalance:
    try:
        with Session(_engine()) as session, session.begin():
            _seed_accounts(session)
            account = session.get(SandboxAccountBalance, account_id)
            if account is None:
                raise SandboxAccountNotFound
            return SandboxBalance(
                account_id=account.account_id,
                balance=account.balance,
                currency=account.currency,
            )
    except SandboxAccountNotFound:
        raise
    except SandboxBalanceStoreUnavailable:
        raise
    except Exception:
        raise SandboxBalanceStoreUnavailable from None


def execute_sandbox_transfer(
    *,
    transaction_id: str,
    account_id: str,
    idempotency_key: str,
    source_account_id: str,
    destination_account_id: str,
    amount: Decimal,
    currency: str,
    sca_state: str,
) -> None:
    from services.transfer.audit import TransferAuditEvent

    try:
        with Session(_engine()) as session, session.begin():
            _seed_accounts(session)
            rows = session.scalars(
                select(SandboxAccountBalance)
                .where(
                    SandboxAccountBalance.account_id.in_(
                        (source_account_id, destination_account_id)
                    )
                )
                .order_by(SandboxAccountBalance.account_id)
                .with_for_update()
            ).all()
            accounts = {row.account_id: row for row in rows}
            source = accounts.get(source_account_id)
            destination = accounts.get(destination_account_id)
            if source is None or destination is None:
                raise SandboxAccountNotFound
            if source.balance < amount:
                raise SandboxInsufficientFunds(source.balance)

            source.balance -= amount
            destination.balance += amount
            session.add(
                TransferAuditEvent(
                    event_id=str(uuid4()),
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    account_id=account_id,
                    source_account_id=source_account_id,
                    destination_account_id=destination_account_id,
                    amount=amount,
                    currency=currency,
                    status="SUCCESS",
                    policy_decision="ALLOW",
                    consent_state="GRANTED",
                    sca_state=sca_state,
                )
            )
    except (SandboxAccountNotFound, SandboxInsufficientFunds):
        raise
    except SandboxBalanceStoreUnavailable:
        raise
    except Exception:
        raise SandboxBalanceStoreUnavailable from None
