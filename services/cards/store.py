import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from sqlalchemy import CheckConstraint, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class CardBase(DeclarativeBase):
    pass


class Card(CardBase):
    __tablename__ = "cards"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'frozen')",
            name="ck_cards_status",
        ),
    )

    card_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    card_name: Mapped[str] = mapped_column(String(256), nullable=False)
    last_4: Mapped[str] = mapped_column(String(4), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", server_default="active"
    )


class CardStoreUnavailable(Exception):
    pass


@dataclass(frozen=True)
class CardRecord:
    card_id: str
    user_id: str
    card_name: str
    last_4: str
    status: Literal["active", "frozen"]


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise CardStoreUnavailable
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


@lru_cache(maxsize=1)
def _engine():
    return create_engine(
        _database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


def _record(card: Card) -> CardRecord:
    return CardRecord(
        card_id=card.card_id,
        user_id=card.user_id,
        card_name=card.card_name,
        last_4=card.last_4,
        status=card.status,
    )


def list_user_cards(user_id: str) -> list[CardRecord]:
    try:
        with Session(_engine()) as session:
            cards = session.scalars(
                select(Card).where(Card.user_id == user_id).order_by(Card.card_id)
            )
            return [_record(card) for card in cards]
    except Exception:
        raise CardStoreUnavailable from None


def get_owned_card(user_id: str, card_id: str) -> CardRecord | None:
    try:
        with Session(_engine()) as session:
            card = session.scalar(
                select(Card).where(
                    Card.user_id == user_id,
                    Card.card_id == card_id,
                )
            )
            return _record(card) if card is not None else None
    except Exception:
        raise CardStoreUnavailable from None


def set_card_status(
    user_id: str,
    card_id: str,
    status: Literal["active", "frozen"],
) -> CardRecord | None:
    try:
        with Session(_engine()) as session:
            card = session.scalar(
                select(Card).where(
                    Card.user_id == user_id,
                    Card.card_id == card_id,
                )
            )
            if card is None:
                return None
            card.status = status
            session.commit()
            return _record(card)
    except Exception:
        raise CardStoreUnavailable from None
