import logging
import os
import re
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, ValidationError

from services.authentication import create_access_token
from services.sandbox import SANDBOX_CARD_OWNER_BY_ACCOUNT_ID

logger = logging.getLogger("uvicorn.error")

_HANDOFF_LABEL = "24/7 emergency fraud hotline"
_HANDOFF_URL = "https://albionbank.com/sandbox/fraud-support"


class _CardSummary(BaseModel):
    card_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    last_4: str = Field(pattern=r"^\d{4}$")
    status: Literal["active", "frozen"]


class _FreezeResult(BaseModel):
    frozen_card_ids: list[str]
    newly_frozen_card_ids: list[str] = Field(default_factory=list)
    already_frozen_card_ids: list[str] = Field(default_factory=list)


class _FraudCaseResult(BaseModel):
    case_id: str = Field(pattern=r"^FRD-\d{4}-[A-F0-9]{8}$")
    user_id: str = Field(min_length=1, max_length=256)
    status: Literal["OPEN"]
    frozen_card_ids: list[str]


def _matching_card_ids(
    card_hint: str, cards: list[_CardSummary]
) -> list[str]:
    hint = " ".join(card_hint.casefold().split())
    matches = []
    for card in cards:
        name = " ".join(card.name.casefold().split())
        name_words = set(re.findall(r"[a-z0-9]+", name)) - {"card"}
        aliases = {card.card_id.casefold(), card.last_4, name}
        aliases.update(name_words)
        if hint in aliases or any(
            re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", hint)
            for alias in aliases
            if len(alias) >= 3
        ):
            matches.append(card.card_id)
    return matches


async def report_sandbox_fraud(
    *,
    access_token: str,
    account_id: str,
    reason: str,
    card_hint: str | None,
) -> str:
    if os.getenv("SANDBOX_MODE", "false").lower() != "true":
        return (
            "I'm sorry, I can't open a fraud case from this chat. Please contact "
            "the bank through its official support channels."
        )

    cards_url = os.getenv("CARDS_SERVICE_URL", "http://localhost:8004").rstrip("/")
    fraud_url = os.getenv("FRAUD_SERVICE_URL", "http://localhost:8005").rstrip("/")
    frozen_card_ids: list[str] = []
    newly_frozen_card_ids: list[str] = []
    already_frozen_card_ids: list[str] = []
    card_freeze_failed = False

    card_user_id = SANDBOX_CARD_OWNER_BY_ACCOUNT_ID.get(account_id)
    if card_user_id is not None:
        try:
            card_token = create_access_token(card_user_id, ttl_seconds=60)
            async with httpx.AsyncClient(timeout=5.0) as client:
                target_ids = None
                if card_hint:
                    cards_response = await client.get(
                        f"{cards_url}/sandbox/cards",
                        headers={"Authorization": f"Bearer {card_token}"},
                    )
                    cards_response.raise_for_status()
                    cards_payload = cards_response.json()
                    if not isinstance(cards_payload, list):
                        raise ValueError("Invalid cards response")
                    cards = [_CardSummary.model_validate(item) for item in cards_payload]
                    target_ids = _matching_card_ids(card_hint, cards)
                    if len(target_ids) != 1:
                        card_freeze_failed = True
                        target_ids = []

                if target_ids is None or target_ids:
                    freeze_response = await client.post(
                        f"{cards_url}/sandbox/fraud/freeze",
                        headers={"Authorization": f"Bearer {card_token}"},
                        json={"card_ids": target_ids},
                    )
                    freeze_response.raise_for_status()
                    freeze_result = _FreezeResult.model_validate(freeze_response.json())
                    if target_ids is not None and not set(
                        freeze_result.frozen_card_ids
                    ).issubset(target_ids):
                        raise ValueError("Cards service returned an unexpected card")
                    if set(freeze_result.newly_frozen_card_ids).intersection(
                        freeze_result.already_frozen_card_ids
                    ) or set(freeze_result.frozen_card_ids) != (
                        set(freeze_result.newly_frozen_card_ids)
                        | set(freeze_result.already_frozen_card_ids)
                    ):
                        raise ValueError("Cards service returned inconsistent freeze status")
                    frozen_card_ids = freeze_result.frozen_card_ids
                    newly_frozen_card_ids = freeze_result.newly_frozen_card_ids
                    already_frozen_card_ids = freeze_result.already_frozen_card_ids
        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            ValueError,
            TypeError,
            ValidationError,
        ) as error:
            card_freeze_failed = True
            logger.warning(
                "Sandbox fraud card safety action failed (%s)",
                type(error).__name__,
            )
    else:
        card_freeze_failed = True

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            case_response = await client.post(
                f"{fraud_url}/fraud/cases",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Idempotency-Key": str(uuid4()),
                },
                json={
                    "reason": reason,
                    "frozen_card_ids": frozen_card_ids,
                },
            )
            case_response.raise_for_status()
            case = _FraudCaseResult.model_validate(case_response.json())
            if case.user_id != account_id or case.frozen_card_ids != frozen_card_ids:
                raise ValueError("Fraud service returned a mismatched case")
    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        ValueError,
        TypeError,
        ValidationError,
    ) as error:
        logger.warning(
            "Sandbox fraud case creation failed (%s)",
            type(error).__name__,
        )
        return (
            "I'm sorry you're dealing with this. I couldn't finish opening a fraud "
            "case right now. "
            + (
                "I also couldn't confirm that a card was frozen. "
                if card_freeze_failed or not frozen_card_ids
                else "Your card(s) are frozen. "
            )
            + f"Contact our {_HANDOFF_LABEL} at {_HANDOFF_URL}."
        )

    if frozen_card_ids and not card_freeze_failed:
        if newly_frozen_card_ids and already_frozen_card_ids:
            freeze_confirmation = (
                "I froze the card(s) that were active and confirmed the remaining "
                "reported card(s) were already frozen."
            )
        elif newly_frozen_card_ids:
            freeze_confirmation = "I have immediately frozen your card(s)"
        else:
            freeze_confirmation = "I confirmed your card(s) are already frozen"
        return (
            f"{freeze_confirmation} and opened Fraud Case #{case.case_id} to keep "
            f"your account safe. A member of our specialist "
            f"team is ready to help you. Visit {_HANDOFF_URL} to connect with a "
            f"specialist. Our {_HANDOFF_LABEL} is available."
        )
    return (
        f"I'm sorry you're dealing with this. I opened Fraud Case #{case.case_id} "
        "and logged it for our specialist team. I couldn't confirm that a card was "
        f"frozen, so please contact our {_HANDOFF_LABEL} at {_HANDOFF_URL}. "
        f"Your case reference #{case.case_id} has been logged and prioritized."
    )
