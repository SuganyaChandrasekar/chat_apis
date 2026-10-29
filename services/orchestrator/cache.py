import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod

from redis.asyncio import Redis

logger = logging.getLogger("uvicorn.error")

DEFAULT_FAQ_CACHE_TTL_SECONDS = 300
DEFAULT_FAQ_CACHE_PREFIX = "bank-chatbot:faq"

_PUBLIC_QUERY_PATTERN = re.compile(
    r"\b(?:what is|how does|how do|why does|why do|what are|how can|describe|explain)\b"
)
_PERSONAL_QUERY_PATTERN = re.compile(
    r"\b(?:my|me|mine)\s+(?:balance|account|transaction|transfer|payment|purchase|order|card|statement|invoice|receipt|fee|debit|credit)\b"
    r"|\b(?:balance|account|transaction history|transaction|transfer|payment|purchase|order|card|statement|invoice|receipt)\s+(?:of|for|on|in)\s+(?:my|me|mine)\b"
    r"|\b(?:show|check|find|view|get)\s+(?:my|me|mine)\s+(?:balance|account|transaction|transfer|payment|purchase|order|card|statement|invoice|receipt)\b"
    r"|\b(?:i|we)\s+(?:want|need|have|can|could|should|will)\s+(?:to\s+)?(?:check|view|get|find|know|see|make|send|receive|complete|review|track)\s+(?:my|the)?\s*(?:balance|account|transaction|transfer|payment|purchase|order|card|statement|invoice|receipt)\b"
    r"|\b(?:i|we)\s+(?:balance|check|view|get|find|review|track|transfer|send|receive|make|complete)\s+(?:my|the)?\s*(?:balance|account|transaction|transfer|payment|purchase|order|card|statement|invoice|receipt)\b"
    r"|\b(?:how do|how can)\s+(?:i|we)\s+(?:transfer|send|receive|make|complete|review|track|check|view|get|find)\b"
    r"|\b(?:balance|account|transaction|transfer|payment|purchase|order|card|statement|invoice|receipt)\s+(?:on|for)\s+account\b"
)


class FAQCache(ABC):
    @property
    @abstractmethod
    def ttl_seconds(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def get(self, question: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    async def set(self, question: str, answer: str, ttl_seconds: int) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class NoOpFAQCache(FAQCache):
    @property
    def ttl_seconds(self) -> int:
        return 0

    async def get(self, question: str) -> str | None:
        return None

    async def set(self, question: str, answer: str, ttl_seconds: int) -> None:
        return None

    async def close(self) -> None:
        return None


class RedisFAQCache(FAQCache):
    def __init__(self, client: Redis, prefix: str, ttl_seconds: int) -> None:
        self._client = client
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def _key(self, question: str) -> str:
        digest = hashlib.sha256(question.casefold().encode("utf-8")).hexdigest()
        return f"{self._prefix}:{digest}"

    async def get(self, question: str) -> str | None:
        return await self._client.get(self._key(question))

    async def set(self, question: str, answer: str, ttl_seconds: int) -> None:
        await self._client.set(self._key(question), answer, ex=ttl_seconds)

    async def close(self) -> None:
        await self._client.aclose()


def _cache_ttl_seconds() -> int:
    raw_value = os.getenv("FAQ_CACHE_TTL_SECONDS", str(DEFAULT_FAQ_CACHE_TTL_SECONDS))
    try:
        ttl_seconds = int(raw_value)
    except ValueError:
        logger.warning("Invalid FAQ_CACHE_TTL_SECONDS; using default")
        return DEFAULT_FAQ_CACHE_TTL_SECONDS
    if ttl_seconds <= 0:
        logger.warning("FAQ_CACHE_TTL_SECONDS must be positive; using default")
        return DEFAULT_FAQ_CACHE_TTL_SECONDS
    return ttl_seconds


def is_cacheable_public_question(question: str) -> bool:
    """Return whether a question is safe to serve from a shared FAQ cache."""
    normalized = " ".join(question.casefold().split())
    if not _PUBLIC_QUERY_PATTERN.search(normalized):
        return False
    return not _PERSONAL_QUERY_PATTERN.search(normalized)


def create_faq_cache() -> FAQCache:
    redis_url = os.getenv("REDIS_URL") or os.getenv("VALKEY_URL")
    if not redis_url:
        return NoOpFAQCache()

    try:
        normalized_url = redis_url.removeprefix("valkey://")
        client = Redis.from_url(normalized_url, decode_responses=True)
        return RedisFAQCache(
            client=client,
            prefix=os.getenv("FAQ_CACHE_PREFIX", DEFAULT_FAQ_CACHE_PREFIX),
            ttl_seconds=_cache_ttl_seconds(),
        )
    except Exception:
        logger.error("Unable to configure the FAQ cache", exc_info=True)
        return NoOpFAQCache()


async def ping_faq_cache(cache: FAQCache) -> bool:
    if not isinstance(cache, RedisFAQCache):
        return True
    try:
        return bool(await cache._client.ping())
    except Exception:
        logger.error("Unable to connect to the FAQ cache", exc_info=True)
        return False
