from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from fastapi import Request


@dataclass(frozen=True)
class AuthenticatedCustomer:
    """Customer identity established by a trusted authentication adapter."""

    subject: str


BalanceT = TypeVar("BalanceT")


class CustomerAuthenticator(Protocol):
    async def authenticate(self, request: Request) -> AuthenticatedCustomer: ...


class BalanceProvider(Protocol, Generic[BalanceT]):
    async def get_balance(self, customer: AuthenticatedCustomer) -> BalanceT: ...


class BalanceIntegrationUnavailable(Exception):
    pass


class UnconfiguredCustomerAuthenticator:
    async def authenticate(self, request: Request) -> AuthenticatedCustomer:
        raise BalanceIntegrationUnavailable


class UnconfiguredBalanceProvider:
    async def get_balance(self, customer: AuthenticatedCustomer) -> object:
        raise BalanceIntegrationUnavailable