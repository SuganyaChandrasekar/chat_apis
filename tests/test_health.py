from fastapi.testclient import TestClient

from services.balance.main import app as balance_app
from services.orchestrator.main import app as orchestrator_app
from services.transfer.main import app as transfer_app


def test_each_service_has_a_health_endpoint() -> None:
    services = {
        "orchestrator": TestClient(orchestrator_app),
        "balance": TestClient(balance_app),
        "transfer": TestClient(transfer_app),
    }

    for service_name, client in services.items():
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"service": service_name, "status": "ok"}