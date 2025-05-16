import pytest
from main import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()

def test_chat_endpoint(client):
    resp = client.post(
        "/chat",
        json={"question": "Hello", "language": "en"}  # <-- sends application/json
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "answer" in body
