import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"

@pytest.fixture(scope="module")
def api_base_url():
    return BASE_URL

# Test /chat endpoint with valid question
def test_chat_valid(api_base_url):
    question = "What is AI?"
    response = requests.post(f"{api_base_url}/chat", json={"question": question})
    assert response.status_code == 200
    answer = response.json().get("answer", "")
    assert isinstance(answer, str) and len(answer) > 0

# Test /chat endpoint with empty question
def test_chat_empty(api_base_url):
    response = requests.post(f"{api_base_url}/chat", json={"question": ""})
    assert response.status_code == 200
    answer = response.json().get("answer", "")
    assert isinstance(answer, str)

# Test /chat endpoint with missing question field
def test_chat_missing_field(api_base_url):
    response = requests.post(f"{api_base_url}/chat", json={})
    assert response.status_code in [400, 422]

# Test /logs endpoint
def test_logs(api_base_url):
    response = requests.get(f"{api_base_url}/logs")
    assert response.status_code == 200
    logs = response.json().get("logs", [])
    assert isinstance(logs, list)

# Test /chat and /logs integration
def test_chat_and_logs_integration(api_base_url):
    question = "Define Generative AI."
    requests.post(f"{api_base_url}/chat", json={"question": question})
    logs_response = requests.get(f"{api_base_url}/logs")
    logs = logs_response.json().get("logs", [])
    assert any("Generative AI" in str(log) for log in logs)
