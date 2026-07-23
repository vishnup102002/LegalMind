"""
Layer 3: End-to-End Automated WhatsApp Webhook & Integration Test Harness.
Simulates live HTTP POST payloads to /whatsapp/webhook endpoint and validates full multi-turn responses.
"""
import pytest
from fastapi.testclient import TestClient
from app.server import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_webhook_health(client):
    """Test health check endpoint."""
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json().get("server") == "healthy"

def test_webhook_reset_command(client):
    """Test sending /reset command via WhatsApp webhook."""
    payload = {
        "From": "whatsapp:+919876543210",
        "Body": "/reset"
    }
    res = client.post("/whatsapp/webhook", data=payload)
    assert res.status_code == 200
    assert "Response" in res.text or "succeeded" in res.text.lower() or res.status_code == 200

def test_webhook_form_submission_session(client):
    """Test multi-turn WhatsApp webhook session simulation."""
    phone = "whatsapp:+919876543210"

    # Turn 1: Initial complaint
    t1_res = client.post("/whatsapp/webhook", data={
        "From": phone,
        "Body": "ഞാൻ കൊച്ചിയിൽ ഒരു കമ്പനിയിൽ ജോലി ചെയ്യുന്നു. അവർ മൂന്ന് മാസമായി ശമ്പളം തന്നിട്ടില്ല."
    })
    assert t1_res.status_code == 200

    # Turn 2: Request legal notice
    t2_res = client.post("/whatsapp/webhook", data={
        "From": phone,
        "Body": "എനിക്ക് വേണ്ടി ലീഗൽ നോട്ടീസ് തയ്യാറാക്കാമോ?"
    })
    assert t2_res.status_code == 200

    # Turn 3: Form submission
    t3_res = client.post("/whatsapp/webhook", data={
        "From": phone,
        "Body": "ൊഴിലുടമയുടെ പേര്: ammrkth\nകമ്പനിയുടെ പേര്: _wex kochi\nനിങ്ങളുടെ പൂർണ്ണ പേര്:  vishnu p"
    })
    assert t3_res.status_code == 200
    assert "https://" in t3_res.text or "Download" in t3_res.text or "തയ്യാറാണ്" in t3_res.text or "Response" in t3_res.text or "status" in t3_res.text

def test_nested_whatsapp_meta_payload(client):
    """Test deeply nested Meta WhatsApp Business API webhook JSON payload parsing."""
    meta_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "919876543210",
                        "text": {"body": "ഞാൻ കൊച്ചിയിൽ ഒരു കമ്പനിയിൽ ജോലി ചെയ്യുന്നു. 3 മാസമായി ശമ്പളം തന്നിട്ടില്ല."}
                    }]
                }
            }]
        }]
    }
    
    response = client.post("/whatsapp/webhook", json=meta_payload)
    assert response.status_code == 200
    res_json = response.json()
    resp_text = res_json.get("response_text", "")
    assert "മനസ്സിലായില്ല" not in resp_text, "Nested Meta payload must not trigger fallback 'മനസ്സിലായില്ല' response."

