import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.server import app, session_manager
import json

client = TestClient(app)

def test_whatsapp_webhook():
    print("==================================================")
    print("Starting WhatsApp Webhook & Session Tests")
    print("==================================================")
    
    test_phone = "+919999999999"
    test_whatsapp_phone = f"whatsapp:{test_phone}"
    
    # Clean any old session
    session_manager.clear(test_phone)
    
    # --- Test 1: Send "hi" to Webhook ---
    print("\n--- Test 1: Send 'hi' to Webhook ---")
    response = client.post(
        "/whatsapp/webhook",
        data={
            "From": test_whatsapp_phone,
            "Body": "hi",
            "NumMedia": "0"
        }
    )
    assert response.status_code == 200, "Webhook should return 200"
    assert response.json() == {"status": "ok"}, "Response JSON should be ok"
    
    # Verify session history was saved
    session = session_manager.load(test_phone)
    assert len(session["history"]) == 2, "Session history should have 2 messages (user, assistant)"
    assert session["history"][0]["text"] == "hi"
    assert "ലീഗൽമൈൻഡിലേക്ക് സ്വാഗതം!" in session["history"][1]["text"]
    
    # --- Test 2: Send detailed issue (Greeting bypass) ---
    print("\n--- Test 2: Send detailed issue to Webhook ---")
    bypass_phone = "+918888888888"
    bypass_whatsapp_phone = f"whatsapp:{bypass_phone}"
    session_manager.clear(bypass_phone)
    
    response2 = client.post(
        "/whatsapp/webhook",
        data={
            "From": bypass_whatsapp_phone,
            "Body": "my company abc pvt ltd in bangalore hasnt paid me salary for 3 months. this started from march 2026",
            "NumMedia": "0"
        }
    )
    assert response2.status_code == 200
    session_bypass = session_manager.load(bypass_phone)
    assert len(session_bypass["history"]) == 2
    # Verify that it didn't return greeting but jumped to slot filling or RAG
    assert "ലീഗൽമൈൻഡിലേക്ക് സ്വാഗതം!" not in session_bypass["history"][1]["text"], "Greeting should be bypassed"
    
    # --- Test 3: Send Reset command ---
    print("\n--- Test 3: Send Reset Command ---")
    response3 = client.post(
        "/whatsapp/webhook",
        data={
            "From": test_whatsapp_phone,
            "Body": "reset",
            "NumMedia": "0"
        }
    )
    assert response3.status_code == 200
    session_cleared = session_manager.load(test_phone)
    assert len(session_cleared["history"]) == 0, "Session history should be empty after reset"
    
    print("\n==================================================")
    print("WHATSAPP WEBHOOK TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_whatsapp_webhook()
