import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)

from app.pipeline import LegalMindPipeline
import json

def run_tests():
    pipeline = LegalMindPipeline()
    print("==================================================")
    print("Starting LegalMind Pipeline Integration Tests")
    print("==================================================")

    # ----------------------------------------------------
    # Test 1: Initial Greeting Check
    # ----------------------------------------------------
    print("\n--- Test 1: Initial Greeting ---")
    history = []
    res1 = pipeline.run("hi", history)
    print("Status:", res1["status"])
    print("Response text:\n", res1["response_text"])
    assert "ലീഗൽമൈൻഡിലേക്ക് സ്വാഗതം!" in res1["response_text"], "Greeting should be in response"
    history.append({"role": "user", "text": "hi"})
    history.append({"role": "assistant", "text": res1["response_text"]})

    # ----------------------------------------------------
    # Test 2: Slot Gathering - Part 1 (Incident Description)
    # ----------------------------------------------------
    print("\n--- Test 2: Ingesting Issue (GATHERING State) ---")
    res2 = pipeline.run("i face ragging at my college", history)
    print("Status:", res2["status"])
    print("Response text:\n", res2["response_text"])
    # The output should NOT repeat the welcome greeting, since we have history!
    assert "ലീഗൽമൈൻഡിലേക്ക് സ്വാഗതം!" not in res2["response_text"], "Greeting should NOT be repeated in history questions"
    history.append({"role": "user", "text": "i face ragging at my college"})
    history.append({"role": "assistant", "text": res2["response_text"]})

    # ----------------------------------------------------
    # Test 3: Slot Gathering - Part 2 (Jurisdiction & Date) -> RAG_EVALUATION
    # ----------------------------------------------------
    print("\n--- Test 3: Providing State and Date (RAG_EVALUATION) ---")
    res3 = pipeline.run("it happened yesterday in kerala", history)
    print("Status:", res3["status"])
    print("Response text:\n", res3["response_text"])
    
    # Assert that the Tamil Nadu Prohibition of Ragging Act was NOT cited since the user is in Kerala
    assert "Tamil Nadu" not in res3["response_text"], "Should not cite Tamil Nadu Prohibition of Ragging Act for Kerala user!"
    
    # Assert Malayalam layperson translation section exists
    assert "LAYPERSON_ML:" in res3["response_text"], "Should contain LAYPERSON_ML section"
    assert "LEGAL ROADMAP" in res3["response_text"], "Should contain LEGAL ROADMAP"
    
    history.append({"role": "user", "text": "it happened yesterday in kerala"})
    history.append({"role": "assistant", "text": res3["response_text"]})

    # ----------------------------------------------------
    # Test 4: Affirmative Response -> NOTICE_DRAFTING (Missing Names)
    # ----------------------------------------------------
    print("\n--- Test 4: Saying 'yes' (Transition to NOTICE_DRAFTING) ---")
    res4 = pipeline.run("yes", history)
    print("Status:", res4["status"])
    print("Response text:\n", res4["response_text"])
    assert "names of the Sender and the Recipient" in res4["response_text"] or "പേരുകൾ" in res4["response_text"], "Should ask for names"
    
    history.append({"role": "user", "text": "yes"})
    history.append({"role": "assistant", "text": res4["response_text"]})

    # ----------------------------------------------------
    # Test 5: Providing Names -> NOTICE_DRAFTING (Generates Notice)
    # ----------------------------------------------------
    print("\n--- Test 5: Providing Names to Draft Notice ---")
    res5 = pipeline.run("sender is Vishnu and recipient is Kavya", history)
    print("Status:", res5["status"])
    print("Response text:\n", res5["response_text"])
    assert "DOWNLOAD_URL" in res5["response_text"], "Should generate document download link"
    assert "Vishnu" in res5["response_text"], "Should include sender name"
    assert "Kavya" in res5["response_text"], "Should include recipient name"
    
    # ----------------------------------------------------
    # Test 6: Greeting Bypass (Detailed first message)
    # ----------------------------------------------------
    print("\n--- Test 6: Greeting Bypass (Detailed First Message) ---")
    bypass_history = []
    res_bypass = pipeline.run("my company abc pvt ltd in bangalore hasnt paid me salary for 3 months. this started from march 2026", bypass_history)
    print("Status:", res_bypass["status"])
    print("Response text:\n", res_bypass["response_text"])
    assert "ലീഗൽമൈൻഡിലേക്ക് സ്വാഗതം!" not in res_bypass["response_text"], "Should bypass initial greeting if details provided on turn 1"
    
    # ----------------------------------------------------
    # Test 7: Post-IRAC Follow-up (No Slot Reset Loop)
    # ----------------------------------------------------
    print("\n--- Test 7: Post-IRAC Follow-up (No Slot Reset Loop) ---")
    # We will use history from Test 3 where IRAC was delivered
    followup_history = []
    for turn in history[:6]:
        followup_history.append(turn)
    
    res_followup = pipeline.run("can you show me exact law", followup_history)
    print("Status:", res_followup["status"])
    print("Response text:\n", res_followup["response_text"])
    assert "when" not in res_followup["response_text"].lower() and "where" not in res_followup["response_text"].lower(), "Should not ask for slots again in follow-up"
    assert "പേരുകൾ" not in res_followup["response_text"] and "names of the" not in res_followup["response_text"].lower(), "Should not ask for names in general follow-up"
    
    print("\n==================================================")
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
