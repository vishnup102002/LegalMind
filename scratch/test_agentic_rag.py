import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalMind.Test.AgenticRAG")

from app.pipeline import LegalMindPipeline
import json

def run_test():
    pipeline = LegalMindPipeline(threshold=0.72)
    print("==================================================")
    print("Starting Agentic RAG Dynamic Ingestion Test")
    print("==================================================")
    
    # We query for an issue not in the local database (Transgender Persons (Protection of Rights) Act, 2019).
    # Since we provide all details, it should bypass slot gathering and trigger retrieval.
    query = "My college abc in Ernakulam, Kerala has discriminated against me because of my transgender identity yesterday, violating the Transgender Persons Protection of Rights Act 2019. Please advise."
    history = []
    
    print(f"\nSending query: '{query}'")
    res = pipeline.run(query, history, threshold=0.99)
    
    print("\n------------------ PIPELINE OUTPUT ------------------")
    print("Status:", res["status"])
    print("Response:\n", res["response_text"])
    print("-----------------------------------------------------")
    
    # Assertions:
    assert res["status"] == "SUCCESS", "Pipeline should succeed after agentic research and ingestion!"
    assert "Transgender" in res["response_text"] or "transgender" in res["response_text"].lower(), "Roadmap should contain references to transgender rights!"
    assert "LEGAL ROADMAP" in res["response_text"], "Roadmap should be in formal IRAC format!"
    assert "LAYPERSON_ML:" in res["response_text"], "Should contain layperson advice in Malayalam!"
    
    print("\n==================================================")
    print("AGENTIC RAG FALLBACK TEST PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_test()
