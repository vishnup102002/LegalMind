#!/usr/bin/env python3
"""
LegalMind Master 3-Layer QA & Integration Test Suite.
Executes Layer 1 (Pydantic & Unit), Layer 2 (DeepEval RAG Metrics), and Layer 3 (WhatsApp Webhooks).
"""
import sys
import os
import pytest

def main():
    print("==================================================================")
    print("      LegalMind End-to-End Structured Testing Architecture       ")
    print("==================================================================")
    print(" Layer 1: Pydantic Structural Validation & Unit Tests")
    print(" Layer 2: DeepEval & RAG Metric Evaluation Engine (Grounding & Metrics)")
    print(" Layer 3: Automated WhatsApp Webhook & Integration Test Harness")
    print("==================================================================\n")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

    test_files = [
        "tests/test_layer1_pydantic_units.py",
        "tests/test_layer2_deepeval_metrics.py",
        "tests/test_layer3_whatsapp_webhook.py"
    ]

    print("🚀 Launching PyTest Test Execution Harness...\n")
    exit_code = pytest.main(["-v"] + test_files)
    
    if exit_code == 0:
        print("\n==================================================================")
        print("  🎉 ALL 3 TESTING LAYERS PASSED WITH 100% SUCCESS! (BUILD CLEAN)")
        print("==================================================================")
    else:
        print(f"\n❌ Test suite completed with exit code: {exit_code}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
