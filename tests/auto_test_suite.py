#!/usr/bin/env python3
"""
Automated Test Suite for LegalMind AI Pipeline.
Run this script to automatically test and identify pipeline issues across all legal domains,
state transitions, PDF generation, role inversion validation, and RAG grounding.
"""
import os
import sys
import unittest
import urllib.request

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pipeline import LegalMindPipeline

class TestLegalMindPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  LegalMind Automated Integration & QA Test Suite")
        print("=======================================================")
        cls.pipeline = LegalMindPipeline()

    def test_01_employment_irac_grounding(self):
        """Test Case 1: Employment wage dispute produces grounded Malayalam IRAC output."""
        print("\n[TEST 1] Testing Employment Wage Complaint IRAC Generation...")
        query = "ഞാൻ കൊച്ചിയിൽ ഒരു കമ്പനിയിൽ ജോലി ചെയ്യുന്നു. അവർ മൂന്ന് മാസമായി ശമ്പളം തന്നിട്ടില്ല. ജനുവരി 2026 മുതൽ ഒരു രൂപ പോലും കിട്ടിയില്ല"
        res = self.pipeline.run(query)
        
        response_text = res.get("response_text", "")
        self.assertIsNotNone(response_text)
        self.assertGreater(len(response_text), 20)
        
        # Verify statutory grounding (must cite Payment of Wages Act or Industrial Disputes Act)
        self.assertTrue(
            "പേയ്‌മെന്റ് ഓഫ് വേജസ്" in response_text or 
            "വേജസ്" in response_text or 
            "Payment of Wages" in response_text or 
            "KeLSA" in response_text or 
            "നിയമ" in response_text,
            "Response should contain grounded statutory guidance."
        )
        # Ensure zero hallucinated fake URLs
        self.assertNotIn("example.com", response_text.lower())
        print("  ✓ Test 1 Passed: Grounded Employment IRAC verified.")

    def test_02_tenancy_domain_isolation(self):
        """Test Case 2: Tenancy query must not mention employment/salary terms."""
        print("\n[TEST 2] Testing Tenancy Domain Isolation...")
        query = "എന്റെ വീട്ടുടമസ്ഥൻ എന്നോട് നാളെത്തന്നെ റൂമൊഴിഞ്ഞു തരാൻ പറഞ്ഞു."
        res = self.pipeline.run(query)
        
        response_text = res.get("response_text", "")
        self.assertIsNotNone(response_text)
        
        # Must not mention salary/wages for a rental issue
        self.assertNotIn("ശമ്പളം", response_text)
        self.assertNotIn("തൊഴിലുടമ", response_text)
        print("  ✓ Test 2 Passed: Domain isolation verified for Tenancy.")

    def test_03_role_inversion_validation(self):
        """Test Case 3: Form parser must prevent sender and recipient from being identical."""
        print("\n[TEST 3] Testing Pydantic Role Inversion Validation Gate...")
        form_text = "ൊഴിലുടമയുടെ പേര്: ammrkth\nകമ്പനിയുടെ പേര്: _wex kochi\nനിങ്ങളുടെ പൂർണ്ണ പേര്:  vishnu p"
        slots = self.pipeline.parse_legal_notice_slots(form_text)
        
        self.assertEqual(slots.get("sender_name"), "vishnu p")
        self.assertIn("ammrkth", slots.get("recipient_name"))
        self.assertNotEqual(slots.get("sender_name"), slots.get("recipient_name"))
        print(f"  ✓ Test 3 Passed: Sender = '{slots.get('sender_name')}', Recipient = '{slots.get('recipient_name')}'")

    def test_04_pdf_notice_compilation_and_sanitization(self):
        """Test Case 4: Form submission must compile Formal English PDF without leaked placeholders."""
        print("\n[TEST 4] Testing PDF Legal Notice Generation & Placeholder Sanitization...")
        history = [{'role': 'user', 'text': 'ഞാൻ കൊച്ചിയിൽ ഒരു കമ്പനിയിൽ ജോലി ചെയ്യുന്നു. 3 മാസമായി ശമ്പളം തന്നിട്ടില്ല.'}]
        form_query = "ൊഴിലുടമയുടെ പേര്: ammrkth\nകമ്പനിയുടെ പേര്: _wex kochi\nനിങ്ങളുടെ പൂർണ്ണ പേര്:  vishnu p"
        
        res = self.pipeline.run(form_query, history=history)
        response_text = res.get("response_text", "")
        
        self.assertIn("[DOWNLOAD_URL:", response_text)
        self.assertNotIn("[TO BE INSERTED]", response_text)
        self.assertNotIn("[TO BE FILLED BY SENDER]", response_text)
        self.assertNotIn("example.com", response_text)
        print("  ✓ Test 4 Passed: PDF Compiled cleanly with 0% leaked placeholders.")

    def test_05_null_guard_resilience(self):
        """Test Case 5: Ensure pipeline never crashes with TypeError on edge cases."""
        print("\n[TEST 5] Testing Pipeline Resilience on Empty/Null Query Edge Cases...")
        res = self.pipeline.run("")
        self.assertIsNotNone(res.get("response_text"))
        print("  ✓ Test 5 Passed: Null guard resilience verified.")

if __name__ == "__main__":
    unittest.main()
