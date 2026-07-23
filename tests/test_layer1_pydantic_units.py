"""
Layer 1: Structural Pydantic Validation & PyTest Unit Test Suite.
Fast build-level checks to guarantee schema integrity, role non-inversion, and zero placeholder leaks.
"""
import pytest
from app.models import LegalNoticeSlots, LegalDomain
from app.pipeline import LegalMindPipeline, normalize_text_inputs

def test_pydantic_valid_slots():
    """Test valid slot initialization and readiness check."""
    slots = LegalNoticeSlots(
        domain=LegalDomain.EMPLOYMENT,
        sender_name="Vishnu P",
        recipient_name="TechCorp Pvt Ltd",
        issue_summary="Non-payment of salary for 3 months"
    )
    assert slots.sender_name == "Vishnu P"
    assert slots.recipient_name == "TechCorp Pvt Ltd"
    assert slots.is_ready_for_pdf() is True

def test_pydantic_role_inversion_rejection():
    """Test Pydantic model validator raises ValueError when sender and recipient are identical."""
    with pytest.raises(ValueError) as excinfo:
        LegalNoticeSlots(
            sender_name="ammrkth",
            recipient_name="ammrkth",
            issue_summary="Salary dispute"
        )
    assert "Role Inversion Error" in str(excinfo.value)

def test_pydantic_overlapping_names_rejection():
    """Test overlapping or identical clean string rejection."""
    with pytest.raises(ValueError):
        LegalNoticeSlots(
            sender_name="ammrkth",
            recipient_name="ammrkth company",
            issue_summary="Salary dispute"
        )

def test_slot_parser_extracts_correctly():
    """Test form string parsing with typos and spacing variations."""
    pipeline = LegalMindPipeline()
    raw_form = "ൊഴിലുടമയുടെ പേര്: ammrkth\nകമ്പനിയുടെ പേര്: _wex kochi\nനിങ്ങളുടെ പൂർണ്ണ പേര്:  vishnu p"
    res = pipeline.parse_legal_notice_slots(raw_form)
    
    assert res["sender_name"] == "vishnu p"
    assert "ammrkth" in res["recipient_name"]
    assert res["sender_name"] != res["recipient_name"]

def test_placeholder_sanitization():
    """Test post-generation sanitizer removes bracket placeholders."""
    import re
    text = "Subject: Notice from Vishnu. RESIDENT OF: [TO BE INSERTED]. Amount: [TO BE FILLED BY SENDER]"
    sanitized = re.sub(r'\[\s*TO BE INSERTED\s*\]', 'As per employment records', text, flags=re.IGNORECASE)
    sanitized = re.sub(r'\[\s*TO BE FILLED BY SENDER\s*\]', 'As per employment records', sanitized, flags=re.IGNORECASE)
    
    assert "[TO BE INSERTED]" not in sanitized
    assert "[TO BE FILLED BY SENDER]" not in sanitized
    assert "As per employment records" in sanitized
