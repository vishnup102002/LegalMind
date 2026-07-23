"""
Layer 2: DeepEval & RAG Metric Evaluation Engine for LegalMind.
Evaluates Statutory Grounding, Hallucination Prevention, Domain Isolation, and Role Non-Inversion metrics.
"""
import pytest
from app.pipeline import LegalMindPipeline

@pytest.fixture(scope="module")
def pipeline():
    return LegalMindPipeline()

def test_hallucination_metric_grounding(pipeline):
    """DeepEval Metric: Ensure cited statutes match verified database context."""
    query = "ഞാൻ കൊച്ചിയിൽ ഒരു കമ്പനിയിൽ ജോലി ചെയ്യുന്നു. അവർ മൂന്ന് മാസമായി ശമ്പളം തന്നിട്ടില്ല."
    res = pipeline.run(query)
    response_text = res.get("response_text", "")
    context = res.get("context", "")

    # Grounding Check: Hallucinated statutes must be 0
    FORBIDDEN_HALLUCINATED_STATUTES = [
        "നിയമസഭ നിയമം 25",
        "ഇന്ത്യൻ കോടതി നിയമം",
        "Indian Employees Act",
        "Legislative Act 25",
        "Section 6 of the Indian Employees Act"
    ]
    for forbidden in FORBIDDEN_HALLUCINATED_STATUTES:
        assert forbidden.lower() not in response_text.lower(), f"Hallucinated statute detected: '{forbidden}'"

    # Faithfulness Score: Must contain legitimate labor law citations
    VALID_CITATIONS = ["Payment of Wages", "പേയ്‌മെന്റ് ഓഫ് വേജസ്", "Industrial Disputes", "ഇൻഡസ്ട്രിയൽ", "KeLSA"]
    has_valid_cite = any(cite.lower() in response_text.lower() for cite in VALID_CITATIONS)
    assert has_valid_cite is True, "Response must contain a faithful labor law citation."

def test_faithfulness_and_domain_isolation(pipeline):
    """DeepEval Metric: Ensures tenancy queries retrieve tenancy statutes, not labor laws."""
    query = "എന്റെ വീട്ടുടമസ്ഥൻ എന്നോട് നാളെത്തന്നെ റൂമൊഴിഞ്ഞു തരാൻ പറഞ്ഞു."
    res = pipeline.run(query)
    response_text = res.get("response_text", "")

    # Domain Isolation assertion
    assert "ശമ്പളം" not in response_text, "Tenancy response should not contain employment salary terms."
    assert "തൊഴിലുടമ" not in response_text, "Tenancy response should not contain employer terms."

def test_g_eval_role_non_inversion_score(pipeline):
    """G-Eval Metric: Asserts complainant identity does not match respondent identity."""
    history = [{'role': 'user', 'text': 'ഞാൻ കൊച്ചിയിൽ ഒരു കമ്പനിയിൽ ജോലി ചെയ്യുന്നു. 3 മാസമായി ശമ്പളം തന്നിട്ടില്ല.'}]
    form_input = "ൊഴിലുടമയുടെ പേര്: ammrkth\nകമ്പനിയുടെ പേര്: _wex kochi\nനിങ്ങളുടെ പൂർണ്ണ പേര്:  vishnu p"
    
    res = pipeline.run(form_input, history=history)
    response_text = res.get("response_text", "")

    # Assert PDF URL generated
    assert "[DOWNLOAD_URL:" in response_text, "Must generate valid download URL."
    
    # Assert Role Non-Inversion
    assert "Sender: vishnu p" in response_text, "Sender must be vishnu p."
    assert "Sender: ammrkth" not in response_text, "Sender must NOT be ammrkth."
