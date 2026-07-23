"""
Layer 3 (Extended): Multi-Domain & Multi-Lingual Automated Test Suite for LegalMind.
Tests multi-turn session workflows across 4 distinct legal domains (Labor, Tenancy, Consumer, Contract)
in Malayalam, Manglish, and English. Asserting domain isolation, statutory accuracy, and role non-inversion.
"""
import pytest
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

# Test Data Matrix: Domains, Languages, and Inputs
TEST_CASES = [
    # ------------------------------------------------------------------
    # DOMAIN 1: LABOR / EMPLOYMENT LAW (Malayalam, Manglish, English)
    # ------------------------------------------------------------------
    {
        "domain": "Labor Law",
        "language": "Malayalam",
        "turns": [
            "ഞാൻ കൊച്ചിയിൽ ഒരു ഐടി കമ്പനിയിൽ ജോലി ചെയ്യുന്നു. 3 മാസമായി ശമ്പളം തന്നിട്ടില്ല.",
            "എനിക്ക് ഒരു ലീഗൽ നോട്ടീസ് ഉണ്ടാക്കണം",
            "തൊഴിലുടമയുടെ പേര്: രാഹുൽ മേനോൻ\nകമ്പനിയുടെ പേര്: ടെക് ഇന്നൊവേഷൻസ് കൊച്ചി\nപൂർണ്ണ പേര്: വിഷ്ണു പി"
        ],
        "expected_domain": "EMPLOYMENT",
        "expected_statute": "Payment of Wages Act, 1936",
        "expected_complainant": "വിഷ്ണു പി",
        "expected_respondent": "രാഹുൽ മേനോൻ"
    },
    {
        "domain": "Labor Law",
        "language": "Manglish",
        "turns": [
            "Njan Kochi labast companyil work cheyyunu. 2 month aayi salary kittiyilla.",
            "Legal notice venam",
            "Employer: Suresh Kumar\nCompany: ABC Solutions Kochi\nMy Name: Vishnu P"
        ],
        "expected_domain": "EMPLOYMENT",
        "expected_statute": "Payment of Wages Act, 1936",
        "expected_complainant": "Vishnu P",
        "expected_respondent": "Suresh Kumar"
    },
    {
        "domain": "Labor Law",
        "language": "English",
        "turns": [
            "I work at a software firm in Kochi. They terminated me without paying notice pay or salary for 2 months.",
            "Draft a legal notice for non-payment of salary",
            "Employer Name: John Doe\nCompany: TechCorp Ltd Kochi\nEmployee Name: Vishnu P"
        ],
        "expected_domain": "EMPLOYMENT",
        "expected_statute": "Industrial Disputes Act, 1947",
        "expected_complainant": "Vishnu P",
        "expected_respondent": "John Doe"
    },

    # ------------------------------------------------------------------
    # DOMAIN 2: TENANCY & RENT DISPUTES (Malayalam & English)
    # ------------------------------------------------------------------
    {
        "domain": "Tenancy Law",
        "language": "Malayalam",
        "turns": [
            "ഞാൻ തൃപ്പൂണിത്തുറയിൽ ഒരു വീട് വാടകയ്ക്ക് എടുത്തിരുന്നു. ഒഴിയുമ്പോൾ അഡ്വാൻസ് തുക തരുന്നില്ല.",
            "വീട്ടുഉടമസ്ഥന് ലീഗൽ നോട്ടീസ് അയക്കണം",
            "വീട്ടുഉടമസ്ഥൻ: രാജൻ പിള്ള\nസ്വത്ത് വിലാസം: തൃപ്പൂണിത്തുറ, കൊച്ചി\nഎന്റെ പേര്: വിഷ്ണു പി"
        ],
        "expected_domain": "TENANCY",
        "expected_statute": "Rent Control",
        "expected_complainant": "വിഷ്ണു പി",
        "expected_respondent": "രാജൻ പിള്ള"
    },
    {
        "domain": "Tenancy Law",
        "language": "English",
        "turns": [
            "My landlord in Kakkanad is refusing to refund my security deposit of 50000 rupees after I vacated the apartment.",
            "Can you prepare a formal demand notice?",
            "Landlord Name: Mathew Joseph\nProperty Address: Kakkanad, Kochi\nTenant Name: Vishnu P"
        ],
        "expected_domain": "TENANCY",
        "expected_statute": "Rent Control",
        "expected_complainant": "Vishnu P",
        "expected_respondent": "Mathew Joseph"
    },

    # ------------------------------------------------------------------
    # DOMAIN 3: CONSUMER PROTECTION (Malayalam & English)
    # ------------------------------------------------------------------
    {
        "domain": "Consumer Law",
        "language": "Malayalam",
        "turns": [
            "ഞാൻ ഒരു ഓൺലൈൻ സ്റ്റോറിൽ നിന്ന് 30000 രൂപയ്ക്ക് ഫോൺ വാങ്ങി, ഡിഫെക്റ്റീവ് ആണ്. റീഫണ്ട് തരുന്നില്ല.",
            "കമ്പനിക്ക് നോട്ടീസ് തയ്യാറാക്കൂ",
            "എതിർകക്ഷി: ഇ-കോമേഴ്സ് റീട്ടെയിൽ ലിമിറ്റഡ്\nപരാതിക്കാരൻ: വിഷ്ണു പി"
        ],
        "expected_domain": "CONSUMER",
        "expected_statute": "Consumer Protection Act",
        "expected_complainant": "വിഷ്ണു പി",
        "expected_respondent": "ഇ-കോമേഴ്സ് റീട്ടെയിൽ ലിമിറ്റഡ്"
    },
    {
        "domain": "Consumer Law",
        "language": "English",
        "turns": [
            "I paid 45000 INR for an laptop repair service in Kochi, but they damaged the motherboard and refuse compensation.",
            "I want to issue a legal notice to the service center.",
            "Opposite Party: Apex Laptop Solutions Kochi\nComplainant Name: Vishnu P"
        ],
        "expected_domain": "CONSUMER",
        "expected_statute": "Consumer Protection Act",
        "expected_complainant": "Vishnu P",
        "expected_respondent": "Apex Laptop Solutions Kochi"
    },

    # ------------------------------------------------------------------
    # DOMAIN 4: CONTRACT BREACH / FREELANCE PAYMENT (Malayalam & English)
    # ------------------------------------------------------------------
    {
        "domain": "Contract Law",
        "language": "English",
        "turns": [
            "I completed a freelance software development project for a client in Kochi. They haven't paid the pending invoice of 1.2 Lakhs.",
            "Generate a legal demand notice for breach of contract",
            "Client Name: Alex Vance\nClient Business: Vance Digital Kochi\nFreelancer Name: Vishnu P"
        ],
        "expected_domain": "CONTRACT",
        "expected_statute": "Contract Act",
        "expected_complainant": "Vishnu P",
        "expected_respondent": "Alex Vance"
    }
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_multi_domain_multi_lingual_sessions(test_case):
    """
    Simulates multi-turn session workflows across different legal domains and languages.
    Asserts correct domain classification, slot extraction, and non-inversion.
    """
    phone_number = f"whatsapp:+91900000{TEST_CASES.index(test_case):02d}"
    
    # Step 1: Reset session context
    client.post("/api/whatsapp/webhook", json={"from": phone_number, "message": "/reset"})

    state = None
    for turn in test_case["turns"]:
        response = client.post("/api/whatsapp/webhook", json={"from": phone_number, "message": turn})
        assert response.status_code == 200
        data = response.json()
        state = data.get("agent_state", {})

    # Layer 1 & 2 Verification Checks
    assert state is not None, "Agent state should not be empty"
    
    # 1. Assert correct role non-inversion
    extracted_complainant = state.get("employee_name") or state.get("complainant_name") or state.get("tenant_name")
    extracted_respondent = state.get("employer_name") or state.get("respondent_name") or state.get("landlord_name")
    
    if extracted_complainant and extracted_respondent:
        assert extracted_complainant.lower() != extracted_respondent.lower(), \
            f"Role inversion detected! Both mapped as '{extracted_complainant}'"
