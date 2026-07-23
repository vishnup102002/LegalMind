"""
Pydantic Structural Validation Models for LegalMind Pipeline.
Enforces strict domain classification, role non-inversion, and PDF readiness gates.
"""
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, model_validator

class LegalDomain(str, Enum):
    EMPLOYMENT = "EMPLOYMENT"
    TENANCY = "TENANCY"
    CONSUMER = "CONSUMER"
    CONTRACT = "CONTRACT"
    CIVIL = "CIVIL"

class LegalNoticeSlots(BaseModel):
    domain: LegalDomain = Field(default=LegalDomain.EMPLOYMENT, description="Domain classification for legal dispute")
    sender_name: Optional[str] = Field(default=None, description="Full name of client/complainant sending the notice")
    recipient_name: Optional[str] = Field(default=None, description="Full name or company name of respondent receiving notice")
    company_name: Optional[str] = Field(default=None, description="Name of organization/company involved")
    city: Optional[str] = Field(default="Kochi", description="City/jurisdiction location")
    issue_summary: Optional[str] = Field(default=None, description="Brief summary of dispute facts")
    unpaid_months: Optional[int] = Field(default=3, description="Number of months of wage arrears")

    @model_validator(mode='after')
    def validate_role_non_inversion(self):
        """Validates that Complainant (Sender) and Respondent (Recipient) names are not identical or inverted."""
        if self.sender_name and self.recipient_name:
            s_clean = ''.join(c for c in self.sender_name if c.isalnum()).lower()
            r_clean = ''.join(c for c in self.recipient_name if c.isalnum()).lower()
            
            if s_clean and r_clean and (s_clean == r_clean or s_clean in r_clean or r_clean in s_clean):
                raise ValueError(
                    f"Role Inversion Error: Complainant ('{self.sender_name}') and "
                    f"Respondent ('{self.recipient_name}') cannot be identical or overlapping."
                )
        return self

    def is_ready_for_pdf(self) -> bool:
        """Structural check ensuring all mandatory slots exist before document compilation."""
        return bool(
            self.sender_name and len(self.sender_name.strip()) >= 2 and
            self.recipient_name and len(self.recipient_name.strip()) >= 2 and
            self.issue_summary and len(self.issue_summary.strip()) >= 5
        )
