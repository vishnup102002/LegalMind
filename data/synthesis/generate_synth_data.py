import json
import os
import logging

logger = logging.getLogger("LegalMind.Synthesis.Dataset")

class SyntheticDatasetGenerator:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "eval"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_eval_samples(self) -> list[dict]:
        """
        Creates synthetic colloquial Malayalam inputs representing eviction/lease disputes
        and maps them to ideal (chosen) IRAC notice structures and naive (rejected) comments.
        """
        samples = [
            {
                "id": 1,
                "dialect": "Malabar",
                "colloquial_panic": "എന്റെ വീട്ടുടമസ്ഥൻ എന്നോട് നാളെത്തന്നെ റൂമൊഴിഞ്ഞു തരാൻ പറഞ്ഞു. എനിക്ക് പോകാൻ വേറെ ഒരിടവുമില്ല. ഞാൻ എന്തുചെയ്യും?",
                "ground_truth_citation": "Section 14 of the State Rent Control Act",
                "chosen": "LEGAL ROADMAP (IRAC FORMAT):\n"
                          "ISSUE: Whether the landlord can demand immediate vacancy without a 30-day notice.\n"
                          "RULE: State Rent Control Act Section 14 mandates a 30-day written notice for eviction.\n"
                          "APPLICATION: The verbal eviction demand issued on [Date] violates Section 14 statutory guidelines.\n"
                          "CONCLUSION: You are legally entitled to remain in the property. Refuse eviction without written notice.",
                "rejected": "അയ്യോ വിഷമിക്കേണ്ട. നിങ്ങളുടെ ഓണറോട് പറയുക പെട്ടെന്ന് അങ്ങനെ ഇറക്കിവിടാൻ പറ്റില്ല എന്ന്. നിയമപ്രകാരം മുപ്പത് ദിവസത്തെ സമയം തരണമെന്ന് പറയാം. പേടിക്കേണ്ട ആവശ്യമില്ല."
            },
            {
                "id": 2,
                "dialect": "Travancore",
                "colloquial_panic": "അലവൻസ് തരാതെ കമ്പനിക്കാരൻ എന്നെ പിരിച്ചുവിട്ടു. പൈസ ചോദിച്ചപ്പോൾ ഭീഷണിപ്പെടുത്തുന്നു. കേസ് കൊടുക്കാൻ പറ്റുമോ?",
                "ground_truth_citation": "Section 25F of the Industrial Disputes Act",
                "chosen": "LEGAL ROADMAP (IRAC FORMAT):\n"
                          "ISSUE: Unlawful termination without payment of retrenchment compensation.\n"
                          "RULE: Section 25F of the Industrial Disputes Act requires one month's notice and compensation payment before retrenchment.\n"
                          "APPLICATION: Your retrenchment without compensation violates statutory procedures under Section 25F.\n"
                          "CONCLUSION: Draft and issue a formal demand notice for due wages and severance before approaching the Labor Court.",
                "rejected": "പൈസ തരാതെ പിരിച്ചുവിട്ടത് മോശമാണ്. നിങ്ങൾക്ക് ലേബർ ഓഫീസിൽ പരാതി കൊടുക്കാം. അവർ സഹായിക്കും."
            },
            {
                "id": 3,
                "dialect": "General Colloquial",
                "colloquial_panic": "വീട്ടുടമസ്ഥൻ കറന്റും വെള്ളവും മുറിച്ചു. വാടക കുടിശ്ശിക വന്നതിനാണ് ഇങ്ങനെ ചെയ്തത്. ഇപ്പൊ കറന്റില്ലാഞ്ഞിട്ട് പിള്ളേർക്ക് പഠിക്കാൻ പറ്റുന്നില്ല.",
                "ground_truth_citation": "Section 24 of the Rent Control Act",
                "chosen": "LEGAL ROADMAP (IRAC FORMAT):\n"
                          "ISSUE: Whether a landlord can disconnect essential services (electricity and water) due to rent default.\n"
                          "RULE: Section 24 of the Rent Control Act prohibits landlords from cutting off essential services without court authorization.\n"
                          "APPLICATION: The disconnection of electricity for rent default is an illegal act under Section 24.\n"
                          "CONCLUSION: File an emergency petition before the Rent Control Authority to restore services immediately and penalize the landlord.",
                "rejected": "വാടക കൊടുക്കാത്തതിന് കറന്റ് കട്ട് ചെയ്യുന്നത് തെറ്റാണ്. നിങ്ങൾ കെഎസ്ഇബിയിൽ വിളിച്ചു പറയാൻ നോക്കുക. അല്ലെങ്കിൽ വാടക പൈസ ഉടനെ കൊടുത്ത് തീർക്കുക."
            }
        ]
        return samples

    def write_dataset(self):
        samples = self.generate_eval_samples()
        output_path = os.path.join(self.output_dir, "synthetic_dataset.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Synthetic evaluation dataset generated at: {output_path}")
        print(f"✓ Synthetic evaluation dataset generated: {output_path}")

if __name__ == "__main__":
    generator = SyntheticDatasetGenerator()
    generator.write_dataset()
