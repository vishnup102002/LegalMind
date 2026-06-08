import json
import os
import sys
import logging

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.pipeline import LegalMindPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalMind.Evaluation")

def run_evaluation_suite():
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "eval", "synthetic_dataset.json"
    )
    
    if not os.path.exists(dataset_path):
        print("x Synthetic dataset not found. Generating sample dataset first...")
        from data.synthesis.generate_synth_data import SyntheticDatasetGenerator
        generator = SyntheticDatasetGenerator()
        generator.write_dataset()

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Test under different threshold cutoffs to find optimal balance
    test_thresholds = [0.65, 0.72, 0.80]
    print("\n=============================================")
    print("      LEGALMIND EVALUATION HARNESS SUITE     ")
    print("=============================================\n")

    for th in test_thresholds:
        print(f"--- Running Evaluation with Threshold: {th} ---")
        pipeline = LegalMindPipeline(threshold=th)
        
        passed_shield = 0
        blocked_shield = 0
        errors = 0
        
        for sample in samples:
            try:
                res = pipeline.run(sample["colloquial_panic"])
                if res["status"] == "SUCCESS":
                    passed_shield += 1
                elif res["status"] == "UNVERIFIED_LEGAL_GROUNDS":
                    blocked_shield += 1
            except Exception as e:
                logger.error(f"Error evaluating sample {sample['id']}: {e}")
                errors += 1
                
        total = len(samples)
        print(f"Total Evaluated Samples: {total}")
        print(f"Passed Shield Gate (Valid Grounds): {passed_shield} ({passed_shield/total*100:.1f}%)")
        print(f"Blocked by Shield Gate (Safety Filter): {blocked_shield} ({blocked_shield/total*100:.1f}%)")
        print(f"Execution Error Rate: {errors} ({errors/total*100:.1f}%)\n")

    print("✓ Threshold evaluation completed successfully.")

if __name__ == "__main__":
    run_evaluation_suite()
