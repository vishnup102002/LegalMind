import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import LegalMindPipeline
from app.evaluator import LegalEvaluator

TEST_SCENARIOS = [
    {
        "category": "Ragging",
        "language": "en",
        "query": "My MCA 2nd year seniors at TKMCE College Kollam slapped me and demanded 5000 rupees yesterday."
    },
    {
        "category": "Ragging",
        "language": "ml",
        "query": "കൊല്ലം ടി.കെ.എം.സി.ഇ കോളേജിലെ എന്റെ സീനിയർ വിദ്യാർത്ഥികൾ എന്നെ തല്ലുകയും പണം ആവശ്യപ്പെടുകയും ചെയ്തു."
    },
    {
        "category": "Eviction",
        "language": "en",
        "query": "My landlord told me to vacate my room in Ernakulam immediately tomorrow without any written notice."
    },
    {
        "category": "Eviction",
        "language": "ml",
        "query": "ഒരു നോട്ടീസും തരാതെ നാളെത്തന്നെ കൊച്ചിയിലെ മുറി ഒഴിഞ്ഞു പോകാൻ എന്റെ വീട്ടുടമസ്ഥൻ എന്നോട് ആവശ്യപ്പെടുന്നു."
    },
    {
        "category": "Wage Theft",
        "language": "en",
        "query": "My employer at the factory has not paid my monthly salary for the past three months."
    },
    {
        "category": "Wage Theft",
        "language": "ml",
        "query": "കഴിഞ്ഞ മൂന്ന് മാസമായി ജോലി ചെയ്ത ശമ്പളം ഫാക്ടറി ഉടമ എനിക്ക് നൽകിയിട്ടില്ല."
    },
    {
        "category": "Consumer Complaint",
        "language": "en",
        "query": "I bought a mobile phone from a shop in Kozhikode, it stopped working in 2 days and shopkeeper refuses repair."
    },
    {
        "category": "Consumer Complaint",
        "language": "ml",
        "query": "കോഴിക്കോട്ടെ കടയിൽ നിന്ന് വാങ്ങിയ ഫോൺ രണ്ട് ദിവസത്തിനകം കേടായി, കടക്കാരൻ അത് നന്നാക്കിത്തരാൻ വിസമ്മതിക്കുന്നു."
    },
    {
        "category": "Greeting",
        "language": "en",
        "query": "Hello, good morning! Who are you?"
    },
    {
        "category": "Greeting",
        "language": "ml",
        "query": "ഹലോ, സുപ്രഭാതം! എന്താണ് നിന്റെ പേര്?"
    },
    {
        "category": "Clarification",
        "language": "en",
        "query": "I need some legal help please."
    },
    {
        "category": "Clarification",
        "language": "ml",
        "query": "എനിക്ക് അടിയന്തിരമായി ഒരു നിയമസഹായം വേണം."
    }
]

def main():
    print("=" * 60)
    print("Starting LegalMind Automated Evaluation Harness")
    print("=" * 60)
    
    # 1. Initialize Pipeline & Evaluator
    print("Loading pipeline and database connections...")
    pipeline = LegalMindPipeline()
    evaluator = LegalEvaluator(pipeline._call_ollama_api)
    print("Pipeline and Evaluator loaded successfully.\n")

    results = []
    
    # 2. Run Scenarios
    for idx, scenario in enumerate(TEST_SCENARIOS, 1):
        category = scenario["category"]
        lang = scenario["language"]
        query = scenario["query"]
        
        print(f"[{idx}/{len(TEST_SCENARIOS)}] Evaluating {category} in {lang.upper()}...")
        print(f"Query: '{query}'")
        
        t0 = time.time()
        pipeline_res = pipeline.run(query, history=[])
        latency = time.time() - t0
        
        response_text = pipeline_res.get("response_text", "")
        context = pipeline_res.get("context", "")
        status = pipeline_res.get("status", "SUCCESS")
        
        print(f"Pipeline executed in {latency:.2f} seconds. Status: {status}")
        
        # Call evaluator
        print("Running LLM-as-Judge evaluation...")
        eval_scores = evaluator.evaluate(query, context, response_text, lang)
        
        # Save results
        results.append({
            "index": idx,
            "category": category,
            "language": lang,
            "query": query,
            "response": response_text,
            "status": status,
            "latency": latency,
            "scores": eval_scores
        })
        print(f"Scores -> Faithfulness: {eval_scores.get('faithfulness')}, Relevance: {eval_scores.get('relevance')}, "
              f"Language Compliance: {eval_scores.get('language_compliance')}, Hallucination: {eval_scores.get('hallucination')}, "
              f"Completeness: {eval_scores.get('completeness')}\n")

    # 3. Generate Report
    print("Generating Markdown report...")
    report_lines = [
        "# LegalMind Automated Quality Evaluation Report",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Metrics",
        ""
    ]
    
    total = len(results)
    avg_faithfulness = sum(r["scores"].get("faithfulness", 0) for r in results) / total
    avg_relevance = sum(r["scores"].get("relevance", 0) for r in results) / total
    avg_lang_compliance = sum(r["scores"].get("language_compliance", 0) for r in results) / total
    avg_hallucination = sum(r["scores"].get("hallucination", 0) for r in results) / total
    avg_completeness = sum(r["scores"].get("completeness", 0) for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total
    
    report_lines.extend([
        "| Metric | Average Score | Target / Threshold | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Faithfulness** | {avg_faithfulness:.2f} | >= 0.70 | {'✅ PASS' if avg_faithfulness >= 0.70 else '❌ FAIL'} |",
        f"| **Relevance** | {avg_relevance:.2f} | >= 0.80 | {'✅ PASS' if avg_relevance >= 0.80 else '❌ FAIL'} |",
        f"| **Language Compliance** | {avg_lang_compliance:.2f} | >= 0.90 | {'✅ PASS' if avg_lang_compliance >= 0.90 else '❌ FAIL'} |",
        f"| **Hallucination Rate** | {avg_hallucination:.2f} | <= 0.10 | {'✅ PASS' if avg_hallucination <= 0.10 else '❌ FAIL'} |",
        f"| **Completeness** | {avg_completeness:.2f} | >= 0.80 | {'✅ PASS' if avg_completeness >= 0.80 else '❌ FAIL'} |",
        "",
        f"- **Average Latency:** {avg_latency:.2f} seconds",
        "",
        "## Detailed Results Table",
        "",
        "| ID | Category | Lang | Status | Faithfulness | Relevance | Lang Compliance | Hallucination | Completeness | Latency |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])
    
    for r in results:
        scores = r["scores"]
        report_lines.append(
            f"| {r['index']} | {r['category']} | {r['language'].upper()} | {r['status']} | "
            f"{scores.get('faithfulness'):.1f} | {scores.get('relevance'):.1f} | "
            f"{scores.get('language_compliance'):.1f} | {scores.get('hallucination'):.1f} | "
            f"{scores.get('completeness'):.1f} | {r['latency']:.1f}s |"
        )
        
    report_lines.extend([
        "",
        "## Detailed Scenario Logs",
        ""
    ])
    
    for r in results:
        scores = r["scores"]
        report_lines.extend([
            f"### Scenario {r['index']}: {r['category']} ({r['language'].upper()})",
            f"- **User Query:** `{r['query']}`",
            f"- **Status:** `{r['status']}`",
            f"- **Latency:** `{r['latency']:.2f}s`",
            f"- **Evaluation Scores:**",
            f"  - Faithfulness: **{scores.get('faithfulness')}**",
            f"  - Relevance: **{scores.get('relevance')}**",
            f"  - Language Compliance: **{scores.get('language_compliance')}**",
            f"  - Hallucination: **{scores.get('hallucination')}**",
            f"  - Completeness: **{scores.get('completeness')}**",
            f"- **Reasoning:** {scores.get('reasoning')}",
            f"- **Pipeline Response:**",
            "```",
            r["response"],
            "```",
            "---",
            ""
        ])
        
    report_path = "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nEvaluation finished! Report generated at: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
