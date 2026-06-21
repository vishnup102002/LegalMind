import json
import logging

logger = logging.getLogger("LegalMind.Evaluator")

class LegalEvaluator:
    def __init__(self, llm_caller):
        """
        Initialize the evaluator.
        :param llm_caller: A callable function/method that takes a prompt (str) and temperature (float) 
                           and returns the LLM response string.
        """
        self.llm_caller = llm_caller

    def evaluate(self, query: str, context: str, response: str, input_lang: str) -> dict:
        """
        Evaluate a single pipeline run.
        """
        prompt = f"""You are an expert legal evaluator (LLM-as-a-Judge) for an Indian legal aid chatbot system.
Analyze the following inputs:
1. User Query: {query}
2. Retrieved Legal Context: {context}
3. Pipeline Response: {response}
4. Input Language: {input_lang}

Evaluate the Pipeline Response based on the following metrics, scoring each from 0.0 to 1.0:

1. "faithfulness": Does the response cite and rely strictly on real statutes/provisions present in the Retrieved Legal Context? (0.0 if it refers to laws not in the context, 1.0 if it is fully faithful)
2. "relevance": Is the response directly relevant and responsive to the user's query? (1.0 if highly relevant, 0.0 if completely off-topic)
3. "language_compliance": Does the response strictly use the language of the user's input? (If input_lang is "ml", the entire response should be in Malayalam. If input_lang is "en", the entire response should be in English. 1.0 if compliant, 0.0 if not compliant)
4. "hallucination": Does the response fabricate legal sections, acts, or factual rules that are NOT present in the retrieved context? (0.0 if there are no hallucinations/inventions, 1.0 if there is high hallucination)
5. "completeness": For IRAC roadmaps, does the response contain all required parts: ISSUE, RULE, APPLICATION, CONCLUSION, and LAYPERSON advice? (1.0 if complete, 0.0 if parts are missing. Note: for simple greetings or clarification questions, completeness should be 1.0 if the greeting/clarification is generated successfully.)

Return ONLY a JSON object matching this schema:
{{
  "faithfulness": 1.0,
  "relevance": 1.0,
  "language_compliance": 1.0,
  "hallucination": 0.0,
  "completeness": 1.0,
  "reasoning": "brief explanation for the scores"
}}
Do not include any notes, markdown codeblock formatting, or preamble. Return only the raw JSON.
"""
        try:
            # We call the LLM caller with format_json hint or let it handle format JSON
            # since the prompt requests ONLY JSON, we pass format_json=True if the function supports it,
            # or try/except block. Since _call_ollama_api supports format_json, we can pass it if we inspect
            # the callable or just use format_json=True
            res_str = self.llm_caller(prompt, temperature=0.0, format_json=True)
            # Remove potential markdown wraps if the model still outputs them
            res_str = res_str.strip()
            if res_str.startswith("```json"):
                res_str = res_str[7:]
            if res_str.endswith("```"):
                res_str = res_str[:-3]
            res_str = res_str.strip()
            
            return json.loads(res_str)
        except Exception as e:
            logger.error(f"Evaluation LLM call failed: {e}")
            return {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "language_compliance": 0.0,
                "hallucination": 1.0,
                "completeness": 0.0,
                "reasoning": f"Evaluation error: {str(e)}"
            }
