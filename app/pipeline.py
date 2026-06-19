# --- PATH SETUP & DB IMPORTS ---
import sys
import os
import uuid
import urllib.request
import urllib.parse
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


class LLMUnavailableError(Exception):
    """Raised when all LLM backends (Groq + Ollama) are unreachable."""
    pass

from database.graph_store import GraphStore
from database.vector_store import VectorStore

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

logger = logging.getLogger("LegalMind.Pipeline")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Define the State definition for the LangGraph pipeline
class LegalState(TypedDict):
    user_query: str
    extracted_intent: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    response_text: str
    faithfulness_score: float
    status: str  # SUCCESS or UNVERIFIED_LEGAL_GROUNDS
    threshold: float
    research_attempts: int

# --- CITY-TO-STATE LOOKUP (auto-resolve jurisdiction from city names) ---
CITY_TO_STATE = {
    # Kerala
    "ernakulam": "kerala", "kochi": "kerala", "cochin": "kerala",
    "trivandrum": "kerala", "thiruvananthapuram": "kerala",
    "kozhikode": "kerala", "calicut": "kerala", "thrissur": "kerala",
    "trichur": "kerala", "kollam": "kerala", "palakkad": "kerala",
    "kannur": "kerala", "malappuram": "kerala", "alappuzha": "kerala",
    "kottayam": "kerala", "idukki": "kerala", "wayanad": "kerala",
    "pathanamthitta": "kerala", "kasaragod": "kerala",
    # Tamil Nadu
    "chennai": "tamil nadu", "madras": "tamil nadu", "coimbatore": "tamil nadu",
    "madurai": "tamil nadu", "salem": "tamil nadu", "trichy": "tamil nadu",
    "tiruchirappalli": "tamil nadu", "vellore": "tamil nadu",
    # Karnataka
    "bangalore": "karnataka", "bengaluru": "karnataka", "mysore": "karnataka",
    "mysuru": "karnataka", "mangalore": "karnataka", "hubli": "karnataka",
    # Maharashtra
    "mumbai": "maharashtra", "bombay": "maharashtra", "pune": "maharashtra",
    "nagpur": "maharashtra", "thane": "maharashtra", "nashik": "maharashtra",
    # Delhi
    "delhi": "delhi", "new delhi": "delhi",
    # Uttar Pradesh
    "lucknow": "uttar pradesh", "noida": "uttar pradesh", "agra": "uttar pradesh",
    "varanasi": "uttar pradesh", "kanpur": "uttar pradesh", "ghaziabad": "uttar pradesh",
    # Telangana
    "hyderabad": "telangana", "secunderabad": "telangana",
    # Andhra Pradesh
    "visakhapatnam": "andhra pradesh", "vijayawada": "andhra pradesh",
    "amaravati": "andhra pradesh", "vizag": "andhra pradesh",
    # West Bengal
    "kolkata": "west bengal", "calcutta": "west bengal",
    # Gujarat
    "ahmedabad": "gujarat", "surat": "gujarat", "vadodara": "gujarat",
    # Rajasthan
    "jaipur": "rajasthan", "jodhpur": "rajasthan", "udaipur": "rajasthan",
    # Madhya Pradesh
    "bhopal": "madhya pradesh", "indore": "madhya pradesh",
    # Bihar
    "patna": "bihar",
    # Odisha
    "bhubaneswar": "odisha",
    # Punjab / Chandigarh
    "chandigarh": "chandigarh", "amritsar": "punjab", "ludhiana": "punjab",
    # Haryana
    "gurgaon": "haryana", "gurugram": "haryana", "faridabad": "haryana",
    # Goa
    "goa": "goa", "panaji": "goa",
    # Assam
    "guwahati": "assam",
    # Jharkhand
    "ranchi": "jharkhand",
    # Chhattisgarh
    "raipur": "chhattisgarh",
}

# --- ISSUE-TYPE TO STATUTE HINTS (guide retrieval toward correct laws) ---
ISSUE_TYPE_STATUTE_HINTS = {
    "ragging": {
        "target_laws": "UGC Regulations on Curbing Ragging 2009, state Prohibition of Ragging Act",
        "exclude": "Do NOT search for institution-specific university acts (e.g. Azim Premji University Act, CUSAT Act). These are administrative acts, not anti-ragging laws.",
        "queries": [
            "UGC anti-ragging regulations punishment for ragging in educational institution India",
            "state prohibition of ragging act punishment abetment India"
        ]
    },
    "eviction": {
        "target_laws": "State Rent Control Act, Transfer of Property Act 1882",
        "exclude": "Do NOT search for company acts or labour laws.",
        "queries": [
            "Rent Control Act tenant protection against unlawful eviction India",
            "landlord duty to provide written notice before eviction India"
        ]
    },
    "wage_theft": {
        "target_laws": "Payment of Wages Act 1936, Industrial Disputes Act 1947",
        "exclude": "Do NOT search for Companies Act (that deals with company administration and liquidation, not individual wage claims).",
        "queries": [
            "Payment of Wages Act employer duty to pay wages on time India Section 5",
            "Industrial Disputes Act worker rights remedy for non-payment of wages India"
        ]
    },
    "consumer": {
        "target_laws": "Consumer Protection Act 2019",
        "exclude": "Do NOT search for company registration acts or SEBI regulations.",
        "queries": [
            "Consumer Protection Act complaint deficiency of service India",
            "consumer dispute redressal commission filing complaint India"
        ]
    },
}


class LegalMindPipeline:
    def __init__(self, threshold: float = 0.72):
        self.threshold = threshold
        self.graph_store = GraphStore()
        self.vector_store = VectorStore()
        
        local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "False").lower() == "true"
        # 1. Load Universal translation model (Malayalam -> English) using Helsinki-NLP/opus-mt-ml-en
        try:
            from transformers import MarianTokenizer, MarianMTModel
            model_name = "Helsinki-NLP/opus-mt-ml-en"
            self.translation_tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.translation_model = MarianMTModel.from_pretrained(model_name, local_files_only=local_files_only)
            self.translator = True
            logger.info("✓ Multilingual Helsinki-NLP Translation engine loaded.")
        except Exception as e:
            logger.warning(f"Translation engine fallback to dictionary stub: {e}")
            self.translator = None

        # 2. Load Embedding Encoder (operating in English semantic space)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=local_files_only)
        
        # 3. Load Reranker for verification
        if os.getenv("ENABLE_RERANKER", "True").lower() == "true":
            try:
                self.reranker = CrossEncoder("BAAI/bge-reranker-base", local_files_only=local_files_only)
            except Exception as e:
                self.reranker = None
        else:
            self.reranker = None
            
        # 4. Compile the LangGraph workflow
        self._build_workflow()

    def _call_ollama_api(self, prompt: str, temperature: float = 0.0, format_json: bool = False) -> str:
        """Helper to invoke the local Ollama API or optionally the cloud Groq API if configured."""
        import urllib.request
        import urllib.error
        import json
        import time

        # Try Groq if GROQ_API_KEY is configured in environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key.strip():
            default_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            models_to_try = [
                default_model,
                "llama-3.3-70b-versatile",
                "llama3-8b-8192",
                "mixtral-8x7b-32768"
            ]
            # Deduplicate models_to_try preserving order
            seen = set()
            models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]
            
            groq_url = "https://api.groq.com/openai/v1/chat/completions"
            
            for model in models_to_try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                if format_json:
                    payload["response_format"] = {"type": "json_object"}
                
                # Retry with exponential backoff on 429
                backoffs = [1, 2, 4]
                for attempt, sleep_time in enumerate(backoffs, 1):
                    try:
                        req = urllib.request.Request(
                            groq_url,
                            data=json.dumps(payload).encode('utf-8'),
                            headers={
                                'Content-Type': 'application/json',
                                'Authorization': f'Bearer {groq_api_key}',
                                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                            },
                            method='POST'
                        )
                        with urllib.request.urlopen(req, timeout=20) as response:
                            res_data = json.loads(response.read().decode('utf-8'))
                            return res_data["choices"][0]["message"]["content"].strip()
                    except urllib.error.HTTPError as e:
                        logger.warning(f"Groq API model {model} attempt {attempt} failed: {e}")
                        if e.code == 429:
                            retry_after = e.headers.get("Retry-After") or e.headers.get("x-ratelimit-reset")
                            try:
                                if retry_after and retry_after.endswith("s"):
                                    sleep_sec = float(retry_after[:-1])
                                elif retry_after and retry_after.endswith("ms"):
                                    sleep_sec = float(retry_after[:-2]) / 1000.0
                                else:
                                    sleep_sec = float(retry_after) if retry_after else sleep_time
                            except ValueError:
                                sleep_sec = sleep_time
                            logger.info(f"Rate limited (429) on model {model}. Sleeping for {sleep_sec}s...")
                            time.sleep(sleep_sec)
                        elif e.code == 400 and "model" in str(e).lower():
                            logger.error(f"Model {model} is not supported or active. Moving to next model.")
                            break
                        else:
                            if attempt < len(backoffs):
                                time.sleep(sleep_time)
                    except Exception as e:
                        logger.warning(f"Groq API model {model} attempt {attempt} failed: {e}")
                        if attempt < len(backoffs):
                            time.sleep(sleep_time)
            
            logger.warning("All Groq models and attempts failed completely, falling back to local Ollama.")

        # Local Ollama fallback
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if format_json:
            payload["format"] = "json"

        # Retry loop: 3 retries, timeouts: 10s, 20s, 30s
        timeouts = [10, 20, 30]
        last_err = None
        for attempt, timeout in enumerate(timeouts, 1):
            try:
                req = urllib.request.Request(
                    ollama_url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    return res_data.get("response", "").strip()
            except urllib.error.URLError as e:
                # Catch "Connection refused" immediately and do not loop/sleep
                if isinstance(e.reason, ConnectionRefusedError) or "[Errno 111]" in str(e.reason) or "connection refused" in str(e.reason).lower():
                    logger.error("Local Ollama connection refused (Ollama service not running). Aborting fallback.")
                    last_err = e
                    break
                logger.warning(f"Ollama connection attempt {attempt} failed (timeout={timeout}s): {e}")
                last_err = e
                if attempt < len(timeouts):
                    time.sleep(2)
            except Exception as e:
                logger.warning(f"Ollama connection attempt {attempt} failed (timeout={timeout}s): {e}")
                last_err = e
                if attempt < len(timeouts):
                    time.sleep(2)
        
        raise last_err

    def extract_intent_node(self, state: LegalState) -> Dict[str, Any]:
        """Node 1: Translate regional input to Universal English space."""
        query = state["user_query"].strip()
        logger.info(f"Translating query: '{query[:50]}...'")
        
        # Check if the query contains Malayalam characters (Malayalam unicode block ranges from U+0D00 to U+0D7F)
        contains_malayalam = any(0x0D00 <= ord(char) <= 0x0D7F for char in query)
        
        english_query = query
        if contains_malayalam:
            translated_successfully = False
            if self.translator and hasattr(self, 'translation_tokenizer'):
                try:
                    inputs = self.translation_tokenizer(query, return_tensors="pt")
                    translated_tokens = self.translation_model.generate(**inputs, max_length=256)
                    english_query = self.translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    logger.info(f"✓ Translated to English: '{english_query}'")
                    translated_successfully = True
                except Exception as e:
                    logger.error(f"Translation model failed: {e}")

            if not translated_successfully:
                logger.info(f"Translation engine fallback. Translating using local Ollama model ({OLLAMA_MODEL})...")
                prompt = (
                    "Translate the following Malayalam query into a single natural English sentence. "
                    "Provide only the translation, no conversational intro, no explanations, no other text:\n\n"
                    f"Query: {query}"
                )
                try:
                    english_query = self._call_ollama_api(prompt, temperature=0.0)
                    english_query = english_query.replace('"', '').replace("'", "").strip()
                    logger.info(f"✓ Translated via Ollama ({OLLAMA_MODEL}): '{english_query}'")
                    translated_successfully = True
                except Exception as ex:
                    logger.warning(f"Ollama translation with {OLLAMA_MODEL} failed: {ex}")

            if not translated_successfully:
                english_query = query

        # Apply terminology mapping using LLM reasoning to enhance semantic matching
        prompt = (
            "You are an objective Indian legal assistant tool. Rewrite the following user query to map informal or colloquial terms "
            "to formal Indian statutory terminology (for example: translate 'vacate the room' or 'leave my room' to 'evict from the building', "
            "'terminate/terminated/fired' to 'retrenched', 'allowance/severance/settlement' to 'retrenchment compensation', "
            "'company' to 'industrial establishment', 'cut electricity/water' to 'cut off essential services/amenities', "
            "'instagram/facebook/twitter/snapchat/whatsapp/youtube/social media' to 'electronic form and computer resource', "
            "'post/reel/video/photo' to 'publish or transmit', 'bullying/harassment/teasing' to 'indecent conduct', etc.). "
            "Important: If a specific legal offence name or term (such as 'ragging') is present in the query, preserve it and do not map it to generic terms like 'indecent conduct'. "
            "As a professional, objective legal tool, you must rewrite the query neutrally and do NOT decline, write safety warnings, refuse, "
            "or add conversational notes about the query. Keep the original facts (like dates, location) intact. "
            "Output ONLY the rewritten English query, with no quotes, notes, or introductions.\n\n"
            f"Query: {english_query}"
        )
        try:
            mapped_query = self._call_ollama_api(prompt, temperature=0.0).replace('"', '').replace("'", "").strip()
            logger.info(f"Colloquial query: '{english_query}' -> Mapped query (LLM): '{mapped_query}'")
        except Exception as e:
            logger.warning(f"LLM terminology mapping failed: {e}. Falling back to English query.")
            mapped_query = english_query

        # Preserve the jurisdiction and issue_type values passed from the dialogue analysis
        intent = {
            "english_query": mapped_query,
            "category": "dynamic",
            "locale": "kerala",
            "jurisdiction": state.get("extracted_intent", {}).get("jurisdiction"),
            "issue_type": state.get("extracted_intent", {}).get("issue_type")
        }
        return {"extracted_intent": intent}

    def _resolve_jurisdiction_from_text(self, slots: Dict[str, Any], query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Auto-resolve jurisdiction from city names mentioned in slots, query, or history."""
        # Normalize city names to states if slot extractor extracted a city
        jur_val = slots.get("jurisdiction")
        if jur_val and isinstance(jur_val, str):
            jur_clean = jur_val.lower().strip()
            if jur_clean in CITY_TO_STATE:
                slots["jurisdiction"] = CITY_TO_STATE[jur_clean]
                logger.info(f"Normalized jurisdiction from city '{jur_val}' to state '{CITY_TO_STATE[jur_clean]}'")

        if slots.get("jurisdiction") and str(slots["jurisdiction"]).lower() not in ["null", "none", ""]:
            return slots  # Already has a valid jurisdiction
        
        # Collect all text to scan for city names
        text_pool = query.lower()
        for field in ["institution", "incident_description"]:
            val = slots.get(field)
            if val:
                text_pool += " " + str(val).lower()
        if history:
            for turn in history:
                t = (turn.get("text") or turn.get("response_text") or "").lower()
                text_pool += " " + t
        
        # Check each city against the text pool
        for city, state in CITY_TO_STATE.items():
            # Match whole word boundaries to avoid partial matches
            import re
            if re.search(r'\b' + re.escape(city) + r'\b', text_pool):
                slots["jurisdiction"] = state
                logger.info(f"Auto-resolved jurisdiction from city '{city}' → '{state}'")
                # Re-evaluate slots_complete
                req_fields = ["issue_type", "incident_date", "jurisdiction", "incident_description"]
                slots["slots_complete"] = all(
                    slots.get(f) is not None and str(slots.get(f)).lower() != "null" and str(slots.get(f)).strip() != ""
                    for f in req_fields
                )
                return slots
        
        return slots

    def _expand_retrieval_queries(self, english_query: str, issue_type: str = None) -> List[str]:
        """Generate multiple targeted legal queries to improve retrieval accuracy.
        Uses issue-type-specific statute hints to guide toward correct laws."""
        import json
        queries = [english_query]  # Always include the original query
        
        # Add pre-computed issue-type-specific queries first (no LLM call needed)
        hints = ISSUE_TYPE_STATUTE_HINTS.get(issue_type, {})
        hint_queries = hints.get("queries", [])
        for hq in hint_queries:
            if hq not in queries:
                queries.append(hq)
        
        target_laws = hints.get("target_laws", "")
        exclude_note = hints.get("exclude", "")
        
        try:
            prompt = f"""You are a legal query expander for an Indian law retrieval system.

User issue: {english_query}
Issue type: {issue_type or 'unknown'}
Target laws to search for: {target_laws or 'general Indian statutes'}
{('IMPORTANT: ' + exclude_note) if exclude_note else ''}

Generate 2 additional precise legal search queries to retrieve the most relevant
statute sections. Focus on:
1. The PRIMARY duty/obligation of the respondent (employer/landlord etc.)
2. The REMEDY or enforcement mechanism available to the complainant

Output ONLY valid JSON:
{{
  "query_1": "...",
  "query_2": "..."
}}"""
            resp = self._call_ollama_api(prompt, temperature=0.0, format_json=True)
            res_dict = json.loads(resp)
            for key in ["query_1", "query_2"]:
                val = res_dict.get(key, "").strip()
                if val and val not in queries:
                    queries.append(val)
            logger.info(f"Expanded retrieval queries ({issue_type}): {queries}")
        except Exception as e:
            logger.warning(f"Query expansion LLM call failed, using hint queries + original: {e}")
        
        return queries

    def retrieve_context_node(self, state: LegalState) -> Dict[str, Any]:
        """
        Node 2: Embeds the English query, queries Qdrant for semantic hybrid matches,
        filtering by user jurisdiction, and queries Neo4j to fetch citing cases.
        Uses query expansion to retrieve duty, remedy, and enforcement sections.
        """
        import re
        english_query = state["extracted_intent"]["english_query"]
        logger.info(f"Executing Universal GraphRAG retrieval for: '{english_query[:50]}'")
        
        jurisdiction = state["extracted_intent"].get("jurisdiction")
        
        # 1. Expand query into multiple targeted legal queries
        issue_type = state["extracted_intent"].get("issue_type")
        expanded_queries = self._expand_retrieval_queries(english_query, issue_type=issue_type)
        
        # 2. Run hybrid search for each expanded query and merge results
        seen_ids = set()
        all_qdrant_results = []
        for eq in expanded_queries:
            query_vector = self.model.encode(eq).tolist()
            results = self.vector_store.hybrid_search(
                query_vector=query_vector,
                query_text=eq,
                top_k=5,
                jurisdiction=jurisdiction
            )
            for doc in results:
                doc_id = doc.get("id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_qdrant_results.append(doc)
        
        retrieved_contexts = []
        for doc in all_qdrant_results:
            # 3. Clean up the internal RAPTOR layers annotation metadata from citation
            raw_citation = doc.get("citation", "Relevant Statute")
            cleaned_citation = raw_citation.split(" (RAPTOR")[0].strip()

            # 4. For each search hit, query Neo4j for surrounding structural precedents
            db_sec_id = doc.get("section_id") or doc["id"]
            if isinstance(db_sec_id, int):
                db_sec_id = f"kerala_buildings_rent_control_1965_sec_{db_sec_id}"
            else:
                db_sec_id = str(db_sec_id)
            
            graph_data = self.graph_store.get_related_provisions(db_sec_id)
            
            context_entry = {
                "id": doc["id"],
                "text": doc["text"],
                "citation": cleaned_citation,
                "layer_depth": doc["layer_depth"],
                "graph_precedents": graph_data.get("citing_cases", []) if graph_data else [],
                "section_id": doc.get("section_id"),
            }
            retrieved_contexts.append(context_entry)
            
        logger.info(f"Retrieved {len(retrieved_contexts)} integrated contexts (from {len(expanded_queries)} expanded queries).")
        return {"retrieved_docs": retrieved_contexts}

    def filter_rerank_node(self, state: LegalState) -> Dict[str, Any]:
        """Node 3: Context filtering using BGE-Reranker in English space."""
        logger.info("Reranking retrieved passages...")
        english_query = state["extracted_intent"]["english_query"]
        
        if not state["retrieved_docs"]:
            return {"reranked_docs": []}
            
        reranked = []
        if self.reranker:
            pairs = [(english_query, doc["text"]) for doc in state["retrieved_docs"]]
            scores = self.reranker.predict(pairs, activation_fn=lambda x: x)
            
            for idx, doc in enumerate(state["retrieved_docs"]):
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = float(scores[idx])
                reranked.append(doc_copy)
                
            reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        else:
            for idx, doc in enumerate(state["retrieved_docs"]):
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = float(doc.get("score", 0.5))
                reranked.append(doc_copy)
            reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
                
        logger.info(f"Reranked {len(reranked)} docs. Top score: {reranked[0]['relevance_score'] if reranked else 'N/A'}")
        return {"reranked_docs": reranked}

    def generate_roadmap_node(self, state: LegalState) -> Dict[str, Any]:
        """Node 5: Dynamic LLM Generation using retrieved statutory grounds."""
        print("DEBUG STATE STATUS:", state.get("status"))
        if state["status"] == "UNVERIFIED_LEGAL_GROUNDS":
            rejection_msg = (
                "STATUS: UNVERIFIED_LEGAL_GROUNDS\n"
                "The legal information required to answer your query could not be verified "
                "against authenticated statutory sources. To prevent structural risk and incorrect "
                "guidance, the request has been short-circuited."
            )
            return {"response_text": rejection_msg}
            
        logger.info("Generating verified IRAC roadmap...")
        
        # Iterate over reranked docs to find the first one that passes jurisdiction validation
        for doc_idx, doc in enumerate(state["reranked_docs"]):
            citation = doc.get("citation", "Relevant Statute")
            text = doc.get("text", "")
            
            ollama_success = False
            roadmap = ""
            
            try:
                # First sequential call: generate assessment and English layperson advice
                prompt = (
                    "You are a professional legal assistant. Analyze the user query and the retrieved legal context to create a detailed legal assessment and a layperson action guide.\n\n"
                    f"User Query: {state['user_query']}\n"
                    f"Retrieved Legal Context (Citation: {citation}):\n{text}\n\n"
                    "CRITICAL SAFETY WARNINGS:\n"
                    "- The 'RULE' section MUST be based strictly and directly on the provided Retrieved Legal Context.\n"
                    "- Do NOT make up, assume, or fabricate any statutory language or sections (e.g. do not invent section text or state that a section contains a rule that is not in the text).\n"
                    "- If the Retrieved Legal Context is just a short title/extent clause, do NOT invent details of other substantive sections that are not in the context.\n"
                    "- If the rule is not explicitly clear or detailed in the provided context, state that clearly under RULE.\n\n"
                    "Return ONLY a JSON object matching this schema (do not include any conversational preamble, notes, or markdown formatting, just pure raw JSON):\n"
                    "{\n"
                    '  "ISSUE": "state the specific legal issue, referencing the citation",\n'
                    '  "RULE": "state the legal rule based strictly on the retrieved context",\n'
                    '  "APPLICATION": "apply the rule to the user\'s specific facts",\n'
                    '  "CONCLUSION": "state the legal conclusion and remedies available",\n'
                    '  "LAYPERSON": "provide simple, actionable, empathetic advice in English for a common citizen, under 3 sentences"\n'
                    "}"
                )
                
                import json
                response_json = self._call_ollama_api(prompt, temperature=0.2, format_json=True)
                res_dict = json.loads(response_json)
                
                issue_val = res_dict.get("ISSUE", "").strip()
                rule_val = res_dict.get("RULE", "").strip()
                app_val = res_dict.get("APPLICATION", "").strip()
                conc_val = res_dict.get("CONCLUSION", "").strip()
                lay_val = res_dict.get("LAYPERSON", "").strip()
                
                if issue_val and rule_val and app_val and conc_val and lay_val:
                    # Second sequential call: Translate English advice to Malayalam
                    translation_prompt = (
                        "You are an expert English-to-Malayalam translator. Translate the following English text into natural, readable, and polite Malayalam. "
                        "Do not include any conversational filler, explanations, or introductory notes. Only return the Malayalam translation:\n\n"
                        f"{lay_val}"
                    )
                    lay_ml_val = self._call_ollama_api(translation_prompt, temperature=0.0)
                    lay_ml_val = lay_ml_val.strip().replace('"', '').replace("'", "")
                    
                    roadmap = (
                        f"LEGAL ROADMAP (IRAC FORMAT):\n"
                        f"ISSUE: {issue_val}\n"
                        f"RULE: {rule_val}\n"
                        f"APPLICATION: {app_val}\n"
                        f"CONCLUSION: {conc_val}\n"
                        f"LAYPERSON: {lay_val}\n"
                        f"LAYPERSON_ML: {lay_ml_val}\n\n"
                        f"Would you like me to draft a formal legal notice based on this assessment?"
                    )
                    ollama_success = True
                    logger.info(f"✓ Dynamically generated and translated roadmap via Ollama for doc {doc_idx}.")
            except Exception as e:
                logger.warning(f"Ollama generation failed in roadmap node for doc {doc_idx}: {e}")
                continue
                
            if not ollama_success:
                continue
                
            # Post-generation gate validation: Check for placeholders or wrong jurisdiction mentions
            has_placeholder = "***" in rule_val or "N/A" in rule_val or not rule_val.strip()
            
            has_wrong_jurisdiction = False
            user_jur = state["extracted_intent"].get("jurisdiction")
            if user_jur and user_jur.lower() != "central":
                user_jur_clean = user_jur.lower().strip()
                # Check for state-specific act citations of a different state to prevent cross-contamination.
                all_states = [
                    "kerala", "tamil nadu", "karnataka", "maharashtra", "delhi", 
                    "uttar pradesh", "telangana", "andhra pradesh", "west bengal", 
                    "gujarat", "rajasthan", "punjab", "haryana", "bihar", "goa"
                ]
                roadmap_lower = roadmap.lower()
                for other_state in all_states:
                    if other_state != user_jur_clean and other_state in roadmap_lower:
                        if (
                            f"{other_state} prohibition" in roadmap_lower or 
                            f"{other_state} rent" in roadmap_lower or 
                            f"{other_state} act" in roadmap_lower or 
                            f"{other_state} regulations" in roadmap_lower
                        ):
                            has_wrong_jurisdiction = True
                            logger.warning(f"Python validation gate: detected mismatching state '{other_state}' in roadmap for user in '{user_jur_clean}'")
                            break
            
            if has_placeholder or has_wrong_jurisdiction:
                logger.warning(f"Doc {doc_idx} failed validation: placeholder={has_placeholder}, wrong_jurisdiction={has_wrong_jurisdiction}")
                continue
                
            # If we reach here, this document and generated roadmap are valid!
            return {"response_text": roadmap}
            
    def _wikipedia_search(self, query: str) -> Dict[str, str]:
        """Search Wikipedia for a topic and retrieve clean text summary."""
        try:
            # 1. Search for titles
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'LegalMind-Agent/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                results = data.get("query", {}).get("search", [])
                if not results:
                    return None
                best_title = results[0]["title"]
                
            # 2. Get full content of the best title
            encoded_title = urllib.parse.quote(best_title)
            content_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={encoded_title}&format=json"
            req = urllib.request.Request(content_url, headers={'User-Agent': 'LegalMind-Agent/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content_data = json.loads(response.read().decode('utf-8'))
                pages = content_data.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    if "extract" in page_info:
                        return {
                            "title": best_title,
                            "text": page_info["extract"]
                        }
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
        return None

    def _ddg_lite_search(self, query: str) -> Dict[str, str]:
        """Search DuckDuckGo Lite and download first organic search result's page text."""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
                links = re.findall(r'href="(https?://[^"]+)"', html)
                valid_links = [l for l in links if "duckduckgo.com" not in l and "ad_provider" not in l]
                if valid_links:
                    first_link = valid_links[0]
                    logger.info(f"DuckDuckGo Lite search top result: {first_link}")
                    req2 = urllib.request.Request(first_link, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req2, timeout=10) as resp2:
                        content = resp2.read().decode('utf-8', errors='ignore')
                        content = re.sub(r'<script[^>]*?>.*?</script>', ' ', content, flags=re.DOTALL)
                        content = re.sub(r'<style[^>]*?>.*?</style>', ' ', content, flags=re.DOTALL)
                        text_only = re.sub(r'<[^>]+>', ' ', content)
                        text_only = re.sub(r'\s+', ' ', text_only).strip()
                        return {
                            "title": query,
                            "text": text_only[:8000]
                        }
        except Exception as e:
            logger.warning(f"DuckDuckGo Lite search failed: {e}")
        return None

    def _agentic_search(self, topic: str) -> Dict[str, str]:
        """Runs the search fallback chain (Wikipedia -> DDG Lite)."""
        logger.info(f"Executing web search chain for query topic: '{topic}'")
        res = self._wikipedia_search(topic)
        if res:
            logger.info("✓ Wikipedia search returned content successfully.")
            return res
        
        logger.info("Wikipedia returned no results or failed. Trying DuckDuckGo Lite...")
        res = self._ddg_lite_search(topic)
        if res:
            logger.info("✓ DuckDuckGo Lite search returned content successfully.")
            return res
            
        return None

    def agentic_research_node(self, state: LegalState) -> Dict[str, Any]:
        """Agentic Research Loop Node: perform web search, parse results, and dynamically ingest."""
        attempts = state.get("research_attempts", 0)
        logger.info(f"Agentic research loop node invoked. Attempt: {attempts + 1}")
        
        query = state.get("user_query", "")
        category = state.get("extracted_intent", {}).get("issue_type")
        hint_text = ""
        if category and category in ISSUE_TYPE_STATUTE_HINTS:
            hints = ISSUE_TYPE_STATUTE_HINTS[category]
            hint_text = f"\nCategory of issue: {category}\nTarget statutes/regulations to search for: {hints.get('target_laws')}"

        prompt = (
            "You are a legal search query formulator. Given the user's legal panic query below and the category hints, "
            "write a single simple search query (e.g. name of the Act, Statute, or Regulation) designed to retrieve the relevant Indian Statute or Act from Wikipedia/Web search.\n"
            "Keep it broad and search-friendly (e.g. 'Maternity Benefit Act 1961 India' or 'Ragging in India' or 'Payment of Wages Act India'). "
            "Do NOT include section numbers in the search query. "
            "Only return the search terms, with no quotes, conversational preamble, or explanations:\n\n"
            f"Query: {query}"
            f"{hint_text}"
        )
        try:
            search_query = self._call_ollama_api(prompt, temperature=0.0).replace('"', '').replace("'", "").strip()
            logger.info(f"Formulated search term: '{search_query}'")
        except Exception as e:
            logger.warning(f"Failed to formulate search term with LLM: {e}. Using raw query.")
            search_query = query
            
        search_res = self._agentic_search(search_query)
        if not search_res:
            extracted_category = state.get("extracted_intent", {}).get("issue_type")
            if extracted_category:
                logger.info(f"Retrying search with category: '{extracted_category}'")
                search_res = self._agentic_search(f"{extracted_category} Act India")
                
        if not search_res:
            logger.warning("No search results could be retrieved from any external sources.")
            return {"research_attempts": attempts + 1}
            
        title = search_res["title"]
        text = search_res["text"]
        
        sections = []
        wiki_sections = re.findall(r'={2,}\s*([^=]+?)\s*={2,}\n(.*?)(?=\n={2,}|$)', text, re.DOTALL)
        if wiki_sections:
            for i, (sec_title, body) in enumerate(wiki_sections):
                body_clean = body.strip()
                if body_clean:
                    sections.append({
                        "num": str(i + 1),
                        "title": sec_title.strip(),
                        "body": body_clean
                    })
        
        if not sections:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunk = ""
            chunk_idx = 1
            for p in paragraphs:
                if len(chunk) + len(p) < 1200:
                    chunk += "\n\n" + p
                else:
                    if chunk.strip():
                        sections.append({
                            "num": str(chunk_idx),
                            "title": f"Paragraph {chunk_idx}",
                            "body": chunk.strip()
                        })
                        chunk_idx += 1
                    chunk = p
            if chunk.strip():
                sections.append({
                    "num": str(chunk_idx),
                    "title": f"Paragraph {chunk_idx}",
                    "body": chunk.strip()
                })
                
        if not sections:
            logger.warning("Parsed text was empty or could not be sectioned.")
            return {"research_attempts": attempts + 1}
            
        statute_id = "agentic_" + re.sub(r'[^a-zA-Z0-9]', '_', title.lower()).strip('_')
        jurisdiction = state.get("extracted_intent", {}).get("jurisdiction") or "Central"
        jurisdiction_clean = jurisdiction.lower().strip()
        
        logger.info(f"Dynamically ingesting statute '{title}' (ID: {statute_id}) to Neo4j and Qdrant...")
        
        try:
            self.graph_store.add_statute(statute_id, title, jurisdiction)
        except Exception as e:
            logger.warning(f"Neo4j add_statute failed: {e}")
            
        points = []
        for sec in sections:
            sec_num = sec["num"]
            sec_title = sec["title"]
            sec_body = sec["body"]
            section_id = f"{statute_id}_sec_{sec_num}"
            citation = f"Section {sec_num} ({sec_title}), {title}"
            
            try:
                self.graph_store.add_section(
                    statute_id=statute_id,
                    section_id=section_id,
                    title=f"Section {sec_num}: {sec_title}",
                    text=sec_body,
                    citation=citation
                )
            except Exception as e:
                logger.warning(f"Neo4j add_section failed: {e}")
                
            vector = self.model.encode(sec_body).tolist()
            point_id = str(uuid.uuid4())
            points.append({
                "id": point_id,
                "vector": vector,
                "payload": {
                    "text": sec_body,
                    "citation": f"{citation} (Agentic RAG)",
                    "layer_depth": 0,
                    "section_id": section_id,
                    "jurisdiction": jurisdiction_clean
                }
            })
            
        if points:
            try:
                self.vector_store.upsert_chunks(points)
                logger.info(f"✓ Dynamic ingestion completed. Upserted {len(points)} sections.")
            except Exception as e:
                logger.error(f"Qdrant dynamic ingestion failed: {e}")
                
        return {"research_attempts": attempts + 1}

    def citation_shield_gate(self, state: LegalState) -> Dict[str, Any]:
        """Node 4: Validate citations dynamically using semantic threshold constraints."""
        logger.info("Evaluating context faithfulness shield...")
        
        if not state["reranked_docs"]:
            return {"faithfulness_score": 0.0, "status": "UNVERIFIED_LEGAL_GROUNDS"}
            
        threshold = state.get("threshold", self.threshold)
        
        # Calculate calibrated score for each document
        verified_docs = []
        best_prob = 0.0
        
        for idx, doc in enumerate(state["reranked_docs"]):
            score = doc.get("relevance_score", 0.0)
            if self.reranker is None:
                prob_score = score
            else:
                import math
                try:
                    calibrated_logit = score + 7.0
                    prob_score = 1.0 / (1.0 + math.exp(-calibrated_logit))
                except Exception:
                    prob_score = score
            
            doc["calibrated_score"] = prob_score
            if idx == 0:
                best_prob = prob_score
            
            if prob_score >= threshold:
                verified_docs.append(doc)
                
        if verified_docs:
            status = "SUCCESS"
            best_prob = verified_docs[0]["calibrated_score"]
        else:
            status = "UNVERIFIED_LEGAL_GROUNDS"
            
        logger.info(f"Context Shield Gate: {len(verified_docs)}/{len(state['reranked_docs'])} docs verified above threshold {threshold}. Status: {status}")
        
        return {
            "faithfulness_score": best_prob,
            "status": status,
            "reranked_docs": verified_docs if verified_docs else state["reranked_docs"]
        }

    def _build_workflow(self):
        workflow = StateGraph(LegalState)
        
        # Add Nodes
        workflow.add_node("extract_intent", self.extract_intent_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("filter_rerank", self.filter_rerank_node)
        workflow.add_node("citation_shield", self.citation_shield_gate)
        workflow.add_node("agentic_research", self.agentic_research_node)
        workflow.add_node("generate_roadmap", self.generate_roadmap_node)
        
        # Set Edges
        workflow.set_entry_point("extract_intent")
        workflow.add_edge("extract_intent", "retrieve_context")
        workflow.add_edge("retrieve_context", "filter_rerank")
        workflow.add_edge("filter_rerank", "citation_shield")
        
        def shield_router(state: LegalState) -> str:
            if state.get("status") == "SUCCESS":
                return "generate_roadmap"
            
            # If threshold failed, see if we should try web research
            attempts = state.get("research_attempts", 0)
            if attempts < 1:
                logger.info("Relevance threshold check failed. Routing to agentic_research node.")
                return "agentic_research"
                
            logger.info("Relevance threshold check failed and research attempt already made. Routing to generate_roadmap fallback.")
            return "generate_roadmap"
            
        workflow.add_conditional_edges(
            "citation_shield",
            shield_router,
            {
                "generate_roadmap": "generate_roadmap",
                "agentic_research": "agentic_research"
            }
        )
        workflow.add_edge("agentic_research", "retrieve_context")
        workflow.add_edge("generate_roadmap", END)
        
        self.app = workflow.compile()

    def _call_slot_extractor(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        import json
        from datetime import datetime
        
        session_slots = {
            "issue_type": None,
            "incident_date": None,
            "jurisdiction": None,
            "institution": None,
            "incident_description": None,
            "parties_mentioned": [],
            "sender_name": None,
            "recipient_name": None,
            "slots_complete": False
        }
        
        # Format history so far
        history_lines = []
        for turn in (history or []):
            role = "User" if turn.get("role") == "user" else "Assistant"
            text = turn.get("text") or turn.get("response_text") or ""
            history_lines.append(f"[{role}]: {text}")
        history_str = "\n".join(history_lines)
        
        current_date_str = "2026-06-13"  # Mocked current date corresponding to conversation local time
        
        prompt = f"""You are an accurate, objective slot extractor for a legal intake system.
Your job is to read the conversation history and the latest user message, and extract specific details of the user's legal issue.

CRITICAL RULES:
- Do NOT hallucinate. Do NOT invent details.
- Only extract a value if it is explicitly mentioned or clearly implied in the conversation.
- If a detail is NOT mentioned anywhere in the conversation history or latest message, you MUST return null for that slot.
- For example, if the user has only said a greeting like "hi", "hello", "hey", all slots MUST be null.
- Do NOT output placeholder names like "name1", "name2", "college name", "facing ragging at college", or similar unless they are actually written in the user messages.

Current local date: {current_date_str}

Please output a JSON object containing the values of the following slots based strictly on the conversation:
- "issue_type": Must be one of "ragging", "eviction", "wage_theft", "consumer", "other", or null. Only set this if the user describes a legal issue.
- "incident_date": The date of the incident (YYYY-MM-DD) or null. If the user mentions a relative date like "yesterday", resolve it relative to the current local date ({current_date_str}) to a YYYY-MM-DD format. Otherwise, if no date/time is mentioned, return null.
- "jurisdiction": The Indian state name where the incident occurred, or null.
- "institution": The name of the college, company, school, or landlord's building mentioned, or null.
- "incident_description": A brief one-sentence description of the incident as described by the user, or null.
- "parties_mentioned": A list of names of specific individuals involved (excluding generic terms like 'sender' or 'recipient'), or [].
- "sender_name": The name of the sender/complainant/client sending the notice, or null. Only extract if explicitly provided (e.g., "sender is Vishnu" or "my name is Vishnu").
- "recipient_name": The name of the recipient/respondent/opposing party receiving the notice, or null. Only extract if explicitly provided (e.g., "recipient is Kavya" or "opposing party is Kavya").

Conversation history:
{history_str}

Latest user message:
{query}

Output ONLY valid JSON matching the format below. No markdown formatting, no code blocks, no explanations.

Output format:
{{
  "issue_type": "ragging" | "eviction" | "wage_theft" | "consumer" | "other" | null,
  "incident_date": "YYYY-MM-DD" or null,
  "jurisdiction": "state name" or null,
  "institution": "institution name" or null,
  "incident_description": "description text" or null,
  "parties_mentioned": ["name"] or [],
  "sender_name": "sender name" or null,
  "recipient_name": "recipient name" or null
}}
"""
        try:
            resp = self._call_ollama_api(prompt, temperature=0.0, format_json=True)
            res_dict = json.loads(resp)
            # Ensure all required keys exist
            for k in session_slots:
                if k not in res_dict:
                    res_dict[k] = session_slots[k]
            
            # Clean up string 'null' or 'None' values to Python None
            for k in list(res_dict.keys()):
                if isinstance(res_dict[k], str) and res_dict[k].lower() in ["null", "none", ""]:
                    res_dict[k] = None
                if k == "jurisdiction" and isinstance(res_dict[k], str):
                    res_dict[k] = res_dict[k].lower()
            
            # Double check slots_complete logic
            req_fields = ["issue_type", "incident_date", "jurisdiction", "incident_description"]
            is_complete = all(res_dict.get(f) is not None and str(res_dict.get(f)).lower() != "null" and str(res_dict.get(f)).strip() != "" for f in req_fields)
            res_dict["slots_complete"] = is_complete
            
            return res_dict
        except Exception as e:
            logger.warning(f"Slot extractor failed: {e}")
            return session_slots

    def _call_state_router(self, slots: Dict[str, Any], query: str, history: List[Dict[str, str]]) -> str:
        import json
        
        # Determine last_state first using Python rules matching previous assistant text
        last_state = "GREETING"
        last_assistant_msg = ""
        for turn in reversed(history or []):
            if turn.get("role") == "assistant":
                last_assistant_msg = turn.get("text") or turn.get("response_text") or ""
                break
        
        if not last_assistant_msg:
            last_state = "GREETING"
        elif "DOWNLOAD_URL" in last_assistant_msg:
            last_state = "NOTICE_DRAFT"
        elif "names of the Sender and the Recipient" in last_assistant_msg or "പേരുകൾ" in last_assistant_msg or "Sender and the Recipient" in last_assistant_msg:
            last_state = "NOTICE_INTAKE"
        elif "LEGAL ROADMAP" in last_assistant_msg or "Would you like me to draft" in last_assistant_msg or "Would you like a notice" in last_assistant_msg:
            last_state = "NOTICE_OFFER"
        elif any(w in last_assistant_msg.lower() for w in ["when", "where", "which state", "what is", "സംസ്ഥാന", "എപ്പോൾ", "എവിടെ"]):
            last_state = "SLOT_FILLING"
        else:
            last_state = "FOLLOWUP"

        # Determine if IRAC roadmap has ever been delivered in the history
        irac_delivered = False
        for turn in (history or []):
            if turn.get("role") == "assistant" and "LEGAL ROADMAP" in (turn.get("text") or turn.get("response_text") or ""):
                irac_delivered = True
                break

        # --- DETERMINISTIC PYTHON ROUTING ---
        if not history:
            return "GREETING"

        def is_valid_name(val):
            if val is None:
                return False
            val_str = str(val).strip().lower()
            if val_str in ["", "null", "none", "undefined"]:
                return False
            if val_str in ["name1", "name2", "college name", "institution name", "recipient name", "sender name"]:
                return False
            return True

        sender_name = slots.get("sender_name")
        recipient_name = slots.get("recipient_name")

        # Once IRAC roadmap has been delivered, NEVER loop back to SLOT_FILLING or RAG_RETRIEVE.
        if irac_delivered:
            if is_valid_name(sender_name) and is_valid_name(recipient_name):
                return "NOTICE_DRAFT"
            
            prompt = f"""You are a dialogue state router for a legal assistant.
The IRAC roadmap has already been delivered to the user.

Latest user message: "{query}"
Last assistant action: {last_state}

Classify the user's intent into one of the following states:
1. NOTICE_INTAKE: The user is agreeing/consenting to draft the notice, OR providing names for the draft.
2. FOLLOWUP: The user is asking a question about the laws, the roadmap, their case, or requesting clarification.
3. NOTICE_DRAFT: Both sender and recipient names are already provided.

Output ONLY the state name (NOTICE_INTAKE, FOLLOWUP, or NOTICE_DRAFT). Do not include any other text.
"""
            try:
                expected_state = self._call_ollama_api(prompt, temperature=0.0).strip().upper()
                for st in ["NOTICE_DRAFT", "NOTICE_INTAKE", "FOLLOWUP"]:
                    if st in expected_state:
                        return st
                return "FOLLOWUP"
            except Exception as e:
                logger.warning(f"Post-IRAC state router failed: {e}")
                return "FOLLOWUP"
        else:
            # Before IRAC delivery, normal slot-filling flow
            if not slots.get("slots_complete"):
                return "SLOT_FILLING"
            return "RAG_RETRIEVE"

        # Format history so far
        history_lines = []
        for turn in (history or []):
            role = "User" if turn.get("role") == "user" else "Assistant"
            text = turn.get("text") or turn.get("response_text") or ""
            history_lines.append(f"[{role}]: {text}")
        history_str = "\n".join(history_lines)

        prompt = f"""You are a dialogue state router for a legal assistant.

Given the session state and conversation history, output the next state.

Session slots: {json.dumps(slots)}
Slots complete: {slots.get('slots_complete')}
IRAC roadmap delivered: {irac_delivered}
Last assistant action: {last_state}
Latest user message: {query}

Conversation history so far:
{history_str}

DECISION FLOW (Evaluate these rules sequentially from 1 to 6. Output the first matching state):
1. If conversation history is empty and this is the first turn -> GREETING
2. If Slots complete is false -> SLOT_FILLING
3. If Slots complete is true AND IRAC roadmap delivered is false -> RAG_RETRIEVE
4. If sender_name and recipient_name in Session slots are both non-null (not null) -> NOTICE_DRAFT
5. If Last assistant action is NOTICE_OFFER AND the user message expresses consent/yes -> NOTICE_INTAKE
6. If IRAC roadmap delivered is true AND Last assistant action is NOTICE_OFFER AND the user says NO/declines -> FOLLOWUP
7. If IRAC roadmap delivered is true AND the user asks a follow-up question -> FOLLOWUP
8. Otherwise -> CLARIFY

Output ONLY the state name (e.g. GREETING, SLOT_FILLING, RAG_RETRIEVE, NOTICE_OFFER, NOTICE_INTAKE, NOTICE_DRAFT, FOLLOWUP, CLARIFY). Do not explain or add other text.
"""
        try:
            expected_state = self._call_ollama_api(prompt, temperature=0.0).strip().upper()
            # Clean up state string
            for st in ["GREETING", "SLOT_FILLING", "RAG_RETRIEVE", "NOTICE_OFFER", "NOTICE_INTAKE", "NOTICE_DRAFT", "FOLLOWUP", "CLARIFY"]:
                if st in expected_state:
                    return st
            return "SLOT_FILLING"
        except Exception as e:
            logger.warning(f"State router failed: {e}")
            return "SLOT_FILLING"

    def _call_shield_validator(self, expected_state: str, slots: Dict[str, Any], response: str, retrieved_ids: List[str], last_assistant_response: str) -> Dict[str, Any]:
        import json
        prompt = f"""You are a response validator for a legal assistant.

Check the assistant's response against these rules and return a JSON checks object.

Expected state: {expected_state}
Session slots: {json.dumps(slots)}
Assistant response: {response}
Last assistant response: {last_assistant_response}
Retrieved statute IDs: {json.dumps(retrieved_ids)}

Checks:
1. state_match: Does the response match the expected_state behavior?
   - GREETING: Should welcome the user.
   - SLOT_FILLING: Should ask a clarifying question for one of the missing slots.
   - RAG_RETRIEVE/NOTICE_OFFER: Should contain a LEGAL ROADMAP/IRAC assessment or offer a notice.
   - NOTICE_INTAKE: Should ask for names of sender/recipient.
   - NOTICE_DRAFT: Should display the formal legal notice.
   - FOLLOWUP: Should answer the user's follow-up question.
   - CLARIFY: Should ask the user to clarify.
   Only set to false if the response completely mismatches the expected state's purpose.

2. no_repeat: Set to true. Only set to false if the assistant response is extremely similar or identical to the last assistant response (e.g. asking the same question again or repeating the same message, causing a loop).

3. slots_respected: Set to true. Only set to false if the session slots are complete (slots_complete = true) AND the assistant response asks for slot details (like when/where it happened) again. If slots are NOT complete, asking for missing slots is expected, so set this to true.

4. statute_grounded: Set to true. Only set to false if the expected_state is RAG_RETRIEVE or NOTICE_OFFER, the response cites/references specific legal statutes, and those statutes are NOT present in the retrieved statute IDs: {json.dumps(retrieved_ids)}.

5. rule_non_empty: Set to true. Only set to false if the response contains a "LEGAL ROADMAP (IRAC FORMAT)", but the "RULE" (or R) section is empty, null, or "***".

6. no_internal_tags: Set to true. Only set to false if the response contains raw internal metadata tags, like "(RAPTOR Layer N)".

Output format:
{{
  "checks": {{
    "state_match": true/false,
    "no_repeat": true/false,
    "slots_respected": true/false,
    "statute_grounded": true/false,
    "rule_non_empty": true/false,
    "no_internal_tags": true/false
  }}
}}
"""
        default_val = {
            "checks": {
                "state_match": True,
                "no_repeat": True,
                "slots_respected": True,
                "statute_grounded": True,
                "rule_non_empty": True,
                "no_internal_tags": True
            },
            "score": 1.0,
            "fail_reasons": []
        }
        try:
            resp = self._call_ollama_api(prompt, temperature=0.0, format_json=True)
            res_dict = json.loads(resp)
            checks = res_dict.get("checks", {})
            # Ensure all check fields are present
            for k in default_val["checks"]:
                if k not in checks:
                    checks[k] = default_val["checks"][k]
            
            passed = sum(1 for v in checks.values() if v is True)
            score = passed / len(checks)
            
            return {
                "checks": checks,
                "score": score,
                "fail_reasons": [k for k, v in checks.items() if v is False]
            }
        except Exception as e:
            logger.warning(f"Shield validator failed to run: {e}")
            return default_val

    def analyze_dialogue_state(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        slots = self._call_slot_extractor(query, history)
        state = self._call_state_router(slots, query, history)
        
        return {
            "thought_process": "",
            "state": state,
            "missing_details": [k for k, v in slots.items() if v is None and k != "slots_complete" and k != "parties_mentioned"],
            "clarifying_question": "",
            "facts_summary": slots.get("incident_description") or "",
            "jurisdiction": slots.get("jurisdiction"),
            "approximate_date": slots.get("incident_date"),
            "incident_description": slots.get("incident_description"),
            "sender_name": None,
            "recipient_name": None
        }

    def generate_notice_document_file(self, draft: str, sender: str, recipient: str) -> str:
        """Helper to compile and save drafted legal notice to PDF/TXT."""
        import os
        output_dir = "data/synthesis"
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, "formal_notice.pdf")
        
        from datetime import date
        notice_date = date.today().strftime("%d %B %Y")
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; color: #111; }}
                .header {{ text-align: center; font-weight: bold; font-size: 22px; text-decoration: underline; margin-bottom: 30px; }}
                .meta {{ margin-bottom: 20px; font-size: 14px; }}
                .content {{ white-space: pre-wrap; font-size: 14px; text-align: justify; }}
                .signature {{ margin-top: 50px; text-align: right; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="header">FORMAL LEGAL NOTICE</div>
            <div class="meta">
                <p><strong>Date:</strong> {notice_date}</p>
                <p><strong>From (Sender):</strong> {sender}</p>
                <p><strong>To (Recipient):</strong> {recipient}</p>
            </div>
            <div class="content">
{draft}
            </div>
            <div class="signature">
                <p>_________________________</p>
                <p>Signature ({sender})</p>
            </div>
        </body>
        </html>
        """
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(pdf_path)
            return "/api/documents/download?file=formal_notice.pdf"
        except Exception as e:
            logger.warning(f"WeasyPrint failed in pipeline generator: {e}")
            txt_path = pdf_path.replace(".pdf", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return "/api/documents/download?file=formal_notice.txt"

    def run(self, query: str, history: List[Dict[str, str]] = None, threshold: float = None) -> Dict[str, Any]:
        import json
        session_slots = self._call_slot_extractor(query, history)
        
        # Auto-resolve jurisdiction from city names (fixes ernakulam → kerala, kochi → kerala, etc.)
        session_slots = self._resolve_jurisdiction_from_text(session_slots, query, history)
        
        if not history:
            # Check if any slot is filled (meaning user did not just say "hi")
            has_slots_filled = any(
                session_slots.get(k) is not None 
                and str(session_slots.get(k)).lower() not in ["null", "none", ""]
                for k in ["issue_type", "incident_date", "jurisdiction", "institution", "incident_description"]
            )
            if has_slots_filled:
                if session_slots.get("slots_complete"):
                    expected_state = "RAG_RETRIEVE"
                else:
                    expected_state = "SLOT_FILLING"
            else:
                expected_state = "GREETING"
        else:
            expected_state = self._call_state_router(session_slots, query, history)
        logger.info(f"Dialogue State Analysis: State = {expected_state}, Slots = {session_slots}")
        
        # Build clean dialogue transcript lines
        history_lines = []
        for turn in (history or []):
            role = "User" if turn.get("role") == "user" else "Assistant"
            text = turn.get("text") or turn.get("response_text") or ""
            history_lines.append(f"[{role}]: {text}")
        history_str = "\n".join(history_lines)

        last_assistant_response = ""
        for turn in reversed(history or []):
            if turn.get("role") == "assistant":
                last_assistant_response = turn.get("text") or turn.get("response_text") or ""
                break

        final_response_text = ""
        final_status = "SUCCESS"
        final_faithfulness = 1.0
        retrieved_ids = []

        for attempt in range(3):
            generated_text = ""
            if expected_state == "GREETING":
                gen_prompt = f"""You are a warm, helpful, and professional legal assistant chatbot named LegalMind.
The user has sent a greeting: "{query}".
Respond to the user with a welcoming greeting in English. Explain that you can help them verify their rights under Indian statutes and draft formal legal notices. Prompt them to describe their legal issue, including when and where it occurred, so you can guide them.
Do not include any placeholders, notes, or explanations. Only return the response text."""
                try:
                    english = self._call_ollama_api(gen_prompt, temperature=0.7).strip()
                except Exception:
                    english = "Welcome to LegalMind! I can help you verify your rights under Indian statutes. To start, could you please describe your legal issue, including when and where it occurred?"
                
                translation_prompt = f"""Translate the following greeting to conversational Malayalam.
Keep it simple, natural, and warm. Output Malayalam only.
Greeting: {english}"""
                try:
                    malayalam = self._call_ollama_api(translation_prompt, temperature=0.0).strip()
                except Exception:
                    malayalam = "ലീഗൽമൈൻഡിലേക്ക് സ്വാഗതം! ഇന്ത്യൻ നിയമപ്രകാരമുള്ള നിങ്ങളുടെ അവകാശങ്ങൾ പരിശോധിക്കാൻ ഞാൻ സഹായിക്കാം. തുടങ്ങുന്നതിനായി, നിങ്ങളുടെ പ്രശ്നം എന്താണെന്നും, അത് എപ്പോൾ, എവിടെയാണ് സംഭവിച്ചതെന്നും ദയവായി വ്യക്തമാക്കാമോ?"
                
                generated_text = f"{malayalam}\n\n{english}"
            elif expected_state == "SLOT_FILLING":
                # Determine if IRAC roadmap has ever been delivered in the history
                irac_delivered = False
                for turn in (history or []):
                    text_content = turn.get("text") or turn.get("response_text") or ""
                    if turn.get("role") == "assistant" and "LEGAL ROADMAP" in text_content:
                        irac_delivered = True
                        break
                
                if not irac_delivered:
                    # Before RAG retrieval, only ask for RAG-essential slots
                    rag_essential = ["issue_type", "incident_date", "jurisdiction", "incident_description"]
                    missing = [k for k in rag_essential if session_slots.get(k) is None or str(session_slots.get(k)).lower() in ["null", "none", ""]]
                else:
                    # After roadmap delivery, ask for all missing slots (notice intake)
                    missing = [k for k, v in session_slots.items() if v is None and k not in ["slots_complete", "parties_mentioned"]]
                
                missing_str = ", ".join(missing) if missing else "more details"
                gen_prompt = f"""You are a helpful legal assistant.
Given the current session slots: {json.dumps(session_slots)}
And the conversation history:
{history_str}

Generate a polite, brief English question asking the user for ONE of the missing slots: {missing_str}.
Do not include any greetings, notes, or intros. Only return the question."""
                english_response = self._call_ollama_api(gen_prompt, temperature=0.5).strip()
                
                translation_prompt = f"""Translate the following legal assistant message to Malayalam.
Keep it conversational and simple. Do not add or remove content.
Use plain Malayalam, not legal jargon.
Source English: {english_response}
Output Malayalam only, no English."""
                malayalam_response = self._call_ollama_api(translation_prompt, temperature=0.0).strip()
                generated_text = f"{malayalam_response}\n\n{english_response}"
            elif expected_state in ["RAG_RETRIEVE", "NOTICE_OFFER"]:
                search_query = session_slots["incident_description"] if session_slots["incident_description"] else query
                initial_state = {
                    "user_query": search_query,
                    "extracted_intent": {
                        "jurisdiction": session_slots.get("jurisdiction"),
                        "issue_type": session_slots.get("issue_type")
                    },
                    "retrieved_docs": [],
                    "reranked_docs": [],
                    "response_text": "",
                    "faithfulness_score": 0.0,
                    "status": "SUCCESS",
                    "threshold": threshold if threshold is not None else self.threshold,
                    "research_attempts": 0
                }
                result = self.app.invoke(initial_state)
                generated_text = result["response_text"]
                final_status = result["status"]
                final_faithfulness = result["faithfulness_score"]
                retrieved_ids = [doc.get("section_id") or doc["id"] for doc in result.get("retrieved_docs", [])] + [doc.get("citation") for doc in result.get("retrieved_docs", [])]
            elif expected_state == "NOTICE_INTAKE":
                gen_prompt = f"""You are a helpful, professional legal assistant chatbot.
The user wants to draft a formal legal notice.
Acknowledge their request in English. Explain politely that before you can compile the document, you need the names of the Sender (the person sending the notice) and the Recipient (the opposing party/person receiving the notice).
Do not include any placeholders, notes, or explanations. Only return the response text."""
                try:
                    english = self._call_ollama_api(gen_prompt, temperature=0.5).strip()
                except Exception:
                    english = "Sure! I can help you draft a formal legal notice. Before we proceed, could you please provide the names of the Sender and the Recipient?"
                
                translation_prompt = f"""Translate the following message to conversational Malayalam.
Keep it simple and clear. Output Malayalam only.
Message: {english}"""
                try:
                    malayalam = self._call_ollama_api(translation_prompt, temperature=0.0).strip()
                except Exception:
                    malayalam = "തീർച്ചയായും! ഒരു ഔദ്യോഗിക ലീഗൽ നോട്ടീസ് തയ്യാറാക്കാൻ ഞാൻ സഹായിക്കാം. തുടങ്ങുന്നതിന് മുന്നോടിയായി, നോട്ടീസ് അയക്കുന്നയാളുടെയും (Sender), ലഭിക്കേണ്ടയാളുടെയും (Recipient) പേരുകൾ ദയവായി വ്യക്തമാക്കാമോ?"
                
                generated_text = f"{malayalam}\n\n{english}"
            elif expected_state == "NOTICE_DRAFT":
                names_prompt = f"""Given the conversation history, extract the names of the Sender (the person sending the notice / client) and the Recipient (the opposing party / person receiving the notice).
Conversation history:
{history_str}
Latest user message:
{query}

Output ONLY valid JSON matching this schema:
{{
  "sender_name": "name or null",
  "recipient_name": "name or null"
}}
"""
                sender_name = session_slots.get("sender_name")
                recipient_name = session_slots.get("recipient_name")
                
                if not sender_name or not recipient_name or str(sender_name).lower() == "null" or str(recipient_name).lower() == "null":
                    try:
                        names_json = self._call_ollama_api(names_prompt, temperature=0.0, format_json=True)
                        names_dict = json.loads(names_json)
                        sender_name = sender_name or names_dict.get("sender_name")
                        recipient_name = recipient_name or names_dict.get("recipient_name")
                    except Exception:
                        pass
                
                if not sender_name or not recipient_name or str(sender_name).lower() == "null" or str(recipient_name).lower() == "null":
                    parties = session_slots.get("parties_mentioned", [])
                    if len(parties) >= 2:
                        sender_name = sender_name or parties[0]
                        recipient_name = recipient_name or parties[1]
                
                if not sender_name or not recipient_name or str(sender_name).lower() == "null" or str(recipient_name).lower() == "null":
                    gen_prompt = f"""You are a helpful legal assistant chatbot.
The user wants to draft a formal legal notice, but you are still missing the names of the Sender and/or the Recipient.
Politely ask the user in English to provide the missing name(s) (either the Sender's name, the Recipient's name, or both).
Do not include any placeholders, notes, or explanations. Only return the response text."""
                    try:
                        english = self._call_ollama_api(gen_prompt, temperature=0.5).strip()
                    except Exception:
                        english = "Sure! I can help you draft a formal legal notice. Before we proceed, could you please provide the names of the Sender and the Recipient?"
                    
                    translation_prompt = f"""Translate the following message to conversational Malayalam.
Keep it simple and clear. Output Malayalam only.
Message: {english}"""
                    try:
                        malayalam = self._call_ollama_api(translation_prompt, temperature=0.0).strip()
                    except Exception:
                        malayalam = "തീർച്ചയായും! ഒരു ഔദ്യോഗിക ലീഗൽ നോട്ടീസ് തയ്യാറാക്കാൻ ഞാൻ സഹായിക്കാം. തുടങ്ങുന്നതിന് മുന്നോടിയായി, നോട്ടീസ് അയക്കുന്നയാളുടെയും (Sender), ലഭിക്കേണ്ടയാളുടെയും (Recipient) പേരുകൾ ദയവായി വ്യക്തമാക്കാമോ?"
                    
                    generated_text = f"{malayalam}\n\n{english}"
                else:
                    statutes_prompt = f"""Given the conversation history, extract the names of any legal statutes or regulations cited in the legal roadmap or assessment (for example: "Kerala Prohibition Of Ragging Act, 1998").
Conversation history:
{history_str}

Output ONLY the names of the statutes as a comma-separated list. Do not add other text."""
                    try:
                        statutes_cited = self._call_ollama_api(statutes_prompt, temperature=0.0).strip()
                    except Exception:
                        statutes_cited = "Kerala Prohibition Of Ragging Act, 1998"
                    
                    from datetime import date
                    notice_date = date.today().strftime("%d %B %Y")
                    
                    draft_prompt = f"""You are a formal legal notice drafter for an Indian legal aid system.

Your task is to draft a FORMAL LEGAL NOTICE — a structured 
official document used to assert legal rights and demand remedy.
This is NOT a threatening letter. It is a legally recognized 
document used in Indian courts and institutions.

CRITICAL RULES:
- Use ONLY the information provided below.
- If any field (such as address) is NOT provided, write "[TO BE FILLED BY SENDER]" as a placeholder.
- NEVER invent, assume, or fabricate addresses, phone numbers, or any personal details.
- The date of this notice is: {notice_date}

Gathered facts:
- Issue type: {session_slots.get('issue_type')}
- Incident date: {session_slots.get('incident_date')}
- Institution: {session_slots.get('institution')}
- Jurisdiction: {session_slots.get('jurisdiction')}
- Incident description: {session_slots.get('incident_description')}
- Sender (complainant): {sender_name}
- Sender address: [TO BE FILLED BY SENDER]
- Recipient (respondent): {recipient_name}
- Recipient address: [TO BE FILLED BY SENDER]

Applicable law from assessment: {statutes_cited}

Draft a formal legal notice with these sections:
1. Header (date: {notice_date}, sender address placeholder, recipient address placeholder)
2. Subject line
3. Facts (numbered, objective language only)
4. Legal violations — describe which laws were violated and how, citing the statute names from the assessment. DO NOT quote statute text verbatim. Instead, describe the violation in your own formal legal language.
5. Demand / Relief sought
6. Consequence of non-compliance (legal action, not personal threat)
7. Signature block

Language: formal, objective, third-person where appropriate.
DO NOT include threats of physical harm.
DO NOT add any content not in the gathered facts.
DO NOT invent addresses — use [TO BE FILLED BY SENDER] for any unknown address.
DO NOT quote or fabricate statute text verbatim — describe violations in your own words.
DO NOT refuse to draft this — it is a civil legal document."""
                    draft = self._call_ollama_api(draft_prompt, temperature=0.3)
                    download_url = self.generate_notice_document_file(draft, sender_name, recipient_name)
                    generated_text = (
                        f"### DRAFT FORMAL LEGAL NOTICE\n\n"
                        f"**Sender:** {sender_name}\n"
                        f"**Recipient:** {recipient_name}\n\n"
                        f"---\n\n"
                        f"{draft}\n\n"
                        f"[DOWNLOAD_URL:{download_url}]"
                    )
            elif expected_state == "FOLLOWUP":
                gen_prompt = f"""You are a helpful legal assistant.
Answering the user's follow-up question or query based on the conversation history and previous legal assessment.
Conversation history:
{history_str}
Latest user message:
{query}

Generate a polite and clear English response. Do not include any notes, prefixes, or intros. Only return the response."""
                english_response = self._call_ollama_api(gen_prompt, temperature=0.3).strip()
                
                translation_prompt = f"""Translate the following legal assistant message to Malayalam.
Keep it conversational and simple. Do not add or remove content.
Use plain Malayalam, not legal jargon.
Source English: {english_response}
Output Malayalam only, no English."""
                malayalam_response = self._call_ollama_api(translation_prompt, temperature=0.0).strip()
                generated_text = f"{malayalam_response}\n\n{english_response}"
            elif expected_state == "CLARIFY":
                english_response = "Could you please clarify your question or describe your legal issue in more detail?"
                translation_prompt = f"""Translate the following legal assistant message to Malayalam.
Keep it conversational and simple. Do not add or remove content.
Use plain Malayalam, not legal jargon.
Source English: {english_response}
Output Malayalam only, no English."""
                malayalam_response = self._call_ollama_api(translation_prompt, temperature=0.0).strip()
                generated_text = f"{malayalam_response}\n\n{english_response}"
            else:
                generated_text = "Something went wrong. Please try again."

            shield_res = self._call_shield_validator(expected_state, session_slots, generated_text, retrieved_ids, last_assistant_response)
            score = shield_res.get("score", 1.0)
            logger.info(f"Shield Validator Attempt {attempt + 1}: score = {score}, expected_state = {expected_state}")
            
            if score >= 0.8:
                final_response_text = generated_text
                break
            else:
                logger.warning(f"Shield validation failed for attempt {attempt + 1}: {shield_res.get('fail_reasons')}")
                final_response_text = generated_text

        return {
            "status": final_status,
            "response_text": final_response_text,
            "faithfulness_score": final_faithfulness
        }

if __name__ == "__main__":
    pipeline = LegalMindPipeline()
    res = pipeline.run("എന്റെ വീട്ടുടമസ്ഥൻ എന്നോട് നാളെത്തന്നെ റൂമൊഴിഞ്ഞു തരാൻ പറഞ്ഞു.")
    print("Pipeline Output Status:", res["status"])
    print("Response:\n", res["response_text"])
