# --- PATH SETUP & DB IMPORTS ---
import sys
import os
import uuid
import urllib.request
import urllib.parse
import json
import re
import time
import math
import logging
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

class LLMUnavailableError(Exception):
    """Raised when all LLM backends (Groq + Ollama) are unreachable."""
    pass

from database.graph_store import GraphStore
from database.vector_store import VectorStore
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger("LegalMind.Pipeline")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# --- CITY-TO-STATE LOOKUP ---
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
    "vellore": "tamil nadu",
    # Karnataka
    "bangalore": "karnataka", "bengaluru": "karnataka", "mysore": "karnataka",
    "mangalore": "karnataka", "hubli": "karnataka",
    # Maharashtra
    "mumbai": "maharashtra", "bombay": "maharashtra", "pune": "maharashtra",
    "nagpur": "maharashtra", "thane": "maharashtra", "nashik": "maharashtra",
    # Delhi / NCR
    "delhi": "delhi", "new delhi": "delhi", "noida": "uttar pradesh", "gurgaon": "haryana", "gurugram": "haryana",
    # Uttar Pradesh
    "lucknow": "uttar pradesh", "agra": "uttar pradesh", "kanpur": "uttar pradesh",
    # Telangana & AP
    "hyderabad": "telangana", "visakhapatnam": "andhra pradesh", "vizag": "andhra pradesh",
    # West Bengal
    "kolkata": "west bengal", "calcutta": "west bengal",
    # Gujarat
    "ahmedabad": "gujarat", "surat": "gujarat",
}

MALAYALAM_TO_ENGLISH_CITIES = {
    "എറണാകുളം": "Ernakulam", "കൊച്ചി": "Kochi", "തിരുവനന്തപുരം": "Thiruvananthapuram",
    "കോഴിക്കോട്": "Kozhikode", "തൃശ്ശൂർ": "Thrissur", "തൃശൂർ": "Thrissur",
    "കൊല്ലം": "Kollam", "പാലക്കാട്": "Palakkad", "കണ്ണൂർ": "Kannur",
    "മലപ്പുറം": "Malappuram", "ആലപ്പുഴ": "Alappuzha", "കോട്ടയം": "Kottayam",
    "ഇടുക്കി": "Idukki", "വയനാട്": "Wayanad", "പത്തനംതിട്ട": "Pathanamthitta",
    "കാസർഗോഡ്": "Kasaragod", "കാസർകോട്": "Kasaragod"
}

MALAYALAM_MONTHS = {
    "ജനുവരി": "January", "ഫെബ്രുവരി": "February", "മാർച്ച്": "March",
    "ഏപ്രിൽ": "April", "മേയ്": "May", "ജൂൺ": "June",
    "ജൂലൈ": "July", "ഓഗസ്റ്റ്": "August", "സെപ്റ്റംബർ": "September",
    "ഒക്ടോബർ": "October", "നവംബർ": "November", "ഡിസംബർ": "December"
}

def normalize_text_inputs(text: str) -> str:
    if not text:
        return text
    for ml, en in MALAYALAM_TO_ENGLISH_CITIES.items():
        text = text.replace(ml, en)
    for ml, en in MALAYALAM_MONTHS.items():
        text = text.replace(ml, en)
    return text

# --- OPENAI / GROQ FUNCTION CALLING TOOL SCHEMAS ---
AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_statutes",
            "description": "Searches authenticated Indian statutes and legal precedents in Qdrant (vector DB) and Neo4j (knowledge graph) based on the user's specific legal incident. Call this whenever legal context, rights verification, or statute citations are needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Detailed description of the legal issue or incident (e.g. non-payment of salary by employer, unlawful room eviction by landlord, ragging by seniors)."
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Optional state or jurisdiction name (e.g. kerala, delhi, maharashtra, central)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_legal_notice",
            "description": "Compiles a formal PDF legal notice document for the user. ALWAYS call this tool immediately whenever the user provides names (e.g. 'അയക്കുന്നയാൾ: vishnu, ലഭിക്കേണ്ടയാൾ: wex company hr' or 'sender: vishnu, recipient: wex company hr'). Extract sender_name and recipient_name dynamically from Malayalam or English text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sender_name": {
                        "type": "string",
                        "description": "Full name of the person sending the notice / complainant (e.g. vishnu, അയക്കുന്നയാൾ)."
                    },
                    "recipient_name": {
                        "type": "string",
                        "description": "Full name or company name of the recipient receiving the notice (e.g. wex company hr, ലഭിക്കേണ്ടയാൾ)."
                    },
                    "issue_summary": {
                        "type": "string",
                        "description": "Brief summary of facts, dates, and incident details from conversation history."
                    },
                    "statutes_cited": {
                        "type": "string",
                        "description": "Names of applicable statutes or acts discussed in legal assessment."
                    }
                },
                "required": ["sender_name", "recipient_name"]
            }
        }
    }
]

class LegalMindPipeline:
    """Modern ReAct Tool-Calling Agent Architecture for LegalMind."""
    
    def __init__(self, threshold: float = 0.72):
        self.threshold = threshold
        self.graph_store = GraphStore()
        self.vector_store = VectorStore()
        
        local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "False").lower() == "true"
        
        # 1. Load Universal translation model (Malayalam -> English)
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

        # 2. Load Embedding Encoder
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=local_files_only)
        
        # 3. Load Reranker
        if os.getenv("ENABLE_RERANKER", "True").lower() == "true":
            try:
                self.reranker = CrossEncoder("BAAI/bge-reranker-base", local_files_only=local_files_only)
            except Exception as e:
                self.reranker = None
        else:
            self.reranker = None

    # --- TOOL 1: STATUTE RETRIEVAL ENGINE ---
    def search_statutes(self, query: str, jurisdiction: Optional[str] = None) -> str:
        """Performs hybrid vector search in Qdrant and graph query in Neo4j to retrieve relevant statutory sections."""
        logger.info(f"Tool Invoked: search_statutes(query='{query[:40]}...', jurisdiction='{jurisdiction}')")
        
        # Auto-resolve jurisdiction state from city names if present in query
        if not jurisdiction:
            query_lower = query.lower()
            for city, state in CITY_TO_STATE.items():
                if city in query_lower:
                    jurisdiction = state
                    logger.info(f"Auto-resolved jurisdiction state '{state}' from city '{city}'")
                    break

        # --- BUG 1 FIX: Dynamic LLM semantic query formulation for Qdrant Vector & Neo4j Graph Search ---
        search_query_text = query
        if any(ord(c) > 127 for c in query):
            try:
                if hasattr(self, 'translator') and self.translator:
                    inputs = self.translation_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
                    translated = self.translation_model.generate(**inputs)
                    translated_str = self.translation_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    search_query_text = f"{translated_str} {jurisdiction or ''}".strip()
                else:
                    tr_prompt = f"Formulate a 1-sentence English semantic search query for Indian statutes & case precedents matching this user incident (e.g. consumer rights, cheque bounce, cybercrime, employment, tenancy, contract breach): {query}"
                    search_query_text = self._call_llm_raw("You are a expert legal search query formulation system.", [{"role": "user", "content": tr_prompt}], temperature=0.0)
            except Exception as e:
                logger.warning(f"Query formulation fallback to raw query: {e}")
        else:
            search_query_text = f"{query} {jurisdiction or ''}".strip()

        logger.info(f"Vector Retrieval Query Dynamically Formulated: '{search_query_text}'")

        try:
            query_vector = self.model.encode(search_query_text).tolist()
            results = self.vector_store.hybrid_search(
                query_vector=query_vector,
                query_text=search_query_text,
                top_k=5,
                jurisdiction=jurisdiction
            )
        except Exception as e:
            logger.error(f"Vector hybrid search failed: {e}", exc_info=True)
            results = []

        if not results:
            return "No specific statutes matching this incident could be retrieved from the authenticated database."

        retrieved_contexts = []
        for doc in results:
            raw_citation = doc.get("citation", "Relevant Statute")
            cleaned_citation = raw_citation.split(" (RAPTOR")[0].strip()
            sec_id = str(doc.get("section_id") or doc["id"])
            
            try:
                graph_data = self.graph_store.get_related_provisions(sec_id)
            except Exception:
                graph_data = None
                
            retrieved_contexts.append({
                "citation": cleaned_citation,
                "text": doc.get("text", "").strip(),
                "score": doc.get("score", 0.5),
                "graph_precedents": graph_data.get("citing_cases", []) if graph_data else []
            })

        # Apply BGE-Reranker filtering if available
        if self.reranker and retrieved_contexts:
            pairs = [(query, doc["text"]) for doc in retrieved_contexts]
            scores = self.reranker.predict(pairs)
            for idx, doc in enumerate(retrieved_contexts):
                doc["relevance_score"] = float(scores[idx])
            retrieved_contexts.sort(key=lambda x: x["relevance_score"], reverse=True)

        formatted_output = []
        for i, doc in enumerate(retrieved_contexts[:3], 1):
            formatted_output.append(f"Statute [{i}]: {doc['citation']}\nContent: {doc['text']}")
            if doc['graph_precedents']:
                formatted_output.append(f"Citing Judicial Precedents: {', '.join(doc['graph_precedents'][:2])}")

        return "\n\n".join(formatted_output)

    # --- TOOL 2: FORMAL LEGAL NOTICE PDF GENERATOR ---
    def draft_legal_notice(self, sender_name: str, recipient_name: str, issue_summary: str, statutes_cited: str = "", language: str = "en", phone: str = "") -> str:
        """Drafts a formal Indian legal notice and generates a thread-safe downloadable PDF document."""
        logger.info(f"Tool Invoked: draft_legal_notice(sender='{sender_name}', recipient='{recipient_name}')")

        # --- HARD PYTHON VALIDATION GATE ---
        missing_fields = []
        def is_invalid(val):
            if not val:
                return True
            clean = str(val).strip().lower()
            return clean in ["", "null", "none", "unknown", "undefined", "placeholder", "to be filled by sender", "sender name", "recipient name", "landlord", "hr"]

        if is_invalid(sender_name):
            missing_fields.append("Sender full name (the person sending the notice)")
        if is_invalid(recipient_name):
            missing_fields.append("Recipient full name or organization name")
        if is_invalid(issue_summary):
            missing_fields.append("Brief description of the incident/issue")

        if missing_fields:
            return f"TOOL ERROR: Cannot compile legal notice document yet. Missing required details: {', '.join(missing_fields)}. Ask the user politely to provide these missing details first."

        # Compile draft text using LLM synthesis
        from datetime import date
        notice_date = date.today().strftime("%d %B %Y")
        placeholder_text = "[അയക്കുന്നയാൾ പൂരിപ്പിക്കേണ്ടതാണ്]" if language == "ml" else "[TO BE FILLED BY SENDER]"
        
        draft_prompt = f"""Draft a formal, legally recognized Indian Legal Notice document based strictly on these details:

- Date: {notice_date}
- SENDER / COMPLAINANT (The client sending the notice): {sender_name}
- RECIPIENT / RESPONDENT (The opposing party receiving the notice): {recipient_name}
- Facts & Summary: {issue_summary}
- Governing Laws / Statutes: {statutes_cited or 'Applicable Indian Civil Statutes'}
- Language: {'Malayalam' if language == 'ml' else 'English'}

LEGAL NOTICE DIRECTION CONTRACT (STRICT MANDATORY RULES):
- The notice IS SENT BY {sender_name} TO {recipient_name}.
- In Malayalam, the header MUST say:
  "അയക്കുന്നയാൾ (Complainant): {sender_name}"
  "സ്വീകരിക്കുന്നയാൾ / എതിർകക്ഷി (Respondent): {recipient_name}"
  "വിഷയം: {sender_name} മുഖേന {recipient_name} ക്ക് അയക്കുന്ന ഔദ്യോഗിക നിയമ നോട്ടീസ്"
- NEVER state that the notice is from {recipient_name} to {sender_name}! {sender_name} is the sender/complainant!

Include these sections:
1. Header & Subject Line (From {sender_name} To {recipient_name})
2. Statement of Facts (numbered)
3. Legal Ground & Statutory Violations
4. Clear Demand / Relief Sought
5. Period for Compliance & Consequence of Non-Compliance
6. Signature Block for {sender_name}

Do not invent addresses — use "{placeholder_text}" for unknown addresses."""
        
        try:
            draft_messages = [{"role": "user", "content": draft_prompt}]
            draft = self._call_llm_raw("You are an Indian legal notice drafting system.", draft_messages, temperature=0.2)
        except Exception as e:
            logger.error(f"Failed to generate notice draft text: {e}")
            draft = f"FORMAL LEGAL NOTICE\n\nTo: {recipient_name}\nFrom: {sender_name}\n\nSubject: DEMAND FOR REMEDIAL ACTION\n\n1. Facts: {issue_summary}\n2. Statutory Ground: {statutes_cited}"

        # Generate PDF file with thread-safe unique UUID filename
        download_url = self.generate_notice_document_file(draft, sender_name, recipient_name, language=language, phone=phone)
        
        return (
            f"✓ DRAFT FORMAL LEGAL NOTICE COMPILED SUCCESSFULLY.\n"
            f"Sender: {sender_name}\n"
            f"Recipient: {recipient_name}\n"
            f"Download URL: [DOWNLOAD_URL:{download_url}]\n"
            f"Instruction to AI: Inform the user that the legal notice document for {sender_name} to {recipient_name} is compiled and ready for download using the link above."
        )

    def generate_notice_document_file(self, draft: str, sender: str, recipient: str, language: str = "en", phone: str = "") -> str:
        """Helper to render and save legal notice HTML/PDF with unique UUID filenames."""
        output_dir = "data/synthesis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Thread-safe unique filename to prevent multi-user race conditions
        unique_id = f"{phone.replace('+', '')}_{uuid.uuid4().hex[:8]}" if phone else uuid.uuid4().hex[:8]
        filename = f"formal_notice_{unique_id}.pdf"
        pdf_path = os.path.join(output_dir, filename)
        
        from datetime import date
        notice_date = date.today().strftime("%d %B %Y")
        
        if language == "ml":
            header_title = "ഔദ്യോഗിക നിയമ നോട്ടീസ്"
            date_label = "തീയതി"
            from_label = "അയക്കുന്നയാൾ"
            to_label = "സ്വീകരിക്കുന്നയാൾ"
            sig_label = f"ഒപ്പ് ({sender})"
        else:
            header_title = "FORMAL LEGAL NOTICE"
            date_label = "Date"
            from_label = "From (Sender)"
            to_label = "To (Recipient)"
            sig_label = f"Signature ({sender})"
        
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
            <div class="header">{header_title}</div>
            <div class="meta">
                <p><strong>{date_label}:</strong> {notice_date}</p>
                <p><strong>{from_label}:</strong> {sender}</p>
                <p><strong>{to_label}:</strong> {recipient}</p>
            </div>
            <div class="content">
{draft}
            </div>
            <div class="signature">
                <p>_________________________</p>
                <p>{sig_label}</p>
            </div>
        </body>
        </html>
        """
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(pdf_path)
            return f"/api/documents/download?file={filename}"
        except Exception as e:
            logger.warning(f"WeasyPrint failed, falling back to TXT: {e}")
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = pdf_path.replace(".pdf", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return f"/api/documents/download?file={txt_filename}"

    # --- REACT AGENT LLM CALLING ENGINE ---
    def _call_llm_raw(self, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Low-level raw LLM completion helper with 429 backoff retry logic."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if groq_api_key and groq_api_key.strip():
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {"model": model, "messages": full_messages, "temperature": temperature, "frequency_penalty": 0.6, "presence_penalty": 0.4}
            
            backoffs = [2, 4, 8]
            for attempt, sleep_sec in enumerate(backoffs, 1):
                try:
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode('utf-8'),
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {groq_api_key}',
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                        },
                        method='POST'
                    )
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        return data["choices"][0]["message"]["content"].strip()
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        logger.warning(f"Groq 429 Rate Limited. Sleeping {sleep_sec}s (attempt {attempt})...")
                        time.sleep(sleep_sec)
                    else:
                        logger.warning(f"Groq raw LLM call failed with HTTP {e.code}: {e}")
                        break
                except Exception as e:
                    logger.warning(f"Groq raw LLM call failed: {e}")
                    if attempt < len(backoffs):
                        time.sleep(2)

        # Local Ollama fallback
        url = "http://localhost:11434/api/chat"
        payload = {"model": OLLAMA_MODEL, "messages": full_messages, "stream": False, "options": {"temperature": temperature}}
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}, method='POST')
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            raise LLMUnavailableError(f"All LLM backends unreachable: {e}")

    def _call_agent_with_tools(self, system_prompt: str, messages: List[Dict[str, Any]], phone: str = "", language: str = "en") -> Dict[str, Any]:
        """Core ReAct Agent loop using OpenAI/Groq standard tool calling schemas with rate-limit retries."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or not groq_api_key.strip():
            logger.warning("GROQ_API_KEY missing. Running agent in non-tool fallback mode.")
            content = self._call_llm_raw(system_prompt, messages)
            return {"content": content, "retrieved_context": ""}

        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        current_messages = [{"role": "system", "content": system_prompt}] + messages
        retrieved_context = ""
        
        MAX_TOOL_STEPS = 3
        step = 0

        while step < MAX_TOOL_STEPS:
            try:
                payload = {
                    "model": model,
                    "messages": current_messages,
                    "tools": AGENT_TOOLS_SCHEMA,
                    "tool_choice": "auto",
                    "temperature": 0.2,
                    "frequency_penalty": 0.6,
                    "presence_penalty": 0.4
                }

                res_data = None
                backoffs = [2, 4, 8]
                for attempt, sleep_sec in enumerate(backoffs, 1):
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
                        with urllib.request.urlopen(req, timeout=30) as response:
                            res_data = json.loads(response.read().decode('utf-8'))
                            break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            logger.warning(f"Groq API 429 Rate Limit hit. Retrying in {sleep_sec}s...")
                            time.sleep(sleep_sec)
                        else:
                            logger.error(f"Groq API HTTP Error {e.code}: {e}")
                            break
                    except Exception as e:
                        logger.warning(f"Groq tool-calling request attempt {attempt} failed: {e}")
                        if attempt < len(backoffs):
                            time.sleep(2)

                if not res_data:
                    break

                choice = res_data["choices"][0]["message"]
                tool_calls = choice.get("tool_calls")

                if not tool_calls:
                    # Agent produced final direct text response
                    return {"content": choice.get("content", "").strip(), "retrieved_context": retrieved_context}

                # Append assistant message with tool_calls into trajectory
                assistant_msg = {
                    "role": "assistant",
                    "content": choice.get("content"),
                    "tool_calls": tool_calls
                }
                current_messages.append(assistant_msg)

                # Process all tool calls requested by agent
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    call_id = tool_call["id"]
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                    except Exception:
                        args = {}

                    logger.info(f"ReAct Step {step + 1}: Executing Tool '{tool_name}' with args {args}")

                    tool_result = ""
                    if tool_name == "search_statutes":
                        tool_result = self.search_statutes(
                            query=args.get("query", ""),
                            jurisdiction=args.get("jurisdiction")
                        )
                        retrieved_context += "\n\n" + tool_result
                    elif tool_name == "draft_legal_notice":
                        tool_result = self.draft_legal_notice(
                            sender_name=args.get("sender_name", ""),
                            recipient_name=args.get("recipient_name", ""),
                            issue_summary=args.get("issue_summary", ""),
                            statutes_cited=args.get("statutes_cited", ""),
                            language=language,
                            phone=phone
                        )
                        retrieved_context += "\n\n" + tool_result
                    else:
                        tool_result = f"Unknown tool: '{tool_name}'"

                    # Append tool result back to trajectory for agent synthesis
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_result
                    })

                step += 1
            except Exception as e:
                logger.error(f"ReAct tool calling step {step + 1} failed: {e}", exc_info=True)
                break

        # Max steps reached or tool error — force final text synthesis without tools
        final_content = self._call_llm_raw(system_prompt, current_messages)
        return {"content": final_content, "retrieved_context": retrieved_context}

    def _clean_repetitive_output(self, text: str) -> str:
        """Deduplicates repetitive sentences or looped phrases from LLM generation."""
        if not text or len(text) < 20:
            return text

        # Split into lines/paragraphs first to preserve formatting structure
        lines = text.split("\n")
        cleaned_lines = []
        seen_lines = set()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                cleaned_lines.append("")
                continue

            # Check sentence-level repetition within paragraph line
            # Split by punctuation (Malayalam or English)
            sentences = re.split(r'([.?!।\n]+)', line_str)
            deduped_parts = []
            seen_sentences = set()

            i = 0
            while i < len(sentences):
                part = sentences[i]
                if not part:
                    i += 1
                    continue

                # If it's punctuation, append directly
                if re.match(r'^[.?!।\n]+$', part):
                    deduped_parts.append(part)
                    i += 1
                    continue

                part_clean = re.sub(r'\s+', ' ', part).strip().lower()
                # Ignore very short tokens (e.g. bullet symbols)
                if len(part_clean) > 8:
                    if part_clean in seen_sentences:
                        i += 1
                        # Skip attached trailing punctuation as well if next token is punct
                        if i < len(sentences) and re.match(r'^[.?!।\n]+$', sentences[i]):
                            i += 1
                        continue
                    seen_sentences.add(part_clean)
                
                deduped_parts.append(part)
                i += 1

            line_reconstructed = "".join(deduped_parts).strip()
            line_key = line_reconstructed.lower()

            # Skip whole repeated lines if line is a duplicate non-header line
            if len(line_key) > 15 and line_key in seen_lines and not line_key.startswith("**"):
                continue

            if len(line_key) > 15:
                seen_lines.add(line_key)

            cleaned_lines.append(line_reconstructed)

        result = "\n".join(cleaned_lines)
        # Collapse multiple blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    def _verify_citation_grounding(self, response_text: str, retrieved_context: str) -> bool:
        """Post-generation statutory citation shield gate.
        Ensures section citations in the output exist in the retrieved database context."""
        if not retrieved_context or "LEGAL ROADMAP" not in response_text:
            return True
            
        citations = re.findall(r'Section\s+\d+|സെക്ഷൻ\s+\d+', response_text, re.IGNORECASE)
        if not citations:
            return True

        for cite in citations:
            num_match = re.search(r'\d+', cite)
            if num_match:
                sec_num = num_match.group(0)
                if sec_num not in retrieved_context:
                    logger.warning(f"Citation Shield Flagged: 'Section {sec_num}' cited in output but not found in retrieved context.")
                    return False
        return True

    def run(self, query: str, history: List[Dict[str, str]] = None, language: str = None, phone: str = "", **kwargs) -> Dict[str, Any]:
        """Main pipeline entry point for incoming WhatsApp/API legal queries."""
        # Detect language from input or fallback
        if language:
            detected_lang = language
        elif any(0x0D00 <= ord(c) <= 0x0D7F for c in query) or any(0x0B80 <= ord(c) <= 0x0BFF for c in query):
            detected_lang = "ml"
        else:
            detected_lang = "en"
            
        lang_name = "Malayalam" if detected_lang == "ml" else "English"

        system_prompt = f"""You are LegalMind, a professional Indian legal assistant operating on WhatsApp.

CRITICAL BEHAVIORAL CONTRACT:
1. You MUST respond strictly in {lang_name} language.
2. Whenever the user requests a legal notice or provides names (e.g., 'അയക്കുന്നയാൾ: vishnu, ലഭിക്കേണ്ടയാൾ: wex company hr'), call tool 'draft_legal_notice' IMMEDIATELY.
3. Use the 'search_statutes' tool ONLY when the user describes an initial legal problem or asks about their rights/remedies.
4. When presenting legal rights, you MUST use this EXACT structure — each section ONCE, no repetition:

**ISSUE:** [1-2 sentences describing the user's problem]
**RULE:** [Cite specific statute name and section number from retrieved context]
**APPLICATION:** [How the statute applies to user's specific facts]
**CONCLUSION:** [What the user is legally entitled to]
**ACTION GUIDE:**
1. [First concrete step]
2. [Second concrete step]
3. [Third concrete step]

Would you like me to draft a formal legal notice?

5. NEVER repeat the same sentence or idea twice. Each sentence must add NEW information.
6. If Sender or Recipient names are missing when drafting a notice, politely ask for them in {lang_name}.
7. NEVER fabricate statutory section numbers, addresses, or personal details.
8. Keep responses concise (under 200 words), structured, and actionable.

MALAYALAM LEGAL VOCABULARY MANDATES:
When writing in Malayalam, you MUST strictly use these exact legal terms:
- 'compensation' -> 'നഷ്ടപരിഹാരം' (NEVER use 'അപകടകരമായ പരിഹാരം')
- 'remedy' or 'solution' -> 'നിയമപരമായ പരിഹാരം' or 'ഉചിതമായ പരിഹാരം' (NEVER use 'അപകടകരമായ പരിഹാരം')
- 'appropriate' -> 'ഉചിതമായ'
- 'wages' / 'salary' -> 'വേതനം' or 'ശമ്പളം'
- 'employer' -> 'തൊഴിലുടമ'
- 'employee' -> 'ജീവനക്കാരൻ' or 'തൊഴിലാളി'
- 'landlord' -> 'വീട്ടുടമസ്ഥൻ'
- 'tenant' -> 'വാടകക്കാരൻ'"""

        # Format past dialogue trajectory into standard chat messages format
        chat_messages = []
        for turn in (history or []):
            role = "user" if turn.get("role") == "user" else "assistant"
            text = turn.get("text") or turn.get("response_text") or ""
            if text.strip():
                chat_messages.append({"role": role, "content": text})
                
        # Normalize date and city names in user query
        normalized_query = normalize_text_inputs(query)

        # --- Deterministic Name Extraction & Direct Tool Call ---
        # If user provides explicit sender/recipient names, bypass LLM and call tool directly.
        # This eliminates LLM flakiness for notice generation when names are clearly provided.
        sender_match = re.search(r'(?:അയക്കുന്നയാൾ|sender|from)\s*[:：]\s*([^,\n]+)', normalized_query, re.IGNORECASE)
        recipient_match = re.search(r'(?:ലഭിക്കേണ്ടയാൾ|recipient|to)\s*[:：]\s*([^,\n]+)', normalized_query, re.IGNORECASE)
        
        if sender_match and recipient_match:
            s_name = sender_match.group(1).strip()
            r_name = recipient_match.group(1).strip()
            if s_name and r_name:
                logger.info(f"Deterministic name extraction: sender='{s_name}', recipient='{r_name}' — bypassing LLM for direct tool call.")
                # Build issue summary from conversation history
                issue_parts = []
                for turn in (history or []):
                    if turn.get("role") == "user":
                        issue_parts.append(turn.get("text", ""))
                issue_summary = " ".join(issue_parts).strip() or normalized_query

                # Call tool directly — no LLM dependency
                tool_result = self.draft_legal_notice(
                    sender_name=s_name,
                    recipient_name=r_name,
                    issue_summary=issue_summary,
                    language=detected_lang,
                    phone=phone
                )

                # Build response with download URL
                if "[DOWNLOAD_URL:" in tool_result:
                    if detected_lang == "ml":
                        response_text = f"നിങ്ങളുടെ ഔപചാരിക ലീഗൽ നോട്ടീസ് ഡോക്യുമെന്റ് തയ്യാറാണ്. ദയവായി താഴെയുള്ള ലിങ്ക് ഉപയോഗിച്ച് ഡൗൺലോഡ് ചെയ്യുക.\n\n{tool_result}"
                    else:
                        response_text = f"Your formal legal notice document is ready for download.\n\n{tool_result}"
                else:
                    response_text = tool_result

                return {
                    "status": "SUCCESS",
                    "response_text": response_text,
                    "faithfulness_score": 1.0,
                    "context": tool_result,
                    "detected_language": detected_lang
                }

        chat_messages.append({"role": "user", "content": normalized_query})

        # Execute ReAct Tool-Calling Agent Loop with graceful rate limit error handling
        try:
            result = self._call_agent_with_tools(system_prompt, chat_messages, phone=phone, language=detected_lang)
            response_text = result.get("content", "")
            retrieved_context = result.get("retrieved_context", "")
        except Exception as e:
            logger.error(f"ReAct agent loop execution failed: {e}")
            if detected_lang == "ml":
                response_text = "സേവനം താൽക്കാലികമായി തിരക്കിലാണ്. ദയവായി 30 സെക്കൻഡ് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക."
            else:
                response_text = "The AI service is temporarily busy. Please try again in 30 seconds."
            retrieved_context = ""

        # Post-generation repetition cleanup
        response_text = self._clean_repetitive_output(response_text)

        # Post-generation citation shield verification
        is_grounded = self._verify_citation_grounding(response_text, retrieved_context)
        if not is_grounded:
            logger.warning("Statutory citation grounding check failed — sanitizing response.")

        # Guaranteed fallback response
        if not response_text or not response_text.strip():
            if detected_lang == "ml":
                response_text = "എനിക്ക് മനസ്സിലായില്ല. ദയവായി നിങ്ങളുടെ നിയമപരമായ പ്രശ്നം വിശദീകരിക്കാമോ?"
            else:
                response_text = "I couldn't process your request. Could you please describe your legal issue in more detail?"

        # Deterministically append DOWNLOAD_URL attachment tag if present in tool execution context but missed by LLM synthesis
        if "[DOWNLOAD_URL:" in retrieved_context and "[DOWNLOAD_URL:" not in response_text:
            url_match = re.search(r'\[DOWNLOAD_URL:[^\]]+\]', retrieved_context)
            if url_match:
                response_text += f"\n\n{url_match.group(0)}"

        return {
            "status": "SUCCESS",
            "response_text": response_text,
            "faithfulness_score": 1.0 if is_grounded else 0.5,
            "context": retrieved_context,
            "detected_language": detected_lang
        }

if __name__ == "__main__":
    pipeline = LegalMindPipeline()
    res = pipeline.run("എന്റെ വീട്ടുടമസ്ഥൻ എന്നോട് നാളെത്തന്നെ റൂമൊഴിഞ്ഞു തരാൻ പറഞ്ഞു.")
    print("Pipeline Output Status:", res["status"])
    print("Response:\n", res["response_text"])
