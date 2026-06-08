from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import logging

logger = logging.getLogger("LegalMind.Pipeline")

# Define the State definition for the LangGraph pipeline
class LegalState(TypedDict):
    user_query: str
    extracted_intent: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    response_text: str
    faithfulness_score: float
    status: str  # SUCCESS or UNVERIFIED_LEGAL_GROUNDS

class LegalMindPipeline:
    def __init__(self, threshold: float = 0.72):
        self.threshold = threshold
        self._build_workflow()

    def extract_intent_node(self, state: LegalState) -> Dict[str, Any]:
        """Node 1: Extract intent from colloquial query. Integrates intent caches."""
        logger.info(f"Extracting intent for: '{state['user_query'][:50]}...'")
        
        # In production, query Redis cache first. If missed, execute Qwen-2.5-0.5B model.
        # Example cache match:
        query = state["user_query"].strip()
        
        # Stub intent lookup matching Malayalam sample colloquials
        if "റൂമൊഴിഞ്ഞു" in query or "ഒഴിപ്പിക്കാൻ" in query:
            intent = {
                "category": "lease_dispute",
                "subcategory": "unlawful_eviction",
                "locale": "kerala",
                "extracted_statutes": ["Section 14 of the State Rent Control Act"]
            }
        else:
            intent = {
                "category": "general",
                "subcategory": "legal_advice",
                "locale": "kerala",
                "extracted_statutes": []
            }
            
        logger.info(f"Extracted Intent: {intent}")
        return {"extracted_intent": intent}

    def retrieve_context_node(self, state: LegalState) -> Dict[str, Any]:
        """Node 2: Query Qdrant vector store and Neo4j graph using extracted intent."""
        intent = state["extracted_intent"]
        logger.info(f"Retrieving context for category: {intent['category']}")
        
        # In production, execute Neo4j Graph queries and Qdrant hybrid searches
        retrieved = []
        if intent["category"] == "lease_dispute":
            retrieved = [
                {
                    "id": "sec_14_rent_act",
                    "text": "Kerala Rent Control Act Section 14 mandates a 30-day written notice for eviction. Tenant cannot be dispossessed without due cause.",
                    "citation": "Section 14 Rent Control Act",
                    "layer_depth": 0
                },
                {
                    "id": "appeal_rules",
                    "text": "Any tenant evicted without compliance with Section 14 can petition the Rent Control Appellate Authority.",
                    "citation": "Section 18 Rent Control Act",
                    "layer_depth": 1
                }
            ]
        else:
            retrieved = [
                {
                    "id": "general_guide",
                    "text": "General legal aid guidelines for state dispute resolution.",
                    "citation": "Article 39A Constitution of India",
                    "layer_depth": 0
                }
            ]
            
        logger.info(f"Retrieved {len(retrieved)} document chunks.")
        return {"retrieved_docs": retrieved}

    def filter_rerank_node(self, state: LegalState) -> Dict[str, Any]:
        """Node 3: Context filtering using BGE-Reranker-Large."""
        logger.info("Reranking retrieved passages...")
        # In production, score retrieved_docs with BGE-Reranker-Large cross-encoder
        # For simulation, we preserve order but assign mock scores
        reranked = []
        for idx, doc in enumerate(state["retrieved_docs"]):
            # Mock high relevance score
            doc_copy = doc.copy()
            doc_copy["relevance_score"] = 0.85 - (idx * 0.1)
            reranked.append(doc_copy)
            
        return {"reranked_docs": reranked}

    def citation_shield_gate(self, state: LegalState) -> Dict[str, Any]:
        """Node 4: Validate citations and calculate faithfulness metrics."""
        logger.info("Evaluating context faithfulness shield...")
        
        # In production, check generated claims against Neo4j indexes and compute NLI scores
        # We simulate this checking step. If correct elements exist in reranked docs, score is high.
        has_correct_citation = any(
            "Section 14" in doc.get("citation", "")
            for doc in state["reranked_docs"]
        )
        
        # Force a fail if query contains unresolvable topics for testing
        if state["user_query"] == "malicious_hack_query":
            score = 0.45
        else:
            score = 0.88 if has_correct_citation else 0.60
            
        logger.info(f"Attributed Faithfulness Score: {score}")
        
        if score >= self.threshold:
            return {"faithfulness_score": score, "status": "SUCCESS"}
        else:
            return {"faithfulness_score": score, "status": "UNVERIFIED_LEGAL_GROUNDS"}

    def generate_roadmap_node(self, state: LegalState) -> Dict[str, Any]:
        """Node 5: LLM Generation step using the local vLLM reasoning profile."""
        if state["status"] == "UNVERIFIED_LEGAL_GROUNDS":
            rejection_msg = (
                "STATUS: UNVERIFIED_LEGAL_GROUNDS\n"
                "The legal information required to answer your query could not be verified "
                "against authenticated statutory sources. To prevent structural risk and incorrect "
                "guidance, the request has been short-circuited."
            )
            return {"response_text": rejection_msg}
            
        logger.info("Generating verified IRAC roadmap...")
        
        # Build prompt using state['reranked_docs'] and run vLLM
        # Mocking an IRAC response
        docs_context = "\n".join([d["text"] for d in state["reranked_docs"]])
        roadmap = (
            "LEGAL ROADMAP (IRAC FORMAT):\n"
            "ISSUE: Whether the verbal eviction demand requires immediate compliance.\n"
            "RULE: Under Section 14 of the State Rent Control Act, all eviction notices must be in writing with 30 days notice.\n"
            "APPLICATION: The landlord's immediate verbal demand violates Section 14 notice protocols.\n"
            "CONCLUSION: You are not legally obligated to vacate. You should demand a formal written notice."
        )
        return {"response_text": roadmap}

    def _build_workflow(self):
        workflow = StateGraph(LegalState)
        
        # Add Nodes
        workflow.add_node("extract_intent", self.extract_intent_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("filter_rerank", self.filter_rerank_node)
        workflow.add_node("citation_shield", self.citation_shield_gate)
        workflow.add_node("generate_roadmap", self.generate_roadmap_node)
        
        # Set Edges
        workflow.set_entry_point("extract_intent")
        workflow.add_edge("extract_intent", "retrieve_context")
        workflow.add_edge("retrieve_context", "filter_rerank")
        workflow.add_edge("filter_rerank", "citation_shield")
        workflow.add_edge("citation_shield", "generate_roadmap")
        workflow.add_edge("generate_roadmap", END)
        
        self.app = workflow.compile()

    def run(self, query: str) -> Dict[str, Any]:
        initial_state = {
            "user_query": query,
            "extracted_intent": {},
            "retrieved_docs": [],
            "reranked_docs": [],
            "response_text": "",
            "faithfulness_score": 0.0,
            "status": "SUCCESS"
        }
        return self.app.invoke(initial_state)

if __name__ == "__main__":
    pipeline = LegalMindPipeline()
    res = pipeline.run("എന്റെ വീട്ടുടമസ്ഥൻ എന്നോട് നാളെത്തന്നെ റൂമൊഴിഞ്ഞു തരാൻ പറഞ്ഞു.")
    print("Pipeline Output Status:", res["status"])
    print("Response:\n", res["response_text"])
