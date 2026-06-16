from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import logging

logger = logging.getLogger("LegalMind.VectorStore")

class VectorStore:
    def __init__(self, host=None, port=None):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        api_key = os.getenv("QDRANT_API_KEY")
        if self.host.startswith("http"):
            self.client = QdrantClient(url=self.host, api_key=api_key)
            self.port = None
        else:
            self.port = int(port or os.getenv("QDRANT_PORT", 6333))
            self.client = QdrantClient(host=self.host, port=self.port, api_key=api_key)
        self.collection_name = "legal_chunks"

    def init_collection(self, vector_size: int = 384):
        """
        Initializes the legal_chunks collection inside Qdrant.
        Configures COSINE similarity index and full-text indexes for payload search (BM25 fallback).
        """
        try:
            collections_resp = self.client.get_collections()
            exist = any(col.name == self.collection_name for col in collections_resp.collections)
            
            if not exist:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                # Create text index on the payload 'text' for BM25 keyword matching
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="text",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=15,
                        lowercase=True
                    )
                )
                logger.info(f"✓ Qdrant collection '{self.collection_name}' initialized.")
            else:
                logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise e

    def upsert_chunks(self, points: list):
        """
        Upsert a batch of document chunks.
        Expects a list of dicts: {"id": int/str, "vector": list[float], "payload": dict}
        """
        qdrant_points = []
        for pt in points:
            qdrant_points.append(
                models.PointStruct(
                    id=pt["id"],
                    vector=pt["vector"],
                    payload=pt["payload"]
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points
        )

    def hybrid_search(self, query_vector: list, query_text: str, top_k: int = 5, jurisdiction: str = None):
        """
        Performs hybrid search combining vector similarity and keyword search, filtering by jurisdiction.
        """
        query_filter = None
        if jurisdiction:
            jurisdiction_lower = jurisdiction.lower()
            query_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="jurisdiction",
                        match=models.MatchValue(value="central")
                    ),
                    models.FieldCondition(
                        key="jurisdiction",
                        match=models.MatchValue(value=jurisdiction_lower)
                    )
                ]
            )

        # Vector similarity search using query_points API
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        vector_results = response.points

        # Keyword lexical match search
        must_conditions = [
            models.FieldCondition(
                key="text",
                match=models.MatchText(text=query_text)
            )
        ]
        if query_filter:
            must_conditions.append(query_filter)

        lexical_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must_conditions),
            limit=top_k
        )[0]
        
        # Merge and deduplicate results
        seen_ids = set()
        merged_results = []
        
        # Add vector results first
        for res in vector_results:
            seen_ids.add(res.id)
            merged_results.append({
                "id": res.id,
                "score": res.score,
                "text": res.payload.get("text", ""),
                "citation": res.payload.get("citation", ""),
                "layer_depth": res.payload.get("layer_depth", 0),
                "section_id": res.payload.get("section_id"),  # <-- Add this line
                "type": "vector"
            })
            
        # Add lexical results if not already present
        for res in lexical_results:
            if res.id not in seen_ids:
                merged_results.append({
                    "id": res.id,
                    "score": 0.5,  # Baseline score for exact matches
                    "text": res.payload.get("text", ""),
                    "citation": res.payload.get("citation", ""),
                    "layer_depth": res.payload.get("layer_depth", 0),
                    "section_id": res.payload.get("section_id"),  # <-- Add this line
                    "type": "lexical"
                })

                
        return merged_results

if __name__ == "__main__":
    # Test init
    vs = VectorStore()
    vs.init_collection()
