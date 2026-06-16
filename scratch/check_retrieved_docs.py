import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from database.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

vs = VectorStore()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

query = "The user has been subjected to indecent conduct by a fellow student at an educational institution in Kerala on [date]."
query_vector = model.encode(query).tolist()

results = vs.hybrid_search(
    query_vector=query_vector,
    query_text=query,
    top_k=5,
    jurisdiction="Kerala"
)

print(f"Retrieved {len(results)} docs:")
for doc in results:
    point_id = doc["id"]
    point_info = vs.client.retrieve(
        collection_name="legal_chunks",
        ids=[point_id],
        with_payload=True,
        with_vectors=False
    )
    payload = point_info[0].payload if point_info else {}
    print(f"ID: {point_id}")
    print(f"Citation: {doc.get('citation')}")
    print(f"Jurisdiction: {payload.get('jurisdiction')}")
    print(f"Text snippet: {doc.get('text')[:200]}")
    print("-" * 50)
