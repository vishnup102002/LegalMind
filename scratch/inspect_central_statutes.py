from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

client = QdrantClient(host="localhost", port=6333)
collection_name = "legal_chunks"

# Scroll all points with jurisdiction "central"
offset = None
central_statutes = set()
while True:
    res, offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="jurisdiction",
                    match=models.MatchValue(value="central")
                )
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
        offset=offset
    )
    for point in res:
        citation = point.payload.get("citation", "")
        central_statutes.add(citation)
    if offset is None:
        break

print(f"Total unique central statutes: {len(central_statutes)}")
for s in sorted(list(central_statutes))[:100]:
    print(s)
