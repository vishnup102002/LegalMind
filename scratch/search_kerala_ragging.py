import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host="localhost", port=6333)
collection_name = "legal_chunks"

res, offset = client.scroll(
    collection_name=collection_name,
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="text",
                match=models.MatchText(text="ragging")
            )
        ]
    ),
    limit=50,
    with_payload=True,
    with_vectors=False
)

print(f"Total matching points: {len(res)}")
for point in res:
    print(f"ID: {point.id}")
    print(f"Citation: {point.payload.get('citation')}")
    print(f"Jurisdiction: {point.payload.get('jurisdiction')}")
    print(f"Text snippet: {point.payload.get('text')[:200]}")
    print("-" * 50)
