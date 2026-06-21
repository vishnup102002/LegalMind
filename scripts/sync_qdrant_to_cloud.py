import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import time

load_dotenv()

# Connect to local Qdrant
local_client = QdrantClient(host="localhost", port=6333)

# Connect to cloud Qdrant
cloud_host = os.getenv("QDRANT_HOST")
cloud_api_key = os.getenv("QDRANT_API_KEY")
if not cloud_host or not cloud_api_key:
    print("Error: QDRANT_HOST or QDRANT_API_KEY not found in .env")
    sys.exit(1)

print(f"Connecting to Cloud Qdrant at: {cloud_host}")
cloud_client = QdrantClient(url=cloud_host, api_key=cloud_api_key)

collection_name = "legal_chunks"

# 1. Initialize cloud collection if needed
try:
    collections_resp = cloud_client.get_collections()
    exist = any(col.name == collection_name for col in collections_resp.collections)
    if not exist:
        print(f"Creating cloud collection '{collection_name}'...")
        cloud_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
        cloud_client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=15,
                lowercase=True
            )
        )
        print("Cloud collection initialized.")
    else:
        print(f"Cloud collection '{collection_name}' already exists.")
except Exception as e:
    print("Error checking/creating collection:", e)
    sys.exit(1)

# 2. Scroll from local and upsert to cloud
print("Starting Qdrant sync from local to cloud...")
offset = None
batch_idx = 0
total_synced = 0
start_time = time.time()

while True:
    res, offset = local_client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=True,
        offset=offset
    )
    if not res:
        break
        
    # Prepare batch for upsert
    points_to_upsert = []
    for pt in res:
        # Convert local point to PointStruct
        points_to_upsert.append(
            models.PointStruct(
                id=pt.id,
                vector=pt.vector,
                payload=pt.payload
            )
        )
        
    # Upsert to cloud
    try:
        cloud_client.upsert(
            collection_name=collection_name,
            points=points_to_upsert
        )
        total_synced += len(points_to_upsert)
        print(f"Batch {batch_idx}: Synced {len(points_to_upsert)} points. Total: {total_synced}")
    except Exception as e:
        print(f"Error upserting batch {batch_idx}: {e}")
        time.sleep(2)
        # Retry once
        try:
            cloud_client.upsert(collection_name=collection_name, points=points_to_upsert)
            total_synced += len(points_to_upsert)
            print(f"Batch {batch_idx}: Synced {len(points_to_upsert)} points on retry. Total: {total_synced}")
        except Exception as retry_err:
            print(f"Retry failed: {retry_err}. Skipping batch.")
            
    batch_idx += 1
    if offset is None:
        break

end_time = time.time()
print(f"\nQdrant sync completed! Synced {total_synced} points in {end_time - start_time:.2f} seconds.")
