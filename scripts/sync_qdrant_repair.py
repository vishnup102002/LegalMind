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
cloud_client = QdrantClient(url=cloud_host, api_key=cloud_api_key)

collection_name = "legal_chunks"

print("Fetching all point IDs from Local Qdrant...")
local_ids = set()
offset = None
while True:
    res, offset = local_client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=False,
        with_vectors=False,
        offset=offset
    )
    if not res:
        break
    for pt in res:
        local_ids.add(pt.id)
    print(f"Loaded {len(local_ids)} local IDs...")
    if offset is None:
        break

print("Fetching all point IDs from Cloud Qdrant...")
cloud_ids = set()
offset = None
while True:
    res, offset = cloud_client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=False,
        with_vectors=False,
        offset=offset
    )
    if not res:
        break
    for pt in res:
        cloud_ids.add(pt.id)
    print(f"Loaded {len(cloud_ids)} cloud IDs...")
    if offset is None:
        break

missing_ids = list(local_ids - cloud_ids)
print(f"\nMissing points count: {len(missing_ids)}")

if not missing_ids:
    print("All points are already present in Cloud Qdrant. Sync is complete!")
    sys.exit(0)

# Upsert missing points in smaller batches (e.g. 200 points) to avoid timeouts
batch_size = 200
total_upserted = 0
start_time = time.time()

print(f"Uploading {len(missing_ids)} missing points to Qdrant Cloud in batches of {batch_size}...")

for idx in range(0, len(missing_ids), batch_size):
    batch_ids = missing_ids[idx:idx + batch_size]
    
    # Retrieve from local
    points_info = local_client.retrieve(
        collection_name=collection_name,
        ids=batch_ids,
        with_payload=True,
        with_vectors=True
    )
    
    # Prepare PointStruct
    points_to_upsert = []
    for pt in points_info:
        points_to_upsert.append(
            models.PointStruct(
                id=pt.id,
                vector=pt.vector,
                payload=pt.payload
            )
        )
        
    # Upsert to cloud with retries
    success = False
    for attempt in range(3):
        try:
            cloud_client.upsert(
                collection_name=collection_name,
                points=points_to_upsert
            )
            success = True
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for batch starting at {idx}: {e}")
            time.sleep(2)
            
    if success:
        total_upserted += len(points_to_upsert)
        print(f"Uploaded batch {idx // batch_size + 1}: {len(points_to_upsert)} points. Total: {total_upserted}")
    else:
        print(f"Failed to upload batch starting at {idx} after 3 attempts.")

print(f"\nRepair sync completed! Uploaded {total_upserted} missing points in {time.time() - start_time:.2f} seconds.")
