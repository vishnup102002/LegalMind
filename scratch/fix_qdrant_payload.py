from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FixQdrant")

client = QdrantClient(host="localhost", port=6333)
collection_name = "legal_chunks"

# Dictionary mapping lowercase strings/keywords to correct lowercase jurisdiction values
JURISDICTION_MAP = {
    "gujarat": "gujarat",
    "orissa": "orissa",
    "odisha": "orissa",
    "west bengal": "west bengal",
    "rajasthan": "rajasthan",
    "punjab": "punjab",
    "jammu": "jammu and kashmir",
    "kashmir": "jammu and kashmir",
    "haryana": "haryana",
    "madhya pradesh": "madhya pradesh",
    "uttar pradesh": "uttar pradesh",
    "himachal pradesh": "himachal pradesh",
    "bihar": "bihar",
    "goa": "goa",
    "karnataka": "karnataka",
    "kerala": "kerala",
    "tamil nadu": "tamil nadu",
    "andhra pradesh": "andhra pradesh",
    "maharashtra": "maharashtra",
    "delhi": "delhi",
    "assam": "assam",
    "pondicherry": "pondicherry",
    "puducherry": "pondicherry",
    "uttarakhand": "uttarakhand",
    "chhattisgarh": "chhattisgarh",
    "jharkhand": "jharkhand",
    "manipur": "manipur",
    "meghalaya": "meghalaya",
    "mizoram": "mizoram",
    "nagaland": "nagaland",
    "sikkim": "sikkim",
    "tripura": "tripura",
    "telangana": "telangana",
    "cochin": "cochin",
    "travancore": "travancore",
    "cyberabad": "telangana",
    "hyderabad": "telangana",
    "c.m.r university": "karnataka",
    "dayananda sagara": "karnataka",
    "alliance university": "karnataka",
    "amrutha sinchana": "karnataka",
    "kannada university": "karnataka",
    "shree siddhi vinayak": "maharashtra",
    "sri malai mahadeswaraswamy": "karnataka"
}

logger.info("Scrolling central points to inspect and correct...")
offset = None
total_scanned = 0
total_updated = 0

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
        limit=500,
        with_payload=True,
        with_vectors=False,
        offset=offset
    )
    
    if not res:
        break
        
    points_to_update = {} # jurisdiction -> list of point ids
    
    for point in res:
        total_scanned += 1
        citation = point.payload.get("citation", "").lower()
        title = point.payload.get("title", "").lower()
        text = point.payload.get("text", "").lower()
        
        # Check if any keyword matches
        matched_jur = None
        for keyword, jur in JURISDICTION_MAP.items():
            if keyword in citation or keyword in title:
                matched_jur = jur
                break
                
        if matched_jur:
            if matched_jur not in points_to_update:
                points_to_update[matched_jur] = []
            points_to_update[matched_jur].append(point.id)
            
    for jur, ids in points_to_update.items():
        client.set_payload(
            collection_name=collection_name,
            payload={"jurisdiction": jur},
            points=ids
        )
        total_updated += len(ids)
        logger.info(f"Updated {len(ids)} points to jurisdiction: '{jur}'")
        
    if offset is None:
        break

logger.info(f"Scan complete. Total points scanned: {total_scanned}, Total points updated: {total_updated}")
