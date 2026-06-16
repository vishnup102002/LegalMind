import os
import sys
from sentence_transformers import SentenceTransformer

# Add the project root to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.vector_store import VectorStore

def run_test():
    print("Connecting to local Qdrant container...")
    store = VectorStore()
    
    # 1. Initialize collection (vector size of all-MiniLM-L6-v2 is 384)
    store.init_collection(vector_size=384)
    
    # 2. Initialize the local embedding model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Define the statutory chunks to index
    documents = [
        {
            "id": 1,
            "text": "Kerala Rent Control Act Section 11: A tenant shall not be evicted, except in accordance with the Rent Control Court guidelines.",
            "citation": "Section 11 Rent Control Act",
            "layer_depth": 0
        },
        {
            "id": 2,
            "text": "Kerala Rent Control Act Section 24: No landlord shall cut off or withhold essential amenities (electricity, water) enjoyed by the tenant.",
            "citation": "Section 24 Rent Control Act",
            "layer_depth": 0
        }
    ]
    
    # 4. Generate embeddings and format points for Qdrant
    print("Embedding legal clauses...")
    points = []
    for doc in documents:
        vector = model.encode(doc["text"]).tolist()
        points.append({
            "id": doc["id"],
            "vector": vector,
            "payload": {
                "text": doc["text"],
                "citation": doc["citation"],
                "layer_depth": doc["layer_depth"]
            }
        })
        
    # 5. Push points to Qdrant
    print("Upserting chunks to Qdrant...")
    store.upsert_chunks(points)
    print("✓ Chunks successfully indexed.")
    
    # 6. Execute a hybrid search query
    query = "tenant eviction rules"
    query_vector = model.encode(query).tolist()
    
    print(f"\nExecuting hybrid search for: '{query}'...")
    results = store.hybrid_search(query_vector=query_vector, query_text=query, top_k=2)
    
    for idx, r in enumerate(results):
        print(f"[{idx+1}] Score: {r['score']:.4f} | Type: {r['type']} | {r['citation']}: {r['text'][:90]}...")

if __name__ == "__main__":
    run_test()
