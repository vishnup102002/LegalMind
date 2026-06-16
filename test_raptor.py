import os
import sys
from sentence_transformers import SentenceTransformer

# Add the project root to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.processing.raptor_processor import RaptorProcessor
from database.vector_store import VectorStore

def run_raptor_test():
    print("Initializing RAPTOR Processor...")
    processor = RaptorProcessor()
    
    print("Loading local Qdrant Store...")
    store = VectorStore()
    store.init_collection(vector_size=384)
    
    # Raw statutory content representing the Rent Control Act
    raw_document = (
        "Kerala Rent Control Act Section 11: A tenant shall not be evicted, except in "
        "accordance with the Rent Control Court guidelines. The court evaluates eviction causes. "
        "Section 24: No landlord shall cut off or withhold essential amenities (electricity, water) "
        "enjoyed by the tenant. The court can order immediate restoration of utilities and penalize "
        "the landlord for wrongful dispossessions."
    )
    
    # 1. Build the hierarchical tree (Leaf layer 0 + Summary layer 1)
    print("Building RAPTOR tree structure...")
    tree = processor.build_tree(raw_document, max_layers=2)
    
    # 2. Embed and upload all tree layers to Qdrant
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    points = []
    point_id_counter = 100  # Start IDs at 100 to avoid conflicts with previous tests
    
    for layer, chunks in tree.items():
        print(f" -> Indexing Layer {layer} ({len(chunks)} chunks)...")
        for chunk in chunks:
            vector = model.encode(chunk).tolist()
            points.append({
                "id": point_id_counter,
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "citation": f"RAPTOR Layer {layer} Node",
                    "layer_depth": layer
                }
            })
            point_id_counter += 1
            
    print("Upserting RAPTOR nodes to Qdrant...")
    store.upsert_chunks(points)
    print("✓ RAPTOR nodes successfully indexed.")
    
    # 3. Query the collection to see what it retrieves
    query = "summary of tenant protections"
    query_vector = model.encode(query).tolist()
    
    print(f"\nExecuting search across all RAPTOR layers for: '{query}'...")
    results = store.hybrid_search(query_vector=query_vector, query_text=query, top_k=2)
    
    for idx, r in enumerate(results):
        print(f"[{idx+1}] Score: {r['score']:.4f} | Source: {r['citation']} | {r['text'][:120]}...")

if __name__ == "__main__":
    run_raptor_test()
