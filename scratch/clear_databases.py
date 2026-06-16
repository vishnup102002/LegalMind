import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.graph_store import GraphStore
from database.vector_store import VectorStore

def clear_databases():
    print("Clearing Neo4j database...")
    gs = GraphStore()
    with gs.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    gs.close()
    print("✓ Neo4j database cleared.")

    print("Clearing Qdrant vector store...")
    vs = VectorStore()
    try:
        vs.client.delete_collection(vs.collection_name)
        print(f"✓ Qdrant collection '{vs.collection_name}' deleted.")
    except Exception as e:
        print(f"Qdrant collection deletion failed (it might not exist): {e}")
    
    # Re-initialize collection
    vs.init_collection()
    
    tracker_path = "data/ingested_files.txt"
    if os.path.exists(tracker_path):
        os.remove(tracker_path)
        print(f"✓ Removed tracker file: {tracker_path}")
        
    print("Database cleaning complete.")

if __name__ == "__main__":
    clear_databases()
