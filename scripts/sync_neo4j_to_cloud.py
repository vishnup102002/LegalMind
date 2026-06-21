import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from dotenv import load_dotenv
import time

load_dotenv()

# Connect to local Neo4j
local_uri = "bolt://localhost:7687"
local_user = "neo4j"
local_pass = "secure_password_123"
print(f"Connecting to Local Neo4j at: {local_uri}")
local_driver = GraphDatabase.driver(local_uri, auth=(local_user, local_pass))

# Connect to cloud Neo4j
cloud_uri = os.getenv("NEO4J_URI")
cloud_user = os.getenv("NEO4J_USER")
cloud_pass = os.getenv("NEO4J_PASSWORD")
if not cloud_uri or not cloud_user or not cloud_pass:
    print("Error: NEO4J credentials not found in .env")
    sys.exit(1)

print(f"Connecting to Cloud Neo4j at: {cloud_uri}")
cloud_driver = GraphDatabase.driver(cloud_uri, auth=(cloud_user, cloud_pass))

def sync_nodes(label):
    print(f"\n--- Syncing {label} nodes ---")
    query_read = f"MATCH (n:{label}) RETURN n.id AS id, properties(n) AS props"
    
    with local_driver.session() as local_sess:
        results = list(local_sess.run(query_read))
        
    print(f"Found {len(results)} {label} nodes locally.")
    if not results:
        return
        
    # Sync in batches of 1000
    batch_size = 1000
    batch = []
    total_synced = 0
    
    query_write = f"""
    UNWIND $batch AS row
    MERGE (n:{label} {{id: row.id}})
    SET n = row.props
    """
    
    start_time = time.time()
    for idx, record in enumerate(results):
        batch.append({
            "id": record["id"],
            "props": dict(record["props"])
        })
        
        if len(batch) >= batch_size or idx == len(results) - 1:
            with cloud_driver.session() as cloud_sess:
                cloud_sess.run(query_write, batch=batch)
            total_synced += len(batch)
            print(f"Synced {total_synced}/{len(results)} {label} nodes...")
            batch = []
            
    print(f"Finished syncing {label} nodes in {time.time() - start_time:.2f} seconds.")

def sync_relationships(rel_type, label_from, label_to):
    print(f"\n--- Syncing {rel_type} relationships ({label_from} -> {label_to}) ---")
    query_read = f"""
    MATCH (a:{label_from})-[r:{rel_type}]->(b:{label_to})
    RETURN a.id AS from_id, b.id AS to_id
    """
    
    with local_driver.session() as local_sess:
        results = list(local_sess.run(query_read))
        
    print(f"Found {len(results)} {rel_type} relationships locally.")
    if not results:
        return
        
    batch_size = 1000
    batch = []
    total_synced = 0
    
    query_write = f"""
    UNWIND $batch AS row
    MATCH (a:{label_from} {{id: row.from_id}})
    MATCH (b:{label_to} {{id: row.to_id}})
    MERGE (a)-[:{rel_type}]->(b)
    """
    
    start_time = time.time()
    for idx, record in enumerate(results):
        batch.append({
            "from_id": record["from_id"],
            "to_id": record["to_id"]
        })
        
        if len(batch) >= batch_size or idx == len(results) - 1:
            with cloud_driver.session() as cloud_sess:
                cloud_sess.run(query_write, batch=batch)
            total_synced += len(batch)
            print(f"Synced {total_synced}/{len(results)} {rel_type} relationships...")
            batch = []
            
    print(f"Finished syncing {rel_type} relationships in {time.time() - start_time:.2f} seconds.")

try:
    # 1. Sync Statute nodes
    sync_nodes("Statute")
    
    # 2. Sync Section nodes
    sync_nodes("Section")
    
    # 3. Sync Case nodes
    sync_nodes("Case")
    
    # 4. Sync HAS_SECTION relationships
    sync_relationships("HAS_SECTION", "Statute", "Section")
    
    # 5. Sync CITES relationships
    sync_relationships("CITES", "Case", "Section")
    
    print("\nAll Neo4j data synced successfully!")
finally:
    local_driver.close()
    cloud_driver.close()
