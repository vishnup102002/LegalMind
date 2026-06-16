#!/usr/bin/env python3
"""
LegalMind Cloud Credentials Verification Utility
This script tests connectivity to:
1. Neo4j AuraDB (managed Graph Database)
2. Qdrant Cloud (managed Vector Database)
3. Groq Cloud API (managed LLM inference)

Run this before deploying or inside the EC2 instance to ensure credentials are correct.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv

# Load env variables from root folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))

def print_result(service_name, status, details=""):
    color = "\033[92m" if status == "SUCCESS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{color}{status}{reset}] {service_name}: {details}")

def test_neo4j():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not password:
        print_result("Neo4j", "FAILED", "Missing NEO4J_URI or NEO4J_PASSWORD in .env")
        return False
        
    try:
        from neo4j import GraphDatabase
        # Using a default 5-second connection timeout
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)
        with driver.session() as session:
            result = session.run("RETURN 1 AS val")
            val = result.single()["val"]
            if val == 1:
                print_result("Neo4j", "SUCCESS", f"Connected to {uri}")
                driver.close()
                return True
            else:
                raise ValueError("Did not receive correct heartbeat response.")
    except Exception as e:
        print_result("Neo4j", "FAILED", str(e))
        return False

def test_qdrant():
    host = os.getenv("QDRANT_HOST")
    port = os.getenv("QDRANT_PORT")
    
    # Qdrant Cloud URIs often use standard 6333 or 6334, and may require api-key
    # Check if a custom key is required (often passed as QDRANT_API_KEY)
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not host:
        print_result("Qdrant", "FAILED", "Missing QDRANT_HOST in .env")
        return False
        
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=host if host.startswith("http") else None,
            host=host if not host.startswith("http") else None,
            port=int(port) if port and not host.startswith("http") else None,
            api_key=api_key,
            timeout=5.0
        )
        collections = client.get_collections()
        names = [col.name for col in collections.collections]
        print_result("Qdrant", "SUCCESS", f"Connected to host. Found {len(names)} collections: {names}")
        return True
    except Exception as e:
        print_result("Qdrant", "FAILED", str(e))
        return False

def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    if not api_key:
        print_result("Groq API", "FAILED", "Missing GROQ_API_KEY in .env")
        return False
        
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5
        }
        encoded_data = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=encoded_data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=5) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            answer = res_body["choices"][0]["message"]["content"].strip()
            print_result("Groq API", "SUCCESS", f"Successfully reached model '{model}'. Response: '{answer}'")
            return True
    except Exception as e:
        print_result("Groq API", "FAILED", str(e))
        return False

if __name__ == "__main__":
    print("=== Starting Cloud Credentials Connectivity Check ===")
    all_ok = True
    all_ok &= test_neo4j()
    all_ok &= test_qdrant()
    all_ok &= test_groq()
    
    print("=====================================================")
    if all_ok:
        print("\033[92mAll database/API configurations are valid! You are ready to deploy to AWS.\033[0m")
        sys.exit(0)
    else:
        print("\033[91mSome connectivity tests failed. Please check your credentials in .env.\033[0m")
        sys.exit(1)
