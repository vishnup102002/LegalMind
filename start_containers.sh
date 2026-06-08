#!/bin/bash

echo "Starting LegalMind localized database containers..."

# Run Qdrant Vector Store
docker run -d \
  --name legalmind-qdrant \
  -p 6333:6333 \
  -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest

# Run Neo4j Graph DB
docker run -d \
  --name legalmind-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/secure_password_123 \
  -v "$(pwd)/data/neo4j_data:/data" \
  neo4j:latest

echo "✓ Database containers deployed."
echo "Qdrant endpoint: http://localhost:6333"
echo "Neo4j Browser: http://localhost:7474 (user: neo4j, pass: secure_password_123)"
echo "Next step: Run 'pip install -r requirements.txt' and boot up 'python app/server.py'."
