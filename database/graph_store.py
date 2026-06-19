from neo4j import GraphDatabase
import logging
import os

logger = logging.getLogger("LegalMind.GraphStore")

class GraphStore:
    def __init__(self, uri=None, username=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Initializes relational constraints and indices for the legal ontology."""
        queries = [
             "CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:Statute) REQUIRE s.id IS UNIQUE",
             "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
             "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
             "CREATE INDEX section_citation_idx IF NOT EXISTS FOR (s:Section) ON (s.citation)",
             "CREATE INDEX case_name_idx IF NOT EXISTS FOR (c:Case) ON (c.name)"
        ]
        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                     # pyrefly: ignore [parse-error]
                    logger.info(f"Executed index/constraint setup: {query}")
                except Exception as e:
                    logger.warning(f"Failed to execute database schema instruction: {e}")

            # Warm up schema to register labels/relationships and silence warnings
            warmup_queries = [
                "MERGE (c:Case {id: 'schema_warmup_temp'})",
                "MERGE (sec:Section {id: 'schema_warmup_temp'})",
                "MERGE (st:Statute {id: 'schema_warmup_temp'})",
                "MERGE (c)-[:CITES]->(sec)",
                "MERGE (st)-[:HAS_SECTION]->(sec)",
                "MATCH (n {id: 'schema_warmup_temp'}) DETACH DELETE n"
            ]
            for query in warmup_queries:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Warmup query failed: {e}")
            print("✓ Database constraint and index structures initialized.")


    def add_statute(self, statute_id: str, title: str, jurisdiction: str = "Central"):
        """Creates a Statute node in the graph."""
        query = """
        MERGE (s:Statute {id: $id})
        SET s.title = $title, s.jurisdiction = $jurisdiction
        RETURN s
        """
        with self.driver.session() as session:
            session.run(query, id=statute_id, title=title, jurisdiction=jurisdiction)

    def add_section(self, statute_id: str, section_id: str, title: str, text: str, citation: str):
        """Creates a Section node linked to its parent Statute."""
        query = """
        MATCH (s:Statute {id: $statute_id})
        MERGE (sec:Section {id: $section_id})
        SET sec.title = $title, sec.text = $text, sec.citation = $citation
        MERGE (s)-[:HAS_SECTION]->(sec)
        RETURN sec
        """
        with self.driver.session() as session:
            session.run(query, statute_id=statute_id, section_id=section_id, title=title, text=text, citation=citation)

    def add_case_precedent(self, case_id: str, name: str, citation: str, summary: str, cites_section_ids: list):
        """Creates a Case node, links it to sections it cites, or other cases it overrules."""
        query = """
        MERGE (c:Case {id: $case_id})
        SET c.name = $name, c.citation = $citation, c.summary = $summary
        WITH c
        UNWIND $cites_section_ids AS sec_id
        MATCH (sec:Section {id: sec_id})
        MERGE (c)-[:CITES]->(sec)
        """
        with self.driver.session() as session:
            session.run(query, case_id=case_id, name=name, citation=citation, summary=summary, cites_section_ids=cites_section_ids)

    def get_related_provisions(self, section_id: str):
        """
        Executes an optimized multi-hop Cypher query to retrieve sections,
        statutes, and citing cases surrounding a particular section.
        Sub-10ms target query using indexed id constraint.
        """
        query = """
        MATCH (sec:Section {id: $section_id})
        OPTIONAL MATCH (s:Statute)-[:HAS_SECTION]->(sec)
        OPTIONAL MATCH (c:Case)-[:CITES]->(sec)
        RETURN sec AS section, s AS statute, collect(c) AS citing_cases
        """
        with self.driver.session() as session:
            result = session.run(query, section_id=section_id)
            record = result.single()
            if record:
                return {
                    "section": dict(record["section"]) if record["section"] else None,
                    "statute": dict(record["statute"]) if record["statute"] else None,
                    "citing_cases": [dict(case) for case in record["citing_cases"] if case]
                }
            return None

if __name__ == "__main__":
    import sys
    # Support direct execution style matching the README setup workflow
    initializer = GraphStore()
    initializer.create_constraints()
