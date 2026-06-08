// Neo4j Graph Database Constraints & Core Indices
CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:Statute) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (sec:Section) REQUIRE sec.id IS UNIQUE;
CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE;

CREATE INDEX section_citation_idx IF NOT EXISTS FOR (sec:Section) ON (sec.citation);
CREATE INDEX case_name_idx IF NOT EXISTS FOR (c:Case) ON (c.name);

// Seed Statute: Kerala Rent Control Act (Selected Sections)
MERGE (s:Statute {id: "kerala_rent_control_act_1965"})
SET s.title = "Kerala Buildings (Lease and Rent Control) Act, 1965",
    s.jurisdiction = "Kerala"

MERGE (sec11:Section {id: "sec_11_eviction"})
SET sec11.title = "Section 11: Eviction of Tenants",
    sec11.citation = "Section 11, Kerala Buildings Rent Control Act",
    sec11.text = "A tenant shall not be evicted, whether in execution of a decree or otherwise, except in accordance with the provisions of this section..."

MERGE (sec24:Section {id: "sec_24_essential_services"})
SET sec24.title = "Section 24: Landlord not to cut off or withhold amenities",
    sec24.citation = "Section 24, Kerala Buildings Rent Control Act",
    sec24.text = "No landlord shall without just or sufficient cause cut off or withhold any of the amenities enjoyed by the tenant..."

MERGE (s)-[:HAS_SECTION]->(sec11)
MERGE (s)-[:HAS_SECTION]->(sec24);
