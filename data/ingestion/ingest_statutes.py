import os
import re
import sys

# Ensure project root is in import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.graph_store import GraphStore

class StatuteIngester:
    def __init__(self):
        self.store = GraphStore()

    def parse_act_file(self, file_path: str) -> dict:
        """
        Parses raw text acts into a structured dictionary.
        Looks for the Act Title on the first line, then identifies sections matching 'SECTION X: TITLE'.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return {}

        # First non-empty line is the Act Title
        act_title = ""
        for line in lines:
            if line.strip():
                act_title = line.strip()
                break

        # Combine text and find sections using regular expressions
        full_text = "".join(lines)
        section_pattern = re.compile(r"SECTION\s+(\d+):\s+([^\n]+)\n(.*?)(?=SECTION\s+\d+:|$)", re.DOTALL | re.IGNORECASE)
        
        sections = []
        matches = section_pattern.findall(full_text)
        
        for num, title, body in matches:
            sections.append({
                "number": num.strip(),
                "title": f"Section {num.strip()}: {title.strip()}",
                "body": body.strip()
            })

        return {
            "title": act_title,
            "sections": sections
        }

    def load_into_graph(self, act_data: dict, statute_id: str):
        """
        Loads the structured act data directly into the Neo4j graph using GraphStore.
        """
        print(f"Loading Statute: '{act_data['title']}' (ID: {statute_id})")
        self.store.add_statute(statute_id, act_data["title"])

        for sec in act_data["sections"]:
            sec_id = f"{statute_id}_sec_{sec['number']}"
            citation = f"Section {sec['number']}, {act_data['title']}"
            
            print(f" -> Loading Section {sec['number']}: {sec['title']}")
            self.store.add_section(
                statute_id=statute_id,
                section_id=sec_id,
                title=sec["title"],
                text=sec["body"],
                citation=citation
            )
        
        print("✓ Loading completed successfully.")

if __name__ == "__main__":
    # Locate Act text
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_txt = os.path.join(base_dir, "data", "statutes", "kerala_rent_control_act.txt")
    
    ingester = StatuteIngester()
    try:
        # 1. Parse Act
        parsed_data = ingester.parse_act_file(source_txt)
        print(f"Parsed Act: {parsed_data['title']} ({len(parsed_data['sections'])} sections found)")
        
        # 2. Upload to Neo4j
        ingester.load_into_graph(parsed_data, "kerala_buildings_rent_control_1965")
    except Exception as e:
        print(f"Ingestion process halted: {e}")
        print("If database connection failed, ensure Docker containers are running first.")
