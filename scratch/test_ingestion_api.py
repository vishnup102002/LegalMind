import os
import re
import time
import requests
from bs4 import BeautifulSoup

file_path = "/Users/vishnup/Desktop/projects/Legalmind/data/laws-of-india/consolidated/Indian Criminal Law Amendment Act, 1908.xml"

with open(file_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

short_title_tag = soup.find("shorttitle")
title = short_title_tag.text.strip() if short_title_tag else "Unknown Act"
filename = os.path.basename(file_path)
statute_id = "laws_of_india_" + re.sub(r"[^a-zA-Z0-9]+", "_", filename.replace(".xml", "")).lower()

sections = soup.find_all("section")

def get_section_text(sec):
    current_section = sec
    
    def element_to_text(el):
        if not el.name:
            return el.strip()
        if el.name == "section" and el != current_section:
            return ""
        if el.name in ["num", "heading"]:
            if el.parent == current_section:
                return ""
            return el.get_text().strip()
        if el.name in ["p", "listintroduction"]:
            return el.get_text().strip()
        if el.name in ["subsection", "item"]:
            num_child = el.find("num", recursive=False)
            num_text = num_child.get_text().strip() if num_child else ""
            other_parts = []
            for child in el.children:
                if child != num_child:
                    child_text = element_to_text(child)
                    if child_text:
                        other_parts.append(child_text)
            other_text = "\n".join(other_parts)
            if num_text:
                return f"{num_text} {other_text}"
            return other_text
        parts = []
        for child in el.children:
            child_text = element_to_text(child)
            if child_text:
                parts.append(child_text)
        if el.name in ["paragraph", "blocklist"]:
            return "\n".join(parts)
        else:
            return " ".join(parts)

    text = element_to_text(sec)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

formatted_sections = []
for sec in sections:
    num_tag = sec.find("num", recursive=False)
    heading_tag = sec.find("heading", recursive=False)
    num = num_tag.text.strip() if num_tag else ""
    num_cleaned = re.sub(r"\.+$", "", num).strip()
    heading = heading_tag.text.strip() if heading_tag else ""
    body = get_section_text(sec)
    formatted_sections.append(f"SECTION {num_cleaned}: {heading}\n{body}")

payload = {
    "statute_id": statute_id,
    "title": title,
    "text": "\n\n".join(formatted_sections)
}

print(f"Ingesting '{title}' (ID: {statute_id}) with {len(sections)} sections...")
start_time = time.time()
response = requests.post("http://localhost:8080/api/documents/ingest", json=payload)
end_time = time.time()

print(f"Response status code: {response.status_code}")
print(f"Response json: {response.json()}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
