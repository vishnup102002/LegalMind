import os
import re
from bs4 import BeautifulSoup

file_path = "/Users/vishnup/Desktop/projects/Legalmind/data/laws-of-india/consolidated/Indian Criminal Law Amendment Act, 1908.xml"

with open(file_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# Get short title
short_title_tag = soup.find("shorttitle")
title = short_title_tag.text.strip() if short_title_tag else "Unknown Act"
print(f"Title: {title}")

# Find sections
sections = soup.find_all("section")
print(f"Found {len(sections)} sections.")

def get_section_text(sec):
    current_section = sec
    
    def element_to_text(el):
        if not el.name:
            # text node
            return el.strip()
        
        if el.name == "section" and el != current_section:
            return ""
            
        # Direct num/heading of the section itself are skipped
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
    # Clean up double newlines or spaces
    text = re.sub(r' +', ' ', text)
    # Make sure we keep standard formatting
    # Collapse multiple consecutive newlines into at most two
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

for sec in sections[:4]:
    num_tag = sec.find("num", recursive=False)
    heading_tag = sec.find("heading", recursive=False)
    
    num = num_tag.text.strip() if num_tag else ""
    num_cleaned = re.sub(r"\.+$", "", num).strip()
    
    heading = heading_tag.text.strip() if heading_tag else ""
    
    body = get_section_text(sec)
    
    print(f"\nSECTION {num_cleaned}: {heading}")
    print(body)
