#!/usr/bin/env python3
import os
import re
import sys
import glob
import time
import requests
from bs4 import BeautifulSoup

CONSOLIDATED_DIR = "/Users/vishnup/Desktop/projects/Legalmind/data/laws-of-india/consolidated"
API_URL = "http://localhost:8080/api/documents/ingest"
LOG_FILE = "/Users/vishnup/Desktop/projects/Legalmind/logs/ingest_all.log"
TRACKER_FILE = "/Users/vishnup/Desktop/projects/Legalmind/data/ingested_files.txt"

def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def mark_as_ingested(filename):
    with open(TRACKER_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def load_ingested_files():
    if not os.path.exists(TRACKER_FILE):
        return set()
    with open(TRACKER_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def get_section_text(sec, current_section):
    def element_to_text(el):
        if not el.name:
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
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def extract_jurisdiction(filename):
    filename_lower = filename.lower()
    states = [
        "kerala", "karnataka", "andhra pradesh", "assam", "delhi", 
        "maharashtra", "tamil nadu", "cochin", "travancore"
    ]
    for state in states:
        if state in filename_lower:
            if state == "cochin":
                return "Cochin"
            if state == "travancore":
                return "Travancore"
            if state == "andhra pradesh":
                return "Andhra Pradesh"
            if state == "tamil nadu":
                return "Tamil Nadu"
            return state.capitalize()
    return "Central"

def process_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            
        short_title_tag = soup.find("shorttitle")
        title = short_title_tag.text.strip() if short_title_tag else ""
        if not title:
            # Try plain title tag
            title_tag = soup.find("title")
            title = title_tag.text.strip() if title_tag else ""
            
        filename = os.path.basename(file_path)
        base_name = filename.replace(".xml", "")
        if not title:
            title = base_name
            
        statute_id = "laws_of_india_" + re.sub(r"[^a-zA-Z0-9]+", "_", base_name).lower()
        
        sections = soup.find_all("section")
        if not sections:
            log_message(f"⚠️ Skip {filename}: No sections found.")
            return False
            
        formatted_sections = []
        for sec in sections:
            num_tag = sec.find("num", recursive=False)
            heading_tag = sec.find("heading", recursive=False)
            
            num = num_tag.text.strip() if num_tag else ""
            num_cleaned = re.sub(r"\.+$", "", num).strip()
            heading = heading_tag.text.strip() if heading_tag else ""
            
            body = get_section_text(sec, sec)
            if body:
                formatted_sections.append(f"SECTION {num_cleaned}: {heading}\n{body}")
                
        if not formatted_sections:
            # Fallback: treat entire body as one section
            body_tag = soup.find("body")
            body_text = body_tag.get_text().strip() if body_tag else soup.get_text().strip()
            formatted_sections.append(f"SECTION 1: General Provisions\n{body_text}")
            
        jurisdiction = extract_jurisdiction(filename)
        payload = {
            "statute_id": statute_id,
            "title": title,
            "text": "\n\n".join(formatted_sections),
            "jurisdiction": jurisdiction
        }
        
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=120)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            res_json = response.json()
            log_message(f"✅ Ingested '{title}' ({len(sections)} sections) in {duration:.2f}s - {res_json.get('message', '')}")
            mark_as_ingested(filename)
            return True
        else:
            log_message(f"❌ Failed to ingest '{title}' - Status code: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        log_message(f"💥 Exception processing {file_path}: {e}")
        return False

def main():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_message("=== Starting Bulk Ingestion of XML Statutes ===")
    
    xml_files = sorted(glob.glob(os.path.join(CONSOLIDATED_DIR, "*.xml")))
    total_files = len(xml_files)
    log_message(f"Found {total_files} XML files to process.")
    
    ingested_files = load_ingested_files()
    log_message(f"Loaded {len(ingested_files)} already ingested files from tracker.")
    
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    for idx, file_path in enumerate(xml_files, 1):
        filename = os.path.basename(file_path)
        if filename in ingested_files:
            log_message(f"[{idx}/{total_files}] ⏭ Skipped (already ingested): {filename}")
            skipped_count += 1
            continue
            
        log_message(f"[{idx}/{total_files}] Processing {filename}...")
        if process_file(file_path):
            success_count += 1
        else:
            failure_count += 1
            
    log_message(f"=== Ingestion Completed: {success_count} succeeded, {skipped_count} skipped, {failure_count} failed ===")

if __name__ == "__main__":
    main()
