import re
import os

log_path = "/Users/vishnup/Desktop/projects/Legalmind/logs/ingest_all.log"
output_path = "/Users/vishnup/Desktop/projects/Legalmind/data/ingested_files.txt"

if not os.path.exists(log_path):
    print("Log file not found")
    exit(1)

# Find all file processing and ingestion matching pairs
# e.g.:
# [2026-06-10 13:17:09] [1/808] Processing Aadhaar (Targeted Delivery Of Financial And Other Subsidies, Benefits And Services) Act, 2016.xml...
# [2026-06-10 13:17:15] ✅ Ingested 'Aadhaar (Targeted Delivery Of Financial And Other Subsidies, Benefits And Services) Act, 2016' ...

processing_map = {} # title -> filename
ingested_files = set()

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        # Match Processing
        proc_match = re.search(r"Processing (.*?\.xml)\.\.\.", line)
        if proc_match:
            filename = proc_match.group(1)
            # Find the title if possible, or mapping via filename
            # Let's map clean filename/title later.
            # But wait! A simpler way: if the line contains "✅ Ingested", we can match the title:
            # ✅ Ingested 'Title'
            # Let's associate it with the last seen XML file!
            # Since the script runs synchronously: [idx/total] Processing File.xml... and then ✅ Ingested 'Title'
            # So the most recent "Processing File.xml" is the one that succeeded!
            last_processing_file = filename
            
        ingest_match = re.search(r"✅ Ingested '(.*?)'", line)
        if ingest_match:
            if last_processing_file:
                ingested_files.add(last_processing_file)
                
print(f"Extracted {len(ingested_files)} successfully ingested files.")
with open(output_path, "w", encoding="utf-8") as out:
    for filename in sorted(list(ingested_files)):
        out.write(filename + "\n")
print(f"Written to {output_path}")
