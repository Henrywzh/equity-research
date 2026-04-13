import collections
import json
import re

def parse_gics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    classification = []
    current_sector = None
    current_group = None
    current_industry = None

    for line in lines[1:]: # Skip header
        line = line.strip()
        if not line:
            continue
        
        # The structure is tab-ish or space-ish. 
        # Let's try to split by multiple spaces or tabs.
        parts = re.split(r'\t| {2,}', line)
        
        # Logic to handle different lengths and levels
        if len(parts) >= 2 and len(parts[0]) == 2: # Sector
            current_sector = {"code": parts[0], "name": parts[1], "groups": []}
            classification.append(current_sector)
        elif len(parts) >= 2 and len(parts[0]) == 4: # Group
            current_group = {"code": parts[0], "name": parts[1], "industries": []}
            current_sector["groups"].append(current_group)
        elif len(parts) >= 2 and len(parts[0]) == 6: # Industry
            current_industry = {"code": parts[0], "name": parts[1], "sub_industries": []}
            current_group["industries"].append(current_industry)
        elif len(parts) >= 2 and len(parts[0]) == 8: # Sub-Industry
            sub = {"code": parts[0], "name": parts[1]}
            current_industry["sub_industries"].append(sub)
        else:
            # Handle cases where the text might be shifted or missing codes on same line
            # The prompt text has some codes and names mixed.
            # E.g. "10101010	Oil & Gas Drilling"
            if len(parts) == 2 and len(parts[0]) == 8:
                 sub = {"code": parts[0], "name": parts[1]}
                 current_industry["sub_industries"].append(sub)

    return classification

if __name__ == "__main__":
    gics = parse_gics("data/gics_classification.txt")
    with open("data/gics_structured.json", "w") as f:
        json.dump(gics, f, indent=2)
    print("Parsed GICS classification.")
