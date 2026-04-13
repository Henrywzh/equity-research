import json
import csv
import re
import os

# GICS to ETF Mapping Dictionary
# This dictionary contains confirmed mappings for major categories.
# Tickers are listed as a list of strings.
ETF_MASTER_MAPPING = {
    # Sectors (2-digit)
    "10": ["XLE", "VDE", "IYE"], # Energy
    "15": ["XLB", "VAW", "IYM"], # Materials
    "20": ["XLI", "VIS", "IYJ"], # Industrials
    "25": ["XLY", "VCR", "IYC"], # Consumer Discretionary
    "30": ["XLP", "VDC", "IYK"], # Consumer Staples
    "35": ["XLV", "VHT", "IYH"], # Health Care
    "40": ["XLF", "VFH", "IYF"], # Financials
    "45": ["XLK", "VGT", "IYW"], # IT
    "50": ["XLC", "VOX"],        # Comm Services
    "55": ["XLU", "VPU", "IDU"], # Utilities
    "60": ["XLRE", "VNQ", "IYR"], # Real Estate

    # Industry Groups (4-digit)
    "1010": ["XLE", "VDE"],
    "1510": ["XLB", "VAW"],
    "2010": ["XLI", "VIS"],
    "2020": ["XLI", "IJB"], # Small cap industrials?
    "2030": ["IYT", "XTN"], # Transportation
    "2510": ["CARZ", "DRIV"], # Autos
    "2520": ["XHB", "ITB", "XRT"], # Durables/Apparel/Homebuilders
    "2530": ["PEJ", "BJK"], # Consumer Services
    "2550": ["XRT"], # Distribution & Retail
    "3010": ["XLP", "XRT"], # Staples Retail
    "3020": ["PBJ", "FTXG"], # Food & Tobacco
    "3510": ["XHE", "IHI"], # Health Care Equip & Services
    "3520": ["XBI", "IBB", "XPH"], # Biotech/Pharma
    "4010": ["KBE"], # Banks
    "4020": ["KCE", "IAI"], # Diversified Financials
    "4030": ["KIE", "IAK"], # Insurance
    "4510": ["XSW", "IGV"], # Software & Services
    "4520": ["XITK"], # Tech Hardware
    "4530": ["SMH", "SOXX", "XSD"], # Semiconductors
    "5010": ["XTL"], # Telecomm
    "5020": ["PBS", "XLC"], # Media & Entertainment

    # Industries (6-digit)
    "101010": ["XES", "IEZ"], # Energy Equipment
    "101020": ["XOP", "IEO"], # Oil & Gas
    "151010": ["XLB", "VAW"], # Chemicals
    "151040": ["XME", "PICK"], # Metals & Mining
    "201010": ["ITA", "XAR", "PPA"], # Aero & Defense
    "203020": ["JETS"], # Airlines
    "352010": ["XBI", "IBB"], # Biotech
    "352020": ["XPH", "IHE", "PJP"], # Pharma
    "401010": ["KBE"], # Banks
    "453010": ["SMH", "SOXX", "XSD"], # Semiconductors
    "551010": ["XLU", "IDU"], # Electric Utilities
    "601010": ["VNQ", "XLRE"], # REITs

    # Sub-Industries (8-digit)
    "10101010": ["IEZ"], # Oil & Gas Drilling
    "15104025": ["COPX"], # Copper
    "15104030": ["GDX", "GDXJ"], # Gold
    "15104045": ["SIL", "SILJ"], # Silver
    "15104050": ["SLX"], # Steel
    "20302010": ["JETS"], # Airlines
    "25201030": ["XHB", "ITB"], # Homebuilding
    "40101015": ["KRE", "IAT"], # Regional Banks
    "55101010": ["XLU"], # Electric Utilities
    "55105020": ["ICLN", "TAN", "FAN"], # Renewable Elec
}

def load_parsed_gics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_mapping(code, name):
    if code in ETF_MASTER_MAPPING:
        return ETF_MASTER_MAPPING[code], "Direct"
    
    # Fallback logic: check parent levels
    parent_code = code[:-2]
    while len(parent_code) >= 2:
        if parent_code in ETF_MASTER_MAPPING:
            level_map = {2: "Sector", 4: "Industry Group", 6: "Industry"}
            return ETF_MASTER_MAPPING[parent_code], f"{level_map[len(parent_code)]} Fallback"
        parent_code = parent_code[:-2]
    
    return [], "None"

def process_hierarchy(gics):
    full_mapping = []
    
    for sector in gics:
        s_tickers, s_level = get_mapping(sector["code"], sector["name"])
        sector_entry = {
            "code": sector["code"],
            "name": sector["name"],
            "level": "Sector",
            "tickers": s_tickers,
            "mapping_type": s_level,
            "groups": []
        }
        
        for group in sector["groups"]:
            g_tickers, g_level = get_mapping(group["code"], group["name"])
            group_entry = {
                "code": group["code"],
                "name": group["name"],
                "level": "Industry Group",
                "tickers": g_tickers,
                "mapping_type": g_level,
                "industries": []
            }
            
            for industry in group["industries"]:
                i_tickers, i_level = get_mapping(industry["code"], industry["name"])
                industry_entry = {
                    "code": industry["code"],
                    "name": industry["name"],
                    "level": "Industry",
                    "tickers": i_tickers,
                    "mapping_type": i_level,
                    "sub_industries": []
                }
                
                for sub in industry["sub_industries"]:
                    si_tickers, si_level = get_mapping(sub["code"], sub["name"])
                    sub_entry = {
                        "code": sub["code"],
                        "name": sub["name"],
                        "level": "Sub-Industry",
                        "tickers": si_tickers,
                        "mapping_type": si_level
                    }
                    industry_entry["sub_industries"].append(sub_entry)
                
                group_entry["industries"].append(industry_entry)
            
            sector_entry["groups"].append(group_entry)
        
        full_mapping.append(sector_entry)
    
    return full_mapping

def flatten_to_csv(full_mapping, target_level):
    rows = []
    for sector in full_mapping:
        for group in sector["groups"]:
            for industry in group["industries"]:
                if target_level == "Industry":
                    rows.append({
                        "GICS Code": industry["code"],
                        "Name": industry["name"],
                        "Tickers": ", ".join(industry["tickers"]),
                        "Mapping Type": industry["mapping_type"],
                        "Parent Sector": sector["name"]
                    })
                
                for sub in industry["sub_industries"]:
                    if target_level == "Sub-Industry":
                        rows.append({
                            "GICS Code": sub["code"],
                            "Name": sub["name"],
                            "Tickers": ", ".join(sub["tickers"]),
                            "Mapping Type": sub["mapping_type"],
                            "Parent Industry": industry["name"],
                            "Parent Sector": sector["name"]
                        })
    return rows

if __name__ == "__main__":
    structured_gics = load_parsed_gics("/Users/henrywzh/.gemini/antigravity/brain/5bbe5d19-ccd3-43bf-9491-ac6d25f97250/scratch/gics_structured.json")
    final_mapping = process_hierarchy(structured_gics)
    
    # Save JSON
    with open("data/gics_etf_mapping.json", "w") as f:
        json.dump(final_mapping, f, indent=2)
    
    # Save Industry CSV
    industry_rows = flatten_to_csv(final_mapping, "Industry")
    with open("data/industries_etfs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["GICS Code", "Name", "Tickers", "Mapping Type", "Parent Sector"])
        writer.writeheader()
        writer.writerows(industry_rows)
        
    # Save Sub-Industry CSV
    sub_industry_rows = flatten_to_csv(final_mapping, "Sub-Industry")
    with open("data/sub_industries_etfs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["GICS Code", "Name", "Tickers", "Mapping Type", "Parent Industry", "Parent Sector"])
        writer.writeheader()
        writer.writerows(sub_industry_rows)

    print("Successfully generated GICS to ETF mapping files.")
