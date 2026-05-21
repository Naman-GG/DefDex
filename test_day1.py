# test_day1.py
import json
import os

def check_files():
    targets = ['data/platforms.json', 'data/inventories.json', 'data/theaters.json']
    for path in targets:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            return
        
        with open(path, 'r') as f:
            try:
                data = json.load(f)
                print(f"✅ {path} parsed successfully as clean JSON. Keys found: {list(data.keys())}")
            except json.JSONDecodeError:
                print(f"❌ Syntax Error: {path} is not valid JSON.")

if __name__ == "__main__":
    check_files()