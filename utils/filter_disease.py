import os
import json

JSON_DIR = "data/train/json"

def find_disease2_files(json_dir):
    files = []
    for fn in os.listdir(json_dir):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(json_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f)
            if js.get("annotations", {}).get("disease") == 2:
                files.append(fn)
        except Exception as e:
            print(f"Error reading {fn}: {e}")
    return files

if __name__ == "__main__":
    result = find_disease2_files(JSON_DIR)
    print("disease=2인 JSON 파일 목록:")
    for fn in result:
        print(fn)
