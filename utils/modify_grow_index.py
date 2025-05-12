import os
import json
from glob import glob

# 처리할 디렉터리 목록
splits = ["train", "test"]

for split in splits:
    json_paths = glob(os.path.join("data", split, "json", "*.json"))
    for path in json_paths:
        change = False
        # JSON 읽기
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # grow 값 감소
        if "annotations" in data and "grow" in data["annotations"]:
            raw = int(data["annotations"]["grow"])
            if data["annotations"]["grow"] >= 11:
                data["annotations"]["grow"] = raw - 11
                change = True
            if data["annotations"]["grow"] == 2:
                print(f"{path}: grow = 2")
        elif "grow" in data:
            raw = int(data["grow"])
            if data["grow"] >= 11:
                data["grow"] = raw - 11
                change = True
            if data["grow"] == 2:
                print(f"{path}: grow = 2")
        else:
            # grow 필드가 없으면 건너뜀
            continue

        if change:
            # 덮어쓰기
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Updated {path}: grow {raw} → {raw-11}")
