import os
import glob
import json
from collections import defaultdict

# ——— 설정 ———
IMAGE_FOLDER = "data/train/image"   # 원본 이미지 폴더
JSON_FOLDER  = "data/train/json"     # 원본 JSON 폴더
# —————————

# 1) JSON 파일을 읽어서 엔트리 수집
entries = []
for jf in glob.glob(os.path.join(JSON_FOLDER, "*.json")):
    with open(jf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ann = data.get("annotations", {})
    crop    = ann.get("crop",    None)
    disease = ann.get("disease", None)
    img_name = data.get("description", {}).get("image")
    if crop is None or disease is None or not img_name:
        continue

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    if not os.path.exists(img_path):
        print(f"[WARN] 이미지 없음: {img_path}")
        continue

    entries.append({
        "orig_img_name": img_name,
        "orig_img_path": img_path,
        "orig_json_path": jf,
        "crop":          crop,
        "disease":       disease,
    })

# 2) 원본 이미지 이름 기준으로 사전(lex) 정렬
entries.sort(key=lambda e: e["orig_img_name"])

# 3) (crop, disease)별 그룹화 및 인덱스 매기기
groups = defaultdict(list)
for e in entries:
    key = (e["crop"], e["disease"])
    groups[key].append(e)

# 4) 파일명 변경 & JSON 내용 업데이트
for (crop, disease), items in groups.items():
    for idx, e in enumerate(items, start=1):
        base = f"c{crop}_d{disease}_{idx}"
        # 이미지 확장자
        _, ext = os.path.splitext(e["orig_img_name"])
        new_img_name = base + ext
        new_json_name = base + ".json"

        new_img_path  = os.path.join(IMAGE_FOLDER, new_img_name)
        new_json_path = os.path.join(JSON_FOLDER,  new_json_name)

        # 4.1 이미지 파일명 변경
        print(f"Renaming image: {e['orig_img_name']} → {new_img_name}")
        os.rename(e["orig_img_path"], new_img_path)

        # 4.2 JSON 파일명 변경
        orig_json_name = os.path.basename(e["orig_json_path"])
        print(f"Renaming JSON:  {orig_json_name} → {new_json_name}")
        os.rename(e["orig_json_path"], new_json_path)

        # 4.3 JSON 내용 열고 description.image 업데이트
        with open(new_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "description" not in data:
            data["description"] = {}
        data["description"]["image"] = new_img_name
        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

print("모든 이미지 및 JSON 파일명이 변경되고 description.image가 업데이트되었습니다.")
