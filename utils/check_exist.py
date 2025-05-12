import os
import glob
import json

# --- 설정 ---
JSON_FOLDER  = "data/train/json"
IMAGE_FOLDER = "data/train/image"

# 지원할 확장자 목록 (필요에 따라 추가)
EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

missing = []

# 1) JSON 파일 순회
for json_path in glob.glob(os.path.join(JSON_FOLDER, "*.json")):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_name = data.get('description', {}).get('image')
    if not img_name:
        missing.append((json_path, None))
        continue

    base, ext = os.path.splitext(img_name)
    candidates = []

    # 확장자 포함 이름이 있으면 그대로 검사
    if ext:
        candidates.append(os.path.join(IMAGE_FOLDER, img_name))
    # 확장자가 빠져 있거나 대소문자가 다르면 가능한 모든 EXTS로 검사
    for e in EXTS:
        candidates.append(os.path.join(IMAGE_FOLDER, base + e))

    # 2) 하나라도 존재하면 OK, 아니면 missing 기록
    if not any(os.path.isfile(p) for p in candidates):
        missing.append((json_path, img_name))

# 3) 결과 출력
if not missing:
    print("✅ 모든 JSON 파일에 대응하는 이미지가 존재합니다.")
else:
    print("❌ 누락된 이미지가 있습니다:")
    for json_path, img_name in missing:
        if img_name:
            print(f"  • {json_path} → {img_name}")
        else:
            print(f"  • {json_path} → description.image 항목 없음")
