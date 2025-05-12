import os
import glob
import json
import cv2

# --- 설정 부분 ---
mode = 'train'
if mode not in ['train', 'test']:
    raise ValueError("mode는 'train' 또는 'test'여야 합니다.")

IMAGE_FOLDER    = f"data/{mode}/image"
JSON_FOLDER     = f"data/{mode}/json"
OUTPUT_FOLDER   = f"{mode}_image_resized"
TARGET_SIZE     = (1280, 720)   # (width, height)
ASPECT_RATIO    = TARGET_SIZE[0] / TARGET_SIZE[1]
MARGIN_RATIO    = 0.1           # 바운딩박스 주변 여백 비율 (10%)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def compute_roi(box, img_w, img_h,
                aspect=ASPECT_RATIO,
                margin=MARGIN_RATIO):
    """
    box = (xtl, ytl, xbr, ybr)
    이미지 경계를 넘어가지 않게 클리핑된
    16:9 ROI를 반환.
    """
    xtl, ytl, xbr, ybr = box
    # 1) 여백 추가
    w_box = xbr - xtl
    h_box = ybr - ytl
    dx = w_box * margin
    dy = h_box * margin
    xtl -= dx; ytl -= dy
    xbr += dx; ybr += dy

    # 2) 종횡비에 맞춰 영역 확장
    w_box = xbr - xtl
    h_box = ybr - ytl
    if w_box / h_box > aspect:
        W = w_box
        H = W / aspect
    else:
        H = h_box
        W = H * aspect

    # 3) 중심 기준으로 좌표 계산
    cx = (xtl + xbr) / 2
    cy = (ytl + ybr) / 2
    xmin = cx - W/2
    ymin = cy - H/2
    xmax = cx + W/2
    ymax = cy + H/2

    # 4) 이미지 경계 클리핑
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    return int(xmin), int(ymin), int(xmax), int(ymax)


# --- 메인 루프 ---
# 1) 이미지 파일 목록
img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
img_paths = []
for ext in img_exts:
    img_paths += glob.glob(os.path.join(IMAGE_FOLDER, ext))
img_paths.sort()

# 2) JSON 파일 리스트 미리 읽기
json_files = glob.glob(os.path.join(JSON_FOLDER, "*.json"))

for img_path in img_paths:
    img_name = os.path.basename(img_path)
    # 2) 매칭 JSON 찾기
    annotation = None
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('description',{}).get('image') == img_name:
            annotation = data
            break
    if annotation is None:
        print(f"[SKIP] JSON 없음: {img_name}")
        continue

    # 3) 바운딩박스 정보 (여러 점일 경우 모두 포함)
    pts = annotation.get('annotations',{}).get('points', [])
    xtl = min(p['xtl'] for p in pts)
    ytl = min(p['ytl'] for p in pts)
    xbr = max(p['xbr'] for p in pts)
    ybr = max(p['ybr'] for p in pts)

    # 4) 이미지 로드
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 5) ROI 계산 → 크롭
    xmin, ymin, xmax, ymax = compute_roi((xtl, ytl, xbr, ybr), w, h)
    crop = img[ymin:ymax, xmin:xmax]

    # 6) 720p로 리사이즈
    patch = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 7) 저장
    out_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(out_path, patch)
    print(f"[OK] Saved: {out_path}")

print("모든 이미지 처리 완료.")
