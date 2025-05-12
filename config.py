# config.py

import os
import torch

# ── 기본 경로 설정 ─────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
TRAIN_JSON_DIR         = os.path.join(BASE_DIR, 'data', 'train', 'json')
TRAIN_IMG_DIR          = os.path.join(BASE_DIR, 'data', 'train', 'image')
TEST_JSON_DIR         = os.path.join(BASE_DIR, 'data', 'test', 'json')
TEST_IMG_DIR          = os.path.join(BASE_DIR, 'data', 'test', 'image')
CHECKPOINT_DIR   = os.path.join(BASE_DIR, 'checkpoints')   # <-- 추가

# ── 하이퍼파라미터 ────────────────────────────────────────
NUM_DISEASE      = 21    # JSON에 정의된 disease 클래스 수로 수정
NUM_GROW         = 3    # JSON에 정의된 grow 단계 수로 수정
BATCH_SIZE       = 32
LR               = 1e-4
EPOCHS           = 20

# ── 장치 설정 ─────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ── 클래스 이름 맵핑 ───────────────────────────────────────
# AI-Hub '시설 작물 질병 진단 이미지' 데이터셋 기준 (0~20)
DISEASE_CLASSES = {
    0:  '정상',
    1:  '가지잎곰팡이병',
    2:  '가지흰가루병',
    3:  '고추마일드모틀바이러스',
    4:  '고추점무늬병',
    5:  '단호박점무늬병',
    6:  '단호박흰가루병',
    7:  '딸기잿빛곰팡이병',
    8:  '딸기흰가루병',
    9:  '상추균핵병',
   10:  '상추노균병',
   11:  '수박탄저병',
   12:  '수박흰가루병',
   13:  '애호박점무늬병',
   14:  '오이녹반모자이크바이러스',
   15:  '오이모자이크바이러스',
   16:  '참외노균병',
   17:  '참외흰가루병',
   18:  '토마토잎곰팡이병',
   19:  '토마토황화잎말이바이러스',
   20:  '포도노균병'
}

# 원본 grow 코드 11(육묘기),12(생장기),13(착화/과실기) → 0~2로 변환
GROW_STAGES = {
    0: '육묘기',
    1: '생장기',
    2: '착화/과실기'
}