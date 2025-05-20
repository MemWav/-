import os
import time
import cv2
import torch
from PIL import Image
from torchvision import transforms
from config import DEVICE, CHECKPOINT_DIR, DISEASE_CLASSES, GROW_STAGES
from model.efficientnet import MultiTaskEfficientNet
from utils.find_best_checkpoint import find_best_checkpoint

# 1) 모델 로드 함수 (기존)
def load_model(device):
    model = MultiTaskEfficientNet()
    ckpt = find_best_checkpoint(CHECKPOINT_DIR)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f">> Loaded checkpoint: {os.path.basename(ckpt)}")
    return model

# 2) 전처리 파이프라인 (기존)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# 3) 한 프레임에 대해 예측 수행 (threshold 지정 가능)
def predict_frame(model, frame, device, threshold=0.7):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        d_logits, g_logits = model(x)
        d_probs = torch.softmax(d_logits, dim=1)
        g_probs = torch.softmax(g_logits, dim=1)
        d_prob, d_idx = d_probs.max(dim=1)
        g_prob, g_idx = g_probs.max(dim=1)

    disease = DISEASE_CLASSES[d_idx] if d_prob.item() >= threshold else "Unknown"
    grow    = GROW_STAGES[g_idx]     if g_prob.item() >= threshold else "Unknown"
    return disease, d_prob.item(), grow, g_prob.item()

# 4) 메인: 10분마다 캡처 → 예측 → 로그
def main(interval_sec=600, threshold=0.75):
    model = load_model(DEVICE)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 열기 실패") 
        return

    # 캡처 저장용 폴더
    os.makedirs("captures", exist_ok=True)

    while True:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ret, frame = cap.read()
        if not ret:
            print(f"[{timestamp}] 프레임 캡처 실패")
        else:
            # 이미지 저장 (선택)
            img_path = f"captures/{timestamp}.jpg"
            cv2.imwrite(img_path, frame)

            # 예측
            disease, d_conf, grow, g_conf = predict_frame(
                model, frame, DEVICE, threshold
            )

            # 결과 출력 또는 파일로 기록
            log = (f"[{timestamp}] Disease: {disease} ({d_conf:.2f}), "
                   f"Stage: {grow} ({g_conf:.2f})")
            print(log)
            with open("captures/prediction_log.txt", "a") as f:
                f.write(log + "\n")

        # 10분 대기
        time.sleep(interval_sec)

    cap.release()

if __name__ == "__main__":
    # interval_sec=600(10분), threshold=0.75(75% 이상일 때만)
    main(interval_sec=600, threshold=0.75)
