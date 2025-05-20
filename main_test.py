import os
import time
import cv2
import torch
from PIL import Image
from torchvision import transforms
from config import DEVICE, CHECKPOINT_DIR, DISEASE_CLASSES, GROW_STAGES
from model.efficientnet import MultiTaskEfficientNet
from utils.find_best_checkpoint import find_best_checkpoint

# -------------------------------------------------------------------
# 모델 로드 (기존 함수 그대로)
# -------------------------------------------------------------------
def load_model(device):
    model = MultiTaskEfficientNet()
    ckpt = find_best_checkpoint(CHECKPOINT_DIR)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f">> Loaded checkpoint: {os.path.basename(ckpt)}")
    return model

# -------------------------------------------------------------------
# 전처리 (기존 그대로)
# -------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# -------------------------------------------------------------------
# 한 프레임 예측 (threshold 적용)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 메인: 실시간 피드, 'c' 캡처→예측, 'q' 종료
# -------------------------------------------------------------------
def main(threshold=0.75):
    # 모델 로드
    model = load_model(DEVICE)

    # 캡처 저장 폴더
    os.makedirs("captures", exist_ok=True)

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 열기 실패")
        return

    print("실시간 피드 중... 'c' 키: 캡처&예측, 'q' 키: 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        # 도움말 텍스트
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # 타임스탬프 및 파일명
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_path = f"captures/{timestamp}.jpg"
            cv2.imwrite(img_path, frame)

            # 예측
            disease, d_conf, grow, g_conf = predict_frame(
                model, frame, DEVICE, threshold
            )
            result_label = (f"Disease: {disease} ({d_conf:.2f}), "
                            f"Stage: {grow} ({g_conf:.2f})")

            # 콘솔 출력
            print(f"[{timestamp}] {result_label}")

            # 오버레이된 캡처 창에 결과 보여주기
            output = frame.copy()
            cv2.putText(output, result_label, (10, output.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Captured Prediction", output)
            cv2.waitKey(0)
            cv2.destroyWindow("Captured Prediction")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(threshold=0.75)
