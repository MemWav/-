import re
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import JSON_DIR, IMG_DIR, CHECKPOINT_DIR
from dataset import PlantDataset, val_transform
from model.efficientnet import MultiTaskEfficientNet


def find_best_checkpoint(ckpt_dir):
    """
    CHECKPOINT_DIR 내 파일명에서 'acc{accuracy}' 부분을 추출해
    가장 높은 accuracy 파일 경로를 반환.
    예: best_epoch05_acc0.9234.pth → acc=0.9234
    """
    best_acc = -1.0
    best_path = None
    pattern = re.compile(r'acc([0-9]+\.[0-9]+)')
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith('.pth'):
            continue
        m = pattern.search(fn)
        if not m:
            continue
        acc = float(m.group(1))
        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(ckpt_dir, fn)
    if best_path is None:
        raise FileNotFoundError(f"No valid checkpoint in {ckpt_dir}")
    print(f">> Loading best checkpoint: {os.path.basename(best_path)} (acc={best_acc:.4f})")
    return best_path

class TestConfig:
    def __init__(self):
        self.batch_size = 1
        self.model_path = find_best_checkpoint(CHECKPOINT_DIR)
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_quantized_model(model_path, device):
    # 1) FP32 모델 로드
    base_model = MultiTaskEfficientNet()
    state = torch.load(model_path, map_location=device)
    base_model.load_state_dict(state)
    # 2) Dynamic Quantization
    q_model = base_model.quantize_model()
    q_model.to(device)
    q_model.eval()
    return q_model

def evaluate_model(model, loader, device):
    all_d_preds, all_d_labels = [], []
    all_g_preds, all_g_labels = [], []
    total_latency = 0.0
    n_samples = 0

    with torch.no_grad():
        for imgs, d_labels, g_labels in loader:
            imgs = imgs.to(device)
            d_labels = d_labels.to(device)
            g_labels = g_labels.to(device)

            start = time.time()
            d_logits, g_logits = model(imgs)
            torch.cuda.synchronize(device)  # 정확한 timing
            end = time.time()

            total_latency += (end - start) * 1000
            n_samples += imgs.size(0)

            d_preds = d_logits.argmax(dim=1).cpu().numpy()
            g_preds = g_logits.argmax(dim=1).cpu().numpy()
            all_d_preds.extend(d_preds)
            all_g_preds.extend(g_preds)
            all_d_labels.extend(d_labels.cpu().numpy())
            all_g_labels.extend(g_labels.cpu().numpy())

    # Metrics for each task
    def calc_metrics(preds, labels):
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average='weighted')
        cm  = confusion_matrix(labels, preds, labels=list(range(len(set(labels)))))
        fp_rates = []
        for i in range(cm.shape[0]):
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        return acc, f1, cm, fp_rates

    d_acc, d_f1, d_cm, d_fp = calc_metrics(all_d_preds, all_d_labels)
    g_acc, g_f1, g_cm, g_fp = calc_metrics(all_g_preds, all_g_labels)
    avg_latency = total_latency / n_samples if n_samples else 0.0

    return {
        'disease': {
            'accuracy': d_acc, 'f1_score': d_f1,
            'confusion_matrix': d_cm, 'false_positive_rates': d_fp
        },
        'growth': {
            'accuracy': g_acc, 'f1_score': g_f1,
            'confusion_matrix': g_cm, 'false_positive_rates': g_fp
        },
        'average_latency_ms': avg_latency
    }

def print_results(results):
    print("\n=== Disease Task ===")
    print(f"Accuracy: {results['disease']['accuracy']:.4f}")
    print(f"F1 Score:  {results['disease']['f1_score']:.4f}")
    print("False Positive Rates:", np.round(results['disease']['false_positive_rates'],4))
    print("Confusion Matrix:\n", results['disease']['confusion_matrix'])

    print("\n=== Growth Task ===")
    print(f"Accuracy: {results['growth']['accuracy']:.4f}")
    print(f"F1 Score:  {results['growth']['f1_score']:.4f}")
    print("False Positive Rates:", np.round(results['growth']['false_positive_rates'],4))
    print("Confusion Matrix:\n", results['growth']['confusion_matrix'])

    print(f"\nAverage Latency: {results['average_latency_ms']:.2f} ms per sample")

def main():
    cfg = TestConfig()
    # val_transform 은 dataset.py 에서 224×224로 정의해 둔 transform
    dataset = PlantDataset(JSON_DIR, IMG_DIR, transform=val_transform)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = load_quantized_model(cfg.model_path, cfg.device)
    results = evaluate_model(model, loader, cfg.device)
    print_results(results)

if __name__ == '__main__':
    main()
