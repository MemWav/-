import re
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import TEST_JSON_DIR, TEST_IMG_DIR, CHECKPOINT_DIR, DISEASE_CLASSES, GROW_STAGES
from dataset import PlantDataset, val_transform
from model.efficientnet import MultiTaskEfficientNet
from utils.find_best_checkpoint import find_best_checkpoint

class TestConfig:
    def __init__(self):
        self.batch_size = 1
        self.model_path = find_best_checkpoint(CHECKPOINT_DIR)
        # self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device     = torch.device('cpu')

def load_model(model_path, device):
    # FP32 멀티태스크 EfficientNet 로드
    model = MultiTaskEfficientNet()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, loader, device):
    all_d_preds, all_d_labels = [], []
    all_g_preds, all_g_labels = [], []
    total_latency = 0.0
    n_samples = 0

    with torch.no_grad():
        for imgs, d_labels, g_labels in tqdm(loader):
            imgs = imgs.to(device)
            d_labels = d_labels.to(device)
            g_labels = g_labels.to(device)

            start = time.time()
            d_logits, g_logits = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end = time.time()

            total_latency += (end - start) * 1000
            n_samples += imgs.size(0)

            d_preds = d_logits.argmax(dim=1).cpu().numpy()
            g_preds = g_logits.argmax(dim=1).cpu().numpy()
            all_d_preds.extend(d_preds)
            all_g_preds.extend(g_preds)
            all_d_labels.extend(d_labels.cpu().numpy())
            all_g_labels.extend(g_labels.cpu().numpy())

    def calc_metrics(preds, labels, model_type=''):
        # config 에 따라 전체 레이블 리스트 결정
        if model_type == 'disease':
            label_list = list(DISEASE_CLASSES.keys())   # [0,1,…,20]
        elif model_type == 'grow':
            label_list = list(GROW_STAGES.keys())       # [0,1,2]
        else:
            # fallback: 실제 등장하는 라벨만
            label_list = sorted(set(labels))

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average='weighted')
        cm  = confusion_matrix(labels, preds, labels=label_list)

        fp_rates = []
        for i, _ in enumerate(label_list):
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

        return acc, f1, cm, fp_rates


    print(f'unique d_labels: {sorted(set(all_d_labels))}')
    print(f'unique g_labels: {sorted(set(all_g_labels))}')
    d_acc, d_f1, d_cm, d_fp = calc_metrics(all_d_preds, all_d_labels, model_type='')
    g_acc, g_f1, g_cm, g_fp = calc_metrics(all_g_preds, all_g_labels, model_type='grow')
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
    dataset = PlantDataset(TEST_JSON_DIR, TEST_IMG_DIR, transform=val_transform)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = load_model(cfg.model_path, cfg.device)
    results = evaluate_model(model, loader, cfg.device)
    print_results(results)

if __name__ == '__main__':
    main()
