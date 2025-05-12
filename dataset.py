import os, glob, json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PlantDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None):
        self.json_files = glob.glob(os.path.join(json_dir, '*.json'))
        self.img_dir     = img_dir
        self.transform   = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # JSON에서 레이블만 읽어옴
        with open(self.json_files[idx], 'r') as f:
            ann = json.load(f)
        img_name = ann['description']['image']

        # 이미지 로드 (이미 1280×720으로 준비됨)
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Transform 적용
        if self.transform:
            img = self.transform(img)

        # 레이블 tensor 변환
        disease = ann['annotations']['disease']
        grow    = ann['annotations']['grow']
        return img, torch.tensor(disease, dtype=torch.long), torch.tensor(grow, dtype=torch.long)
    

# ── 데이터 변환 정의 ────────────────────────────────────────
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
