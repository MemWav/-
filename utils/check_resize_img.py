import os
import glob
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# 설정
IMAGE_FOLDER = 'data/train/image'    # 이미지 폴더 경로
RESIZE_SIZE  = 224                   # 타깃 해상도

# 리사이즈 + ToTensor 변환 정의
transform = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.ToTensor(),           # (C, H, W), 값 [0.0, 1.0]
])

def check_images(folder, max_checks=5):
    """
    폴더 내 이미지들을 순회하며,
      - 리사이즈된 텐서의 shape, dtype, min/max/mean
      - 시각화
    """
    img_path = "data/test/image/c5_d0_000002.jpg"
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img)

    # 상태 출력
    print(f"  shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  min: {tensor.min():.3f}, max: {tensor.max():.3f}, mean: {tensor.mean():.3f}\n")

    # 시각화
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(np_img)
    plt.title(os.path.basename(img_path))
    plt.axis('off')
    plt.show()
        

# 실행
check_images(IMAGE_FOLDER)
