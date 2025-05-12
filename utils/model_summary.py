# model_summary.py

import torch
from model.efficientnet import MultiTaskEfficientNet

# 1) FP32 모델 인스턴스 생성
model = MultiTaskEfficientNet()

# 2) 전체 모델 구조 출력
print(model)

from torchinfo import summary

# 3) 224×224 입력 기준으로 레이어별 출력 shape & 파라미터 수 보기
summary(model, input_size=(1, 3, 224, 224), depth=4, col_names=["input_size","output_size","num_params"])
