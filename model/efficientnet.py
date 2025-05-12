import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from config import NUM_DISEASE, NUM_GROW

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, effname='efficientnet-b0'):
        super().__init__()
        # 1) pretrained EfficientNet 백본
        self.backbone = EfficientNet.from_pretrained(effname)
        in_feat = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()

        # 2) 각 태스크 별 헤드
        self.disease_head = nn.Linear(in_feat, NUM_DISEASE)
        self.grow_head    = nn.Linear(in_feat, NUM_GROW)

    def forward(self, x):
        # EfficientNet 특징 추출
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.flatten(1)

        # 로짓 반환 (disease_logits, grow_logits)
        return self.disease_head(x), self.grow_head(x)