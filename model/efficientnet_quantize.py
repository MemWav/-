import torch
import torch.nn as nn
import torch.quantization as quant
from efficientnet_pytorch import EfficientNet
from config import NUM_DISEASE, NUM_GROW

# Set quantization engine for ARM devices (e.g., Raspberry Pi)
torch.backends.quantized.engine = 'qnnpack'

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, effname='efficientnet-b0'):
        super().__init__()
        # Pretrained EfficientNet backbone
        self.backbone = EfficientNet.from_pretrained(effname)
        in_feat = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()

        # Task-specific heads
        self.disease_head = nn.Linear(in_feat, NUM_DISEASE)
        self.grow_head    = nn.Linear(in_feat, NUM_GROW)

    def forward(self, x):
        # Feature extraction
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.flatten(1)
        return self.disease_head(x), self.grow_head(x)

    def fuse_model(self):
        # Placeholder for layer fusion if needed (e.g., Conv+BN).
        # EfficientNet-PyTorch does not expose standard fuse patterns,
        # so we skip explicit fusing here.
        pass

    def quantize_model(self):
        """
        Post-training dynamic quantization: only Linear layers to INT8.
        Conv layers remain FP32.
        """
        self.eval()
        qmodel = quant.quantize_dynamic(
            self,
            {nn.Linear},
            dtype=torch.qint8
        )
        return qmodel
