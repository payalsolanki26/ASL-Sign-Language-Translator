"""
Model zoo  (Deep Learning)

asl_net            custom residual CNN with squeeze-excite; small enough to train on CPU/laptop GPU
                   and to run on-device. ~2.8 M parameters (run this file to print the exact count).
mobilenet_v3_small torchvision backbone (optionally ImageNet-pretrained) with a new 27-way head.

Input  : (B, 3, 128, 128)  ImageNet-normalised
Output : (B, 27) logits
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(ch // r, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(cout)
        self.se = SqueezeExcite(cout)
        self.skip = (nn.Identity() if (cin == cout and stride == 1) else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.se(self.bn2(self.conv2(y)))
        return self.act(y + self.skip(x))


class AslNet(nn.Module):
    def __init__(self, num_classes: int = 27, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64))        # 64 -> 32
        self.stage2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))     # 32 -> 16
        self.stage3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))    # 16 -> 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.head(self.pool(x))

    def cam_layer(self):
        return self.stage3[-1].bn2          # last conv feature map before SE / pooling


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "asl_net":
        return AslNet(m["num_classes"], m["dropout"])
    if m["name"] == "mobilenet_v3_small":
        from torchvision import models
        net = models.mobilenet_v3_small(weights="DEFAULT" if m.get("pretrained") else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, m["num_classes"])
        net.cam_layer = lambda: net.features[-1]
        return net
    raise ValueError(f"unknown model {m['name']}")


if __name__ == "__main__":
    net = AslNet()
    n = sum(p.numel() for p in net.parameters())
    print(f"AslNet parameters: {n:,}  output shape: {tuple(net(torch.zeros(2, 3, 128, 128)).shape)}")
