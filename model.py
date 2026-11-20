"""ResNet-34 speaker embedding extractor (WeSpeaker layout) and weight loading."""

import argparse
import hashlib
import io
import math
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PRETRAINED_URL = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/avg_model.pt"
PRETRAINED_SHA256 = "ac00dd664205fdec58e5c6768b1dfbb732225a1e69486f89461e5f88d39d8d22"


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34(nn.Module):
    """Thin ResNet-34 over (frames, mel bins) features with mean+std temporal pooling."""

    def __init__(self, feat_dim=80, embed_dim=256, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        widths = [channels, 2 * channels, 4 * channels, 8 * channels]
        in_planes = channels
        for i, (width, blocks) in enumerate(zip(widths, [3, 4, 6, 3])):
            stride = 1 if i == 0 else 2
            layers = [BasicBlock(in_planes if j == 0 else width, width, stride if j == 0 else 1) for j in range(blocks)]
            setattr(self, f"layer{i + 1}", nn.Sequential(*layers))
            in_planes = width
        self.seg_1 = nn.Linear(2 * widths[-1] * math.ceil(feat_dim / 8), embed_dim)

    def forward(self, feats):
        """(batch, frames, feat_dim) -> (batch, embed_dim), not normalised."""
        x = feats.transpose(1, 2).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = x.flatten(1, 2)
        stats = torch.cat([x.mean(dim=-1), torch.sqrt(x.var(dim=-1) + 1e-7)], dim=1)
        return self.seg_1(stats)


def load_model(weights, embed_dim=256):
    """ResNet-34 from a WeSpeaker-format state dict; any classification head is dropped."""
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("projection.")}
    model = ResNet34(embed_dim=embed_dim)
    model.load_state_dict(state)
    return model


def download_pretrained(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PRETRAINED_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PRETRAINED_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest} does not match the published WeSpeaker checkpoint")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 2**20


def main():
    parser = argparse.ArgumentParser(description="Download and verify the pretrained weights, print model stats.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())["model"]
    download_pretrained(cfg["pretrained_weights"])
    model = load_model(cfg["pretrained_weights"], cfg["embed_dim"])
    print(f"{cfg['architecture']}: {count_parameters(model):,} parameters, {model_size_mb(model):.1f} MB")


if __name__ == "__main__":
    main()
