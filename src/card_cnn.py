"""Inferencia CNN para classificacao de cartas (52 classes)."""
from __future__ import annotations
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path

_MODEL   = Path(__file__).parent.parent / 'models' / 'card_classifier.pt'
_CLASSES = Path(__file__).parent.parent / 'models' / 'card_classes.json'

_PRE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


class _CardNet(nn.Module):
    def __init__(self, num_classes=52):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CardCNN:
    def __init__(self):
        with open(_CLASSES) as f:
            self.classes = json.load(f)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = _CardNet(52)
        m.load_state_dict(torch.load(_MODEL, map_location=self.device))
        m.eval()
        self.model = m.to(self.device)
        print(f'[CardCNN] carregado ({self.device})')

    def predict(self, img_bgr: np.ndarray) -> tuple[str, float]:
        if img_bgr is None or img_bgr.size == 0:
            return '', 0.0
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = _PRE(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)[0]
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])

    def predict_top3(self, img_bgr: np.ndarray) -> list[tuple[str,float]]:
        if img_bgr is None or img_bgr.size == 0:
            return []
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = _PRE(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)[0]
        top = probs.topk(3)
        return [(self.classes[i], float(p)) for i,p in zip(top.indices, top.values)]

    def predict_batch(
        self,
        imgs_bgr: list[np.ndarray],
    ) -> list[tuple[str, float]]:
        """
        Inferência em batch para múltiplos crops de uma vez.
        Mais eficiente que chamar predict() N vezes individualmente.
        Retorna lista de (card, confidence) na mesma ordem de imgs_bgr.
        """
        results: list[tuple[str, float]] = []
        valid_indices: list[int] = []
        tensors: list = []

        for i, img in enumerate(imgs_bgr):
            if img is None or img.size == 0:
                results.append(('', 0.0))
            else:
                try:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensors.append(_PRE(rgb))
                    valid_indices.append(i)
                    results.append(('', 0.0))  # placeholder
                except Exception:
                    results.append(('', 0.0))

        if tensors:
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(batch), dim=1)
            for j, orig_i in enumerate(valid_indices):
                idx = int(probs[j].argmax())
                results[orig_i] = (self.classes[idx], float(probs[j][idx]))

        return results


_cnn: CardCNN | None = None


def get_card_cnn() -> CardCNN:
    global _cnn
    if _cnn is None:
        _cnn = CardCNN()
    return _cnn
