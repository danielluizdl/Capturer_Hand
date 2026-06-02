import json, random
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

BASE = Path('.')
FOLDERS = [
    BASE / 'templates/baralho',
    BASE / 'templates/baralho_seats/left',
    BASE / 'templates/baralho_seats/right',
]
Path('models').mkdir(exist_ok=True)

RANKS = '23456789tjqka'
SUITS = 'cdhs'
ALL_CARDS = sorted([r+s for r in RANKS for s in SUITS])
card2idx = {c: i for i, c in enumerate(ALL_CARDS)}

raw_images, raw_labels = [], []
for folder in FOLDERS:
    for p in sorted(folder.glob('*.png')):
        card = p.stem[:2].lower()
        if card not in card2idx: continue
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        raw_images.append(img)
        raw_labels.append(card2idx[card])

print(f'Base: {len(raw_images)} imagens, {len(set(raw_labels))} classes')

aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.85,1.15)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class CardDataset(Dataset):
    def __init__(self, images, labels, transform, n_aug=60):
        self.data = []
        for img, lbl in zip(images, labels):
            for _ in range(n_aug):
                self.data.append((img, lbl))
        self.tf = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        img, lbl = self.data[i]
        return self.tf(img), lbl

n = len(raw_images)
idx = list(range(n))
random.shuffle(idx)
sp = int(0.8*n)
tr_idx, va_idx = idx[:sp], idx[sp:]

tr_ds = CardDataset([raw_images[i] for i in tr_idx], [raw_labels[i] for i in tr_idx], aug, 60)
va_ds = CardDataset([raw_images[i] for i in va_idx], [raw_labels[i] for i in va_idx], val_tf, 1)
tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0)
print(f'Treino: {len(tr_ds)} | Val: {len(va_ds)}')

class CardNet(nn.Module):
    def __init__(self, num_classes=52):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = CardNet(52)
model = model.to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
crit = nn.CrossEntropyLoss()

best_val = 1e9
for ep in range(1, 101):
    model.train()
    tr_ok = tr_tot = 0
    for xb, yb in tr_dl:
        xb, yb = xb.to(DEVICE), torch.tensor(yb).to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        tr_ok += (out.argmax(1)==yb).sum().item()
        tr_tot += len(yb)
    sched.step()
    model.eval()
    va_ok = va_tot = va_loss = 0
    with torch.no_grad():
        for xb, yb in va_dl:
            xb, yb = xb.to(DEVICE), torch.tensor(yb).to(DEVICE)
            out = model(xb)
            va_loss += crit(out,yb).item()*len(yb)
            va_ok += (out.argmax(1)==yb).sum().item()
            va_tot += len(yb)
    if ep%10==0 or ep==1:
        print(f'Ep {ep:3d}: tr={tr_ok/tr_tot*100:.1f}% val={va_ok/va_tot*100:.1f}%')
    if va_tot and va_loss/va_tot < best_val:
        best_val = va_loss/va_tot
        torch.save(model.state_dict(), 'models/card_classifier.pt')

with open('models/card_classes.json','w') as f:
    json.dump(ALL_CARDS, f)
print('Salvo: models/card_classifier.pt e models/card_classes.json')
