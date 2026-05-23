"""
Calibra a cor do feltro e testa deteccao de cartas por variancia intra-slot.
"""
import cv2
import numpy as np
import os

VIDEO  = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\video_cortado_1min.mp4'
OUTDIR = r'c:\Users\danie\Documents\Projetos\Capturer_Hand\frames_preview'

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

# Slots de carta (x relativo ao crop 960x540)
X_CENTERS = [0.345, 0.405, 0.465, 0.525, 0.585]
CARD_Y1   = 0.34
CARD_Y2   = 0.56
SLOT_HW   = 0.022  # meia-largura


def sample_slots(crop, verbose=False):
    """Retorna (mean_bgr, std_gray) para cada slot."""
    h, w = crop.shape[:2]
    y1 = int(h * CARD_Y1)
    y2 = int(h * CARD_Y2)
    results = []
    for p in X_CENTERS:
        cx = int(w * p)
        sx1, sx2 = max(0, cx - int(w * SLOT_HW)), min(w, cx + int(w * SLOT_HW))
        slot = crop[y1:y2, sx1:sx2]
        mean_bgr = slot.reshape(-1, 3).mean(axis=0)
        std_gray  = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY).std()
        if verbose:
            print(f"  x={p:.3f}  BGR=({mean_bgr[0]:.0f},{mean_bgr[1]:.0f},{mean_bgr[2]:.0f})  std={std_gray:.1f}")
        results.append((mean_bgr, std_gray))
    return results


# 1. Amostra cor nos frames conhecidos
print("=== FRAMES COM BOARD VAZIO (preflop) ===")
for t in [0.5, 33.0, 60.0]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if not ret:
        continue
    crop = frame[0:540, 0:960]
    print(f"\nt={t}s Mesa TL:")
    sample_slots(crop, verbose=True)

print("\n=== FRAMES COM BOARD COM CARTAS ===")
for t in [4.0, 10.0, 20.0]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if not ret:
        continue
    crop = frame[0:540, 0:960]
    print(f"\nt={t}s Mesa TL:")
    sample_slots(crop, verbose=True)

# 2. Varre o video e plota variancia ao longo do tempo para mesa TL
print("\n=== VARIANCIA POR SLOT AO LONGO DO TEMPO (mesa TL, 2fps) ===")
print(f"{'t':>5} | {'slot1':>6} | {'slot2':>6} | {'slot3':>6} | {'slot4':>6} | {'slot5':>6} | total")
print('-' * 55)

skip = max(1, round(fps / 2.0))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_num = 0
prev_total = -1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % skip == 0:
        ts = frame_num / fps
        crop = frame[0:540, 0:960]
        slots = sample_slots(crop)
        stds = [s[1] for s in slots]
        # Conta slots com variancia alta = carta presente
        # Threshold a calibrar:
        card_detected = [s > 18 for s in stds]
        total = sum(card_detected)
        # So imprime se mudou
        if total != prev_total:
            flags = ' '.join(['C' if d else '.' for d in card_detected])
            print(f"{ts:5.1f} | {stds[0]:6.1f} | {stds[1]:6.1f} | {stds[2]:6.1f} | {stds[3]:6.1f} | {stds[4]:6.1f} | {total}  [{flags}]")
            prev_total = total
    frame_num += 1

cap.release()
print("\nC = carta detectada, . = slot vazio")
print("Se total mudar 0->3 = flop, 3->4 = turn, 4->5 = river")
