"""Task 3 — detectores visuais calibrados para 10fps. Baseado em detectors_final.py."""
from __future__ import annotations
import cv2
import numpy as np

# Quadrantes 960x540 dentro do frame 1920x1080
TABLE_REGIONS = [
    (0,    0,   960,  540),   # 0 = TL → HL3458
    (960,  0,  1920,  540),   # 1 = TR → HL4017
    (0,   540,  960, 1080),   # 2 = BL → HL2332
    (960, 540, 1920, 1080),   # 3 = BR → HL3048
]

# Altura util de cada mesa (exclui taskbar Windows nas mesas de baixo)
TABLE_USEFUL_H = [540, 540, 502, 502]

# Posições das 5 cartas comunitárias como fração de x=960px.
# Flop: 0.393, 0.453, 0.513 | Turn: 0.580 | River: 0.640
# (slot 0.333 era gap — nunca dispara; substituído por 0.640 para river)
SLOT_X          = [0.393, 0.453, 0.513, 0.580, 0.640]
CARD_Y1_F       = 0.34
CARD_Y2_F       = 0.56
SLOT_HW_F       = 0.020
SLOT_THRESHOLDS = [25, 33, 38, 25, 30]

ACTION_BAR_TOP_F = 0.900


def crop_table(frame: np.ndarray, tid: int) -> np.ndarray:
    """Retorna crop da mesa, respeitando altura util (sem taskbar)."""
    x1, y1, x2, y2 = TABLE_REGIONS[tid]
    uh = TABLE_USEFUL_H[tid]
    return frame[y1: y1 + uh, x1:x2]


def count_board_cards(crop: np.ndarray) -> int:
    """
    Conta cartas visiveis no board usando variancia por slot.
    Retorna 0, 3, 4 ou 5 (valores validos em poker).
    Valores 1 e 2 sao descartados como ruido do logo NEXA.
    """
    h, w = crop.shape[:2]
    y1 = int(h * CARD_Y1_F)
    y2 = int(h * CARD_Y2_F)
    hw = int(w * SLOT_HW_F)
    total = 0
    for xf, thr in zip(SLOT_X, SLOT_THRESHOLDS):
        cx  = int(w * xf)
        sx1 = max(0, cx - hw)
        sx2 = min(w, cx + hw)
        patch = crop[y1:y2, sx1:sx2]
        if patch.size == 0:
            continue
        std = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std())
        if std > thr:
            total += 1
    return total if total in (0, 3, 4, 5) else 0


def has_action_buttons(crop: np.ndarray) -> tuple[bool, float]:
    """
    Detecta botoes Desistir/Pagar/Aumentar.
    Busca por regioes VERDES (Aumentar), cinza-azulado (Pagar) ou vermelhas (Desistir).
    """
    h, w = crop.shape[:2]
    bar  = crop[int(h * ACTION_BAR_TOP_F):, :]
    if bar.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([55,  60,  50]), np.array([90,  255, 200]))
    blue  = cv2.inRange(hsv, np.array([95,  40,  60]), np.array([130, 160, 180]))
    red1  = cv2.inRange(hsv, np.array([0,  100,  80]), np.array([8,   255, 200]))
    red2  = cv2.inRange(hsv, np.array([172, 100,  80]), np.array([180, 255, 200]))

    mask  = cv2.bitwise_or(green, cv2.bitwise_or(blue, cv2.bitwise_or(red1, red2)))
    score = float(np.sum(mask > 0) / mask.size)
    return score > 0.015, score


def detect_pot_change(prev: np.ndarray, curr: np.ndarray) -> tuple[bool, float]:
    """Detecta mudanca na area de texto do pot."""
    h, w = curr.shape[:2]
    y1, y2 = int(h * 0.28), int(h * 0.36)
    x1, x2 = int(w * 0.28), int(w * 0.72)
    diff = float(np.mean(np.abs(
        curr[y1:y2, x1:x2].astype(float) - prev[y1:y2, x1:x2].astype(float)
    )))
    return diff > 4.0, diff
