"""
Card classifier para WPT Global / Nexa Poker.
Usa cor HSV do fundo para naipe + OCR para rank.

Cores confirmadas por análise visual direta de frames (mesa HL3458):
  ♣ paus    = VERDE vivo   (H 55-90,  S>100, V>80)
  ♦ ouros   = AZUL vivo    (H 95-130, S>80,  V>80)
  ♥ copas   = VERMELHO     (H 0-15 ou H 165-180, S>80, V>80)
  ♠ espadas = ESCURO/PRETO (V < SPADES_V_THRESHOLD)
"""
from __future__ import annotations
import re
import cv2
import numpy as np

# Limiar de brilho para espadas (tunable pelo agent_loop)
SPADES_V_THRESHOLD = 90

VALID_RANKS = set("23456789TJQKA")
RANK_RE = re.compile(r'\b(10|[2-9TJQKA])\b')

# Posições dos slots das cartas do board (espelha detectors.py)
SLOT_X       = [0.393, 0.453, 0.513, 0.580, 0.640]
CARD_Y1_F    = 0.34
CARD_Y2_F    = 0.56
CLASSIFY_HW_F = 0.028   # mais largo que o HW de detecção (0.020)

# Posição das hole cards do herói no crop de mesa (pixels)
HERO_Y1, HERO_Y2          = 430, 520
HERO_LEFT_X1, HERO_LEFT_X2   = 345, 415
HERO_RIGHT_X1, HERO_RIGHT_X2 = 425, 495


def _get_ocr():
    """Reutiliza instância RapidOCR de ocr_engine para não duplicar."""
    from src.ocr_engine import _ocr
    return _ocr()


def detect_suit(card_crop: np.ndarray) -> str:
    """
    Retorna 'c'/'d'/'h'/'s' pela cor de fundo do crop da carta.
    Retorna '?' se não conseguir determinar.
    """
    if card_crop is None or card_crop.size == 0:
        return "?"

    hsv = cv2.cvtColor(card_crop, cv2.COLOR_BGR2HSV)
    h, w = card_crop.shape[:2]

    # Região central (evita bordas)
    cy1, cy2 = max(0, h // 4), min(h, 3 * h // 4)
    cx1, cx2 = max(0, w // 4), min(w, 3 * w // 4)
    center = hsv[cy1:cy2, cx1:cx2]
    if center.size == 0:
        return "?"

    mean_v = float(np.mean(center[:, :, 2]))
    mean_s = float(np.mean(center[:, :, 1]))
    mean_h = float(np.mean(center[:, :, 0]))

    # Espadas: escuro (baixo brilho)
    if mean_v < SPADES_V_THRESHOLD:
        return "s"

    # Dessaturado mas claro → incerto, tratar como espadas
    if mean_s < 50:
        return "s"

    # Usar matiz para paus / ouros / copas
    if 55 <= mean_h <= 90:
        return "c"   # verde = paus
    if 90 < mean_h <= 135:
        return "d"   # azul = ouros
    # Copas: vermelho (H < 55 ou H > 135)
    return "h"


def detect_rank(card_crop: np.ndarray) -> str | None:
    """
    OCR na carta para extrair o rank.
    Tenta múltiplas estratégias de pré-processamento.
    Retorna rank normalizado: '2'-'9', 'T', 'J', 'Q', 'K', 'A' ou None.
    """
    if card_crop is None or card_crop.size == 0:
        return None

    h, w = card_crop.shape[:2]
    gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
    top  = gray[:max(1, h // 2), :]   # rank está no topo

    # Estratégia 1: limiar alto — texto branco em fundo colorido
    _, thresh_white = cv2.threshold(top, 170, 255, cv2.THRESH_BINARY)
    # Estratégia 2: CLAHE para realce de contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(top)
    # Estratégia 3: carta inteira, limiar médio
    _, thresh_full = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    ocr = _get_ocr()

    for img in [thresh_white, enhanced, thresh_full, top]:
        # Upscale para melhor OCR (cartas são pequenas ~55px)
        scale = max(1, 64 // max(img.shape[0], 1))
        if scale > 1:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        result, _ = ocr(img_bgr)
        if not result:
            continue

        all_text = (
            " ".join(r[1] for r in result if r[2] >= 0.3)
            .upper()
            .replace("10", "T")
        )
        m = RANK_RE.search(all_text)
        if m:
            rank = m.group(1).upper()
            if rank == "10":
                rank = "T"
            if rank in VALID_RANKS:
                return rank

    return None


def classify_card(card_crop: np.ndarray) -> str | None:
    """
    Retorna notação padrão: 'Ac', 'Ks', '8h', etc.
    Retorna None se naipe ou rank não puderem ser determinados.
    """
    suit = detect_suit(card_crop)
    if suit == "?":
        return None
    rank = detect_rank(card_crop)
    if not rank:
        return None
    return rank + suit


def extract_board_cards(table_crop: np.ndarray, n_cards: int) -> list[str]:
    """
    Extrai cartas do board usando posições dos slots.
    n_cards deve ser 3, 4 ou 5.
    """
    if n_cards not in (3, 4, 5):
        return []

    h, w = table_crop.shape[:2]
    y1  = int(h * CARD_Y1_F)
    y2  = int(h * CARD_Y2_F)
    hw  = int(w * CLASSIFY_HW_F)

    cards: list[str] = []
    for i in range(n_cards):
        cx  = int(w * SLOT_X[i])
        sx1 = max(0, cx - hw)
        sx2 = min(w, cx + hw)
        card_crop = table_crop[y1:y2, sx1:sx2]
        card = classify_card(card_crop)
        if card:
            cards.append(card)

    return cards


def extract_hole_cards(table_crop: np.ndarray) -> list[str]:
    """
    Extrai hole cards do herói (dLzinN) na posição inferior central.
    Retorna lista de 0, 1 ou 2 cartas.
    """
    h, _w = table_crop.shape[:2]
    y1 = min(HERO_Y1, h - 10)
    y2 = min(HERO_Y2, h)

    left_crop  = table_crop[y1:y2, HERO_LEFT_X1:HERO_LEFT_X2]
    right_crop = table_crop[y1:y2, HERO_RIGHT_X1:HERO_RIGHT_X2]

    cards: list[str] = []
    for crop in [left_crop, right_crop]:
        card = classify_card(crop)
        if card:
            cards.append(card)

    return cards
