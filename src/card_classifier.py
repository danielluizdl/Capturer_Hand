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

# Posição das hole cards do herói como frações do crop (h×w)
HERO_Y1_F         = 0.685   # 370/540
HERO_Y2_F         = 0.806   # 435/540
HERO_LX1_F        = 0.448   # 430/960
HERO_LX2_F        = 0.481   # 462/960
HERO_RX1_F        = 0.481   # 462/960
HERO_RX2_F        = 0.516   # 495/960
HERO_RANK_SHIFT_F = 0.016   # 15/960


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


OCR_SUBSTITUTIONS = {
    "Z": "2", "%": "2",
    "b": "6", "G": "6", "g": "6",
    "q": "9", "S": "5", "B": "8",
    "O": "0", "I": "1", "l": "1",
}


def detect_rank(card_crop: np.ndarray) -> str | None:
    """
    OCR na carta para extrair o rank.
    O rank no WPT Global aparece no centro vertical da carta (y≈45-55%),
    não no topo. Tenta 6x e 4x upscale para cartas pequenas (~50px).
    Retorna rank normalizado: '2'-'9', 'T', 'J', 'Q', 'K', 'A' ou None.
    """
    if card_crop is None or card_crop.size == 0:
        return None

    h, w = card_crop.shape[:2]
    gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)

    full   = gray
    mid_y1 = max(0, int(h * 0.35))
    mid_y2 = min(h, int(h * 0.75))
    mid    = gray[mid_y1:mid_y2, :]
    top_left = gray[0:int(h * 0.50), 0:int(w * 0.60)]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    ocr = _get_ocr()

    candidates = [mid, full, top_left,
                  clahe.apply(mid), clahe.apply(full), clahe.apply(top_left),
                  255 - mid, 255 - full, 255 - top_left]

    for img in candidates:
        for scale in [6, 4]:
            img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img_bgr = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)

            result, _ = ocr(img_bgr)
            if not result:
                continue

            all_text = (
                " ".join(r[1] for r in result if r[2] >= 0.15)
                .upper()
                .replace("10", "T")
            )
            for old, new in OCR_SUBSTITUTIONS.items():
                all_text = all_text.replace(old, new)

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
    Retorna lista com até n_cards cartas (pode ter menos se algum slot falhar).
    """
    slotted = extract_board_cards_slotted(table_crop, n_cards)
    return [c for c in slotted if c is not None]


def extract_board_cards_slotted(table_crop: np.ndarray, n_cards: int) -> list[str | None]:
    """
    Como extract_board_cards, mas retorna lista de tamanho n_cards com None
    em slots onde a detecção falhou. Permite votação independente por slot.
    """
    if n_cards not in (3, 4, 5):
        return []

    h, w = table_crop.shape[:2]
    y1  = int(h * CARD_Y1_F)
    y2  = int(h * CARD_Y2_F)
    hw  = int(w * CLASSIFY_HW_F)

    RANK_LEFT_SHIFT = 25

    result: list[str | None] = [None] * n_cards
    for i in range(n_cards):
        cx  = int(w * SLOT_X[i])

        # --- Naipe: crop centrado ---
        sx1 = max(0, cx - hw)
        sx2 = min(w, cx + hw)
        suit_crop = table_crop[y1:y2, sx1:sx2]
        suit = detect_suit(suit_crop)
        if suit == "?":
            continue

        # --- Rank: crop deslocado para a esquerda ---
        rx1 = max(0, cx - hw - RANK_LEFT_SHIFT)
        rx2 = min(w, cx + hw - RANK_LEFT_SHIFT)
        rank_crop = table_crop[y1:y2, rx1:rx2]
        rank = detect_rank(rank_crop)

        if rank:
            result[i] = rank + suit

    return result


def extract_hole_cards(table_crop: np.ndarray) -> list[str]:
    """
    Extrai hole cards do herói (dLzinN) na posição inferior central.
    Usa coordenadas fracionais para suportar alturas de mesa variáveis (540 vs 502px).
    Retorna lista de 0, 1 ou 2 cartas.
    """
    h, w = table_crop.shape[:2]
    y1    = int(h * HERO_Y1_F)
    y2    = int(h * HERO_Y2_F)
    lx1   = int(w * HERO_LX1_F)
    lx2   = int(w * HERO_LX2_F)
    rx1   = int(w * HERO_RX1_F)
    rx2   = int(w * HERO_RX2_F)
    shift = int(w * HERO_RANK_SHIFT_F)

    cards: list[str] = []
    for suit_x1, suit_x2 in [(lx1, lx2), (rx1, rx2)]:
        suit_crop = table_crop[y1:y2, suit_x1:suit_x2]
        suit = detect_suit(suit_crop)
        if suit == "?":
            continue

        rank = None
        for s in [0, shift, shift * 2]:
            rx1_ = max(0, suit_x1 - s)
            rx2_ = max(0, suit_x2 - s)
            rank_crop = table_crop[y1:y2, rx1_:rx2_]
            rank = detect_rank(rank_crop)
            if rank:
                break

        if rank:
            cards.append(rank + suit)

    return cards
