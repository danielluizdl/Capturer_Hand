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

# Posição das hole cards do herói no crop de mesa (pixels absolutos, h=540)
# Calibrado: '94' encontrado em bbox [441,389,490,424]; Ah detectado em 430:462
HERO_Y1, HERO_Y2          = 370, 435
HERO_LEFT_X1, HERO_LEFT_X2   = 430, 462
HERO_RIGHT_X1, HERO_RIGHT_X2 = 462, 495
# Shift para rank: texto fica ~15px à esquerda do início do crop
HERO_RANK_SHIFT = 15


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
    O rank no WPT Global aparece no centro vertical da carta (y≈45-55%),
    não no topo. Usa 4x upscale fixo para cartas pequenas.
    Retorna rank normalizado: '2'-'9', 'T', 'J', 'Q', 'K', 'A' ou None.
    """
    if card_crop is None or card_crop.size == 0:
        return None

    h, w = card_crop.shape[:2]
    gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)

    # O rank está em ~45% da altura — usar a carta inteira para não cortar
    full   = gray
    # Faixa central onde o rank aparece (evita pot-text acima e suit abaixo)
    mid_y1 = max(0, int(h * 0.35))
    mid_y2 = min(h, int(h * 0.75))
    mid    = gray[mid_y1:mid_y2, :]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    ocr = _get_ocr()

    # Inclui inversão para cartas escuras (espadas) onde texto é claro sobre fundo preto
    candidates = [mid, full, clahe.apply(mid), clahe.apply(full),
                  255 - mid, 255 - full]

    for img in candidates:
        # 4x upscale fixo — cartas são pequenas (~50px)
        img4x = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        img_bgr = cv2.cvtColor(img4x, cv2.COLOR_GRAY2BGR)

        result, _ = ocr(img_bgr)
        if not result:
            continue

        all_text = (
            " ".join(r[1] for r in result if r[2] >= 0.25)
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
    Usa crops separados para naipe (centrado) e rank (deslocado à esquerda).
    Retorna lista de 0, 1 ou 2 cartas.
    """
    h, w = table_crop.shape[:2]
    y1 = min(HERO_Y1, h - 10)
    y2 = min(HERO_Y2, h)

    cards: list[str] = []
    for suit_x1, suit_x2 in [(HERO_LEFT_X1, HERO_LEFT_X2),
                               (HERO_RIGHT_X1, HERO_RIGHT_X2)]:
        suit_crop = table_crop[y1:y2, suit_x1:suit_x2]
        suit = detect_suit(suit_crop)
        if suit == "?":
            continue

        # Rank: tenta sem shift primeiro, depois com shift de 15px
        rank = None
        for shift in [0, HERO_RANK_SHIFT, HERO_RANK_SHIFT * 2]:
            rx1 = max(0, suit_x1 - shift)
            rx2 = max(0, suit_x2 - shift)
            rank_crop = table_crop[y1:y2, rx1:rx2]
            rank = detect_rank(rank_crop)
            if rank:
                break

        if rank:
            cards.append(rank + suit)

    return cards
