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
RANK_RE = re.compile(r'\b(10|[2-9TJQKAD])\b')

# Posições dos slots das cartas do board (espelha detectors.py)
SLOT_X       = [0.393, 0.453, 0.513, 0.580, 0.640]
CARD_Y1_F    = 0.34
CARD_Y2_F    = 0.56
CLASSIFY_HW_F = 0.028   # mais largo que o HW de detecção (0.020)

# Posição das hole cards do herói no crop de mesa (pixels absolutos, h=540)
# HERO_LEFT_X2=465 (+3px vs 462): necessário para OCR encontrar rank '2' de paus
# 'D'→'6' em OCR_SUBSTITUTIONS corrige leitura da fonte WPT para '6' de espadas
HERO_Y1, HERO_Y2          = 370, 435
HERO_LEFT_X1, HERO_LEFT_X2   = 430, 465
HERO_RIGHT_X1, HERO_RIGHT_X2 = 462, 495
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


OCR_SUBSTITUTIONS = {
    "Z": "2", "%": "2",
    "b": "6", "G": "6", "g": "6",
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

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    ocr = _get_ocr()

    candidates = [mid, full,
                  clahe.apply(mid), clahe.apply(full),
                  255 - mid, 255 - full]

    for img in candidates:
        img4x = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        img_bgr = cv2.cvtColor(img4x, cv2.COLOR_GRAY2BGR)

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
            if rank == "D":
                rank = "6"  # '6' misread as 'D' in WPT Global font (word-boundary safe)
            if rank == "10":
                rank = "T"
            if rank in VALID_RANKS:
                return rank

    return None


def classify_card(card_crop: np.ndarray) -> str | None:
    """
    Retorna notação padrão: 'Ac', 'Ks', '8h', etc.
    Retorna None se naipe ou rank não puderem ser determinados.

    Pipeline:
    1. CNN com conf >= 0.85: aceita direto (alta confiança)
    2. CNN com conf >= 0.70: verifica naipe via HSV e corrige se diferente
    3. Fallback: HSV (naipe) + OCR (rank)
    """
    try:
        from src.card_cnn import get_card_cnn
        cnn_card, conf = get_card_cnn().predict(card_crop)
        if conf >= 0.70 and len(cnn_card) == 2:
            cnn_rank = cnn_card[0].upper() if cnn_card[0] in 'tjqka' else cnn_card[0]
            cnn_suit = cnn_card[1]
            if conf >= 0.85:
                # Alta confiança: aceita direto
                return cnn_rank + cnn_suit
            # Confiança moderada: verifica naipe via HSV
            hsv_suit = detect_suit(card_crop)
            final_suit = hsv_suit if (hsv_suit != "?" and hsv_suit != cnn_suit) else cnn_suit
            return cnn_rank + final_suit
    except Exception:
        pass
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

    Pipeline por slot:
    1. CNN no crop centrado (conf >= 0.65) com verificação HSV de naipe
    2. Fallback: HSV (naipe) + OCR (rank)
    """
    if n_cards not in (3, 4, 5):
        return []

    h, w = table_crop.shape[:2]
    y1  = int(h * CARD_Y1_F)
    y2  = int(h * CARD_Y2_F)
    hw  = int(w * CLASSIFY_HW_F)

    RANK_LEFT_SHIFT = 25

    # Extrai os crops de cada slot
    slot_crops: list[np.ndarray | None] = []
    slot_sx: list[tuple[int, int]] = []
    for i in range(n_cards):
        cx  = int(w * SLOT_X[i])
        sx1 = max(0, cx - hw)
        sx2 = min(w, cx + hw)
        crop = table_crop[y1:y2, sx1:sx2]
        slot_crops.append(crop if crop.size > 0 else None)
        slot_sx.append((sx1, sx2))

    result: list[str | None] = [None] * n_cards

    # Tenta CNN em batch para todos os slots de uma vez
    cnn_preds: list[tuple[str, float]] | None = None
    try:
        from src.card_cnn import get_card_cnn
        _cnn = get_card_cnn()
        cnn_preds = _cnn.predict_batch([c for c in slot_crops])
    except Exception:
        pass

    for i in range(n_cards):
        cx = int(w * SLOT_X[i])
        suit_crop = slot_crops[i]

        # Tenta CNN
        if cnn_preds is not None and suit_crop is not None:
            cnn_card, conf = cnn_preds[i]
            if conf >= 0.65 and len(cnn_card) == 2:
                cnn_rank = cnn_card[0].upper() if cnn_card[0] in 'tjqka' else cnn_card[0]
                if cnn_rank in VALID_RANKS:
                    if conf < 0.85:
                        hsv_suit = detect_suit(suit_crop)
                        final_suit = hsv_suit if hsv_suit != "?" else cnn_card[1]
                    else:
                        final_suit = cnn_card[1]
                    result[i] = cnn_rank + final_suit
                    continue

        # Fallback: HSV + OCR
        if suit_crop is None:
            continue
        suit = detect_suit(suit_crop)
        if suit == "?":
            continue

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
    Coords absolutas — a posição y=370-435 é fixa para h=540 e h=502 porque
    a taskbar (38px) é cortada pela base, não pelo topo.
    Retorna lista de 0, 1 ou 2 cartas.

    Pipeline: tenta CNN primeiro no crop completo de cada carta;
    fallback para HSV (naipe) + OCR (rank) se CNN não atingir 0.65.
    """
    h, w = table_crop.shape[:2]
    y1 = min(HERO_Y1, h - 10)
    y2 = min(HERO_Y2, h)

    # Expande levemente a região para ter mais contexto para a CNN
    CNN_X_EXPAND = 10

    cards: list[str] = []
    for suit_x1, suit_x2 in [(HERO_LEFT_X1, HERO_LEFT_X2),
                               (HERO_RIGHT_X1, HERO_RIGHT_X2)]:
        # Crop expandido para CNN
        cnn_x1 = max(0, suit_x1 - CNN_X_EXPAND)
        cnn_x2 = min(w, suit_x2 + CNN_X_EXPAND)
        card_crop = table_crop[y1:y2, cnn_x1:cnn_x2]

        # Tenta CNN
        try:
            from src.card_cnn import get_card_cnn
            cnn_card, conf = get_card_cnn().predict(card_crop)
            if conf >= 0.65 and len(cnn_card) == 2:
                cnn_rank = cnn_card[0].upper() if cnn_card[0] in 'tjqka' else cnn_card[0]
                # Verifica naipe com HSV se confiança moderada
                if conf < 0.85:
                    suit_crop = table_crop[y1:y2, suit_x1:suit_x2]
                    hsv_suit = detect_suit(suit_crop)
                    final_suit = hsv_suit if hsv_suit != "?" else cnn_card[1]
                else:
                    final_suit = cnn_card[1]
                if cnn_rank in VALID_RANKS:
                    cards.append(cnn_rank + final_suit)
                    continue
        except Exception:
            pass

        # Fallback: HSV + OCR
        suit_crop = table_crop[y1:y2, suit_x1:suit_x2]
        suit = detect_suit(suit_crop)
        if suit == "?":
            continue

        rank = None
        for shift in [0, HERO_RANK_SHIFT, HERO_RANK_SHIFT * 2]:
            rx1 = max(0, suit_x1 - shift)
            rx2 = max(0, suit_x2 - shift)
            rank_crop = table_crop[y1:y2, rx1:rx2]
            rank = detect_rank(rank_crop)
            if rank:
                break
        if not rank:
            # Glyph may start 10px left of suit_x1 (e.g. Ah in HL2332)
            wide_crop = table_crop[y1:y2, max(0, suit_x1 - 10):suit_x2]
            rank = detect_rank(wide_crop)

        if rank:
            cards.append(rank + suit)

    return cards
