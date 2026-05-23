"""Task 2 — wrapper RapidOCR para layout WPT Global / Nexa Poker."""
from __future__ import annotations
import re
import numpy as np

_ocr_instance = None


def _ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_instance = RapidOCR()
    return _ocr_instance


def _raw_ocr(img: np.ndarray) -> list[tuple]:
    """Roda OCR e retorna [(bbox, text, conf), ...] filtrando conf < 0.5."""
    result, _ = _ocr()(img)
    if not result:
        return []
    return [(r[0], r[1], r[2]) for r in result if r[2] >= 0.5]


def ocr_title_bar(crop: np.ndarray) -> str | None:
    """
    OCR na barra de titulo. Retorna 'HL4017' ou None.
    ROI: primeiros 25px de altura, 500px de largura.
    """
    roi = crop[0:25, 0:500]
    items = _raw_ocr(roi)
    for _, text, _ in items:
        m = re.search(r'(HL\d+)', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: OCR no crop completo (parte superior)
    roi2 = crop[0:40, 0:600]
    items2 = _raw_ocr(roi2)
    for _, text, _ in items2:
        m = re.search(r'(HL\d+)', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def ocr_pot(crop: np.ndarray) -> float | None:
    """
    OCR na area do pot. Retorna valor em BB como float (converte virgula PT-BR).
    ROI: y=155:210, x=230:750.
    """
    roi = crop[155:210, 230:750]
    items = _raw_ocr(roi)
    all_text = " ".join(t for _, t, _ in items)

    # Formato: "PoteTotal:21,5BB" ou "Pote Total : 21,5 BB"
    m = re.search(r'[Pp]ote\s*[Tt]otal\s*:?\s*([\d,\.]+)', all_text)
    if m:
        val = m.group(1).replace(",", ".")
        try:
            return float(val)
        except ValueError:
            pass

    # Fallback: qualquer numero seguido de BB
    m = re.search(r'([\d,\.]{2,})\s*BB', all_text, re.IGNORECASE)
    if m:
        val = m.group(1).replace(",", ".")
        try:
            return float(val)
        except ValueError:
            pass

    return None


def ocr_stacks(crop: np.ndarray) -> dict[str, float]:
    """
    OCR de stacks dos jogadores. Retorna {nome: chips_bb}.
    Varre a imagem inteira em busca de padrões XX,XXBB.
    """
    result: dict[str, float] = {}

    # Zonas conhecidas onde aparecem jogadores e seus stacks
    regions = [
        crop[0:540, 0:960],   # full crop
    ]

    for roi in regions:
        items = _raw_ocr(roi)
        # Processa pares consecutivos: nome, valor BB
        texts = [(t, c) for _, t, c in items]
        for i, (text, _) in enumerate(texts):
            m = re.search(r'([\d,\.]+)\s*BB', text, re.IGNORECASE)
            if m:
                val = m.group(1).replace(",", ".")
                try:
                    bb = float(val)
                except ValueError:
                    continue
                # Tenta associar com nome anterior
                if i > 0:
                    prev_text = texts[i - 1][0].strip()
                    if prev_text and not re.search(r'[\d,\.]', prev_text):
                        result[prev_text] = bb
                elif i + 1 < len(texts):
                    next_text = texts[i + 1][0].strip()
                    if next_text and not re.search(r'[\d,\.]', next_text):
                        result[next_text] = bb

    return result


def ocr_board_cards(crop: np.ndarray, n_cards: int) -> list[str]:
    """
    Extrai cartas do board via classificador cor+OCR.
    Fallback para lista vazia se classifier retornar incompleto.
    """
    if n_cards == 0:
        return []
    from src.card_classifier import extract_board_cards
    return extract_board_cards(crop, n_cards)


def ocr_hole_cards(crop: np.ndarray) -> list[str]:
    """Extrai hole cards do herói via classificador cor+OCR."""
    from src.card_classifier import extract_hole_cards
    return extract_hole_cards(crop)


def ocr_winner(crop: np.ndarray) -> str | None:
    """
    Detecta o vencedor pela proximidade espacial ao badge 'WINNER'.
    - Encontra bounding box de 'WINNER'
    - Retorna o nome de jogador mais próximo espacialmente
    - Fallback: 'dLzinN' se WINNER estiver na zona do herói (y > 80%)
    """
    import numpy as np

    result, _ = _ocr()(crop)
    if not result:
        return None

    h, w = crop.shape[:2]

    # Textos de UI a ignorar
    UI_SKIP = {
        'LOG DE JOGOS', 'Carta alta', 'Carta Alta', 'GongFuBoy...', 'POKER', 'NEXA',
        'JACKPOT', 'FLIPS', 'MASTER', 'SPF', 'PONER', 'MAOS', 'VPNP',
        'Desistir', 'Passar', 'Aposta', 'Aumentar', 'Pagar', 'Verificar',
        'LOG', 'JOGOS', 'Passar/Desistir', 'Mostrar', 'Rabbit', 'Hunt',
        'Carta', 'alta', 'BB', 'PoteTotal', 'Pote', 'Total',
    }

    def _center(bbox):
        return (
            float(np.mean([p[0] for p in bbox])),
            float(np.mean([p[1] for p in bbox])),
        )

    winner_pos = None
    candidates: list[tuple[str, float, float]] = []  # (name, cx, cy)

    for r in result:
        bbox, tx, conf = r[0], r[1], r[2]
        if conf < 0.7:
            continue
        tx_clean = tx.strip()

        if tx_clean.upper() in ('WINNER', 'VENCEDOR'):
            winner_pos = _center(bbox)
            continue

        # Candidato a nome de jogador: letras+dígitos, 3-20 chars, não é UI
        if len(tx_clean) >= 3 and len(tx_clean) <= 25:
            # Exclui se é puramente numérico ou BB
            if re.fullmatch(r'[\d,\.]+\s*BB?', tx_clean, re.I):
                continue
            if re.fullmatch(r'[\d,\.]+', tx_clean):
                continue
            # Exclui palavras de UI conhecidas
            skip = False
            for ui in UI_SKIP:
                if ui.lower() in tx_clean.lower():
                    skip = True
                    break
            if skip:
                continue
            # Deve ter pelo menos 1 letra
            if not any(c.isalpha() for c in tx_clean):
                continue
            cx, cy = _center(bbox)
            candidates.append((tx_clean, cx, cy))

    if winner_pos is None:
        return None

    wx, wy = winner_pos

    # Hero zone: bottom-center do layout WPT Global (y > 85%, x em 35-65%)
    # Quando o herói (dLzinN) ganha, WINNER aparece sobre a posição dele.
    if wy / h > 0.85 and 0.35 < wx / w < 0.65:
        return "dLzinN"

    # Remove '...' e '…' do final de nomes truncados
    cleaned_candidates = [
        (name.rstrip('.').rstrip('…').strip(), cx, cy)
        for name, cx, cy in candidates
    ]
    cleaned_candidates = [(n, cx, cy) for n, cx, cy in cleaned_candidates if n]

    if not cleaned_candidates:
        return None

    # Encontra o nome de jogador mais próximo
    best_name, best_dist = None, float("inf")
    for name, cx, cy in cleaned_candidates:
        dist = ((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = name

    # Se muito longe (> 350px), provavelmente não é o vencedor
    if best_dist > 350:
        return None

    return best_name


# --- helpers ---

SUIT_MAP = {
    "♠": "s", "♣": "c", "♥": "h", "♦": "d",
    "♡": "h", "♢": "d", "♤": "s", "♧": "c",
    "s": "s", "c": "c", "h": "h", "d": "d",
    "S": "s", "C": "c", "H": "h", "D": "d",
}

VALID_RANKS = set("23456789TJQKA")


def _parse_cards_from_text(text: str) -> list[str]:
    """
    Extrai cartas de texto livre. Tenta múltiplos formatos:
    - "Jd 5s 2h" (standard notation)
    - "J♦ 5♠ 2♥" (unicode suits)
    - Ranks e suits separados
    """
    cards: list[str] = []
    seen: set[str] = set()

    # Normaliza: remove espaços extras
    text = text.replace("10", "T")

    # Pattern 1: rank+suit direto (Jd, 5s, Ah, etc.)
    for m in re.finditer(r'([2-9TJQKA])([sShHdDcC♠♣♥♦♡♢♤♧])', text):
        rank = m.group(1).upper()
        suit_raw = m.group(2)
        suit = SUIT_MAP.get(suit_raw, None)
        if suit and rank in VALID_RANKS:
            card = rank + suit
            if card not in seen:
                seen.add(card)
                cards.append(card)

    # Pattern 2: rank e suit separados por espaco
    if not cards:
        tokens = text.split()
        i = 0
        while i < len(tokens) - 1:
            rank = tokens[i].upper()
            suit_raw = tokens[i + 1]
            if rank in VALID_RANKS and suit_raw in SUIT_MAP:
                card = rank + SUIT_MAP[suit_raw]
                if card not in seen:
                    seen.add(card)
                    cards.append(card)
                i += 2
                continue
            i += 1

    return cards
