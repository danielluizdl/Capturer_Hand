"""Task 3 — detectores visuais calibrados para 10fps. Baseado em detectors_final.py."""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Cache de templates carregados
_tmpl_cache: dict[str, np.ndarray | None] = {}

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

    Usa threshold adaptativo: em cenas escuras (mesa = brilho médio < 60),
    os thresholds sao reduzidos 20% pois as cartas se destacam menos.
    """
    h, w = crop.shape[:2]
    y1 = int(h * CARD_Y1_F)
    y2 = int(h * CARD_Y2_F)
    hw = int(w * SLOT_HW_F)

    # Threshold adaptativo baseado no brilho medio da faixa do board
    board_region = crop[y1:y2, :]
    mean_brightness = float(cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY).mean())
    adapt_factor = 0.80 if mean_brightness < 60 else 1.0

    total = 0
    for xf, thr in zip(SLOT_X, SLOT_THRESHOLDS):
        cx  = int(w * xf)
        sx1 = max(0, cx - hw)
        sx2 = min(w, cx + hw)
        patch = crop[y1:y2, sx1:sx2]
        if patch.size == 0:
            continue
        std = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std())
        if std > thr * adapt_factor:
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


def _load_template(name: str) -> np.ndarray | None:
    """Carrega template PNG/PNG lazy com cache."""
    if name not in _tmpl_cache:
        for ext in (".PNG", ".png"):
            p = _TEMPLATES_DIR / (name + ext)
            if p.exists():
                _tmpl_cache[name] = cv2.imread(str(p))
                break
        else:
            _tmpl_cache[name] = None
    return _tmpl_cache[name]


def detect_dealer_button(crop: np.ndarray) -> tuple[float, float] | None:
    """
    Detecta a posição do dealer button (puck) no crop de mesa.
    Usa template matching contra dealer.PNG.
    Retorna (cx, cy) em fração do crop (0-1), ou None se não encontrado.
    Score mínimo: 0.55.
    """
    tmpl = _load_template("dealer")
    if tmpl is None:
        return None
    ih, iw = crop.shape[:2]
    th, tw = tmpl.shape[:2]
    if th > ih or tw > iw:
        return None
    res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.55:
        return None
    cx = (max_loc[0] + tw / 2) / iw
    cy = (max_loc[1] + th / 2) / ih
    return (cx, cy)


def detect_allin(crop: np.ndarray) -> bool:
    """Detecta se há um badge ALL-IN visível na mesa via template matching."""
    tmpl = _load_template("allin")
    if tmpl is None:
        return False
    ih, iw = crop.shape[:2]
    th, tw = tmpl.shape[:2]
    if th > ih or tw > iw:
        return False
    res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) >= 0.65


def detect_action_type(crop: np.ndarray) -> str | None:
    """
    Detecta qual ação está sendo exibida pelos botões na action bar.
    Usa template matching nos templates de ação disponíveis.

    Retorna: 'fold' | 'call' | 'raise' | 'check' | 'allin' | None
    Score mínimo: 0.60.
    """
    action_map = {
        "desistir":      "fold",
        "pagar":         "call",
        "aumentar":      "raise",
        "passar":        "check",
        "allin":         "allin",
        "mostrar_cartas": None,   # não é ação de aposta
    }

    h = crop.shape[0]
    bar = crop[int(h * ACTION_BAR_TOP_F):, :]
    if bar.size == 0:
        return None

    best_action = None
    best_score = 0.60

    for tmpl_name, action in action_map.items():
        if action is None:
            continue
        tmpl = _load_template(tmpl_name)
        if tmpl is None:
            continue
        th, tw = tmpl.shape[:2]
        bh, bw = bar.shape[:2]
        if th > bh or tw > bw:
            continue
        res = cv2.matchTemplate(bar, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(res.max())
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def detect_pot_change(prev: np.ndarray, curr: np.ndarray) -> tuple[bool, float]:
    """Detecta mudanca na area de texto do pot."""
    h, w = curr.shape[:2]
    y1, y2 = int(h * 0.28), int(h * 0.36)
    x1, x2 = int(w * 0.28), int(w * 0.72)
    diff = float(np.mean(np.abs(
        curr[y1:y2, x1:x2].astype(float) - prev[y1:y2, x1:x2].astype(float)
    )))
    return diff > 4.0, diff
