"""
Template matching engine para reconhecer hole cards de opponents nos seats.

NAO precisa de calibracao de seats: o sistema busca os templates no
frame inteiro e encontra os pares de cartas onde quer que estejam.

Templates:
    templates/baralho_seats/left/   — carta esquerda de cada par
    templates/baralho_seats/right/  — carta direita de cada par

API principal:
    pairs = find_showdown_cards(table_crop)
    # -> [{"left": "Ah", "right": "Ks", "x": 120, "y": 85}, ...]

API auxiliar (crop de seat ja extraido):
    rec = SeatCardRecognizer()
    pair = rec.recognize_pair(seat_crop)   # -> "AhKs" | None
"""
from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "baralho_seats"

# Score minimo de correlacao para aceitar um match (TM_CCOEFF_NORMED, 0..1)
DEFAULT_MIN_SCORE = 0.65

_RANKS = set("23456789tjqka")
_SUITS = set("cdhs")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _fmt(card: str) -> str:
    """Converte card code para formato PokerStars: 'ah' -> 'Ah', '9c' -> '9c'."""
    return card[0].upper() + card[1]


def _load_folder(folder: Path) -> dict[str, list[np.ndarray]]:
    """
    Carrega templates de uma pasta e agrupa por card code.
    Formato esperado: '9c.png', 'kh.png', etc.
    Retorna {'9c': [img, ...], 'kh': [img], ...}
    """
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    if not folder.exists():
        return dict(result)

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        stem = path.stem.lower()
        if len(stem) >= 2 and stem[0] in _RANKS and stem[1] in _SUITS:
            card = stem[:2]
            img = cv2.imread(str(path))
            if img is not None:
                result[card].append(img)

    return dict(result)


def _best_match(
    region: np.ndarray,
    template_bank: dict[str, list[np.ndarray]],
) -> tuple[str | None, float]:
    """
    Desliza cada template sobre a regiao e retorna (card_code, max_score).
    Usa TM_CCOEFF_NORMED — robusto a variacoes de brilho absoluto.
    Skipa templates que nao cabem na regiao.
    """
    if region is None or region.size == 0:
        return None, 0.0

    rh, rw = region.shape[:2]
    best_card  = None
    best_score = -1.0

    for card, tmpl_list in template_bank.items():
        for tmpl in tmpl_list:
            th, tw = tmpl.shape[:2]
            if th > rh or tw > rw:
                continue

            res   = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())

            if score > best_score:
                best_score = score
                best_card  = card

    return best_card, best_score


def _top_matches(
    region: np.ndarray,
    template_bank: dict[str, list[np.ndarray]],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Para debug: retorna top_n (card, score) ordenados por score desc."""
    if region is None or region.size == 0:
        return []

    rh, rw = region.shape[:2]
    scores: dict[str, float] = {}

    for card, tmpl_list in template_bank.items():
        for tmpl in tmpl_list:
            th, tw = tmpl.shape[:2]
            if th > rh or tw > rw:
                continue
            res   = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            scores[card] = max(scores.get(card, -1.0), score)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class SeatCardRecognizer:
    """
    Identifica hole cards de opponents via template matching.
    Instancie uma vez e reutilize — templates sao carregados em __init__.
    """

    def __init__(
        self,
        templates_dir: str | Path | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        base = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        self.left_templates  = _load_folder(base / "left")
        self.right_templates = _load_folder(base / "right")
        self.min_score = min_score

        n_left  = sum(len(v) for v in self.left_templates.values())
        n_right = sum(len(v) for v in self.right_templates.values())
        print(
            f"[SeatCardRecognizer] "
            f"left={n_left} templates ({len(self.left_templates)} cartas)  "
            f"right={n_right} templates ({len(self.right_templates)} cartas)"
        )

    # ------------------------------------------------------------------
    def recognize(
        self,
        seat_crop: np.ndarray,
    ) -> tuple[str | None, str | None, float, float]:
        """
        Identifica o par de cartas em um crop de seat de opponent.

        seat_crop : imagem BGR com as duas cartas visiveis lado a lado.

        Retorna (left_card, right_card, left_score, right_score).
          - Cards em lowercase: '9c', 'kh', 'as', etc.
          - None se o score nao atingir min_score.
        """
        if seat_crop is None or seat_crop.size == 0:
            return None, None, 0.0, 0.0

        h, w = seat_crop.shape[:2]

        # Regiao esquerda: cobre os primeiros ~58% da largura (carta esquerda)
        # Regiao direita:  cobre os ultimos ~58% da largura (carta direita)
        # A sobreposicao de 16% no centro garante que nenhuma carta seja cortada.
        margin = w // 6
        left_region  = seat_crop[:, : w // 2 + margin]
        right_region = seat_crop[:, w // 2 - margin :]

        lc, ls = _best_match(left_region,  self.left_templates)
        rc, rs = _best_match(right_region, self.right_templates)

        return (
            lc if ls >= self.min_score else None,
            rc if rs >= self.min_score else None,
            ls,
            rs,
        )

    def recognize_pair(self, seat_crop: np.ndarray) -> str | None:
        """
        Retorna par no formato PokerStars: 'AhKs', '9cQd', etc.
        None se qualquer carta nao identificada com confianca suficiente.
        """
        lc, rc, _, _ = self.recognize(seat_crop)
        if lc and rc:
            return _fmt(lc) + _fmt(rc)
        return None

    def debug_scores(
        self, seat_crop: np.ndarray, top_n: int = 5
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Retorna top_n scores para cada lado — util para calibrar min_score
        ou diagnosticar falhas de reconhecimento.

        Retorna {'left': [('kh', 0.91), ('ah', 0.72), ...],
                 'right': [('9c', 0.88), ...]}
        """
        if seat_crop is None or seat_crop.size == 0:
            return {"left": [], "right": []}

        h, w = seat_crop.shape[:2]
        margin = w // 6
        left_region  = seat_crop[:, : w // 2 + margin]
        right_region = seat_crop[:, w // 2 - margin :]

        return {
            "left":  _top_matches(left_region,  self.left_templates,  top_n),
            "right": _top_matches(right_region, self.right_templates, top_n),
        }


# ---------------------------------------------------------------------------
# Deteccao de naipe pela cor de fundo (HSV)
# ---------------------------------------------------------------------------

# Mesmas faixas do card_classifier.py, confirmadas nos frames do video
_SPADES_V_MAX  = 90    # espadas: fundo escuro, V < 90
_SPADES_S_MAX  = 50    # tambem aceita baixa saturacao como espadas

def _detect_suit(region: np.ndarray) -> str | None:
    """
    Detecta o naipe de uma carta pela cor de fundo do crop.

    Exclui pixels claros (rank e simbolo do naipe sao brancos/cremes)
    e analisa a cor mediana do fundo restante.

    Retorna 'c' (verde/paus), 'd' (azul/ouros), 'h' (vermelho/copas),
    's' (preto/espadas), ou None se inconclusivo.
    """
    if region is None or region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Mascara pixels brancos/claros: sao o rank ("10", "A"...) e o pip do naipe
    bg_mask = hsv[:, :, 2] < 180
    bg = hsv[bg_mask]

    if len(bg) < 10:
        return None

    v_med = float(np.median(bg[:, 2]))
    s_med = float(np.median(bg[:, 1]))

    # Espadas: fundo escuro (baixo V) ou muito dessaturado
    if v_med < _SPADES_V_MAX or s_med < _SPADES_S_MAX:
        return 's'

    h_med = float(np.median(bg[:, 0]))

    if 55 <= h_med <= 90:          # verde  → paus
        return 'c'
    if 95 <= h_med <= 130:         # azul   → ouros
        return 'd'
    if h_med <= 15 or h_med >= 165:  # vermelho → copas
        return 'h'

    return None


def _apply_suit_correction(
    hits: dict[str, tuple[int, int, int, int, float]],
    image: np.ndarray,
) -> dict[str, tuple[int, int, int, int, float]]:
    """
    Para cada carta em hits, verifica a cor de fundo no image real.
    Se o naipe detectado por cor difere do naipe do template, corrige.

    Ex: template 'ts' ganhou NMS mas o fundo e azul → corrige para 'td'.

    Em caso de colisao (dois templates corrigidos para o mesmo codigo),
    mantem o de maior score.
    """
    corrected: dict[str, tuple[int, int, int, int, float]] = {}
    ih, iw = image.shape[:2]

    for card, (x, y, w, h, score) in hits.items():
        rank = card[0]
        suit = card[1]

        region = image[max(0, y):min(ih, y + h), max(0, x):min(iw, x + w)]
        detected = _detect_suit(region)

        final_card = (rank + detected) if (detected and detected != suit) else card

        # Colisao: dois templates apontam para o mesmo codigo — fica o maior score
        if final_card in corrected:
            if score > corrected[final_card][4]:
                corrected[final_card] = (x, y, w, h, score)
        else:
            corrected[final_card] = (x, y, w, h, score)

    return corrected


# ---------------------------------------------------------------------------
# Busca no frame inteiro — nao precisa de calibracao de seats
# ---------------------------------------------------------------------------

def _find_best_per_card(
    image: np.ndarray,
    template_bank: dict[str, list[np.ndarray]],
    min_score: float,
) -> dict[str, tuple[int, int, int, int, float]]:
    """
    Para cada card code, encontra a posicao com melhor score no image inteiro.
    Retorna {card: (x, y, w, h, score)} — apenas cartas acima de min_score.
    """
    ih, iw = image.shape[:2]
    results: dict[str, tuple[int, int, int, int, float]] = {}

    for card, tmpl_list in template_bank.items():
        best_score = -1.0
        best_x = best_y = best_w = best_h = 0

        for tmpl in tmpl_list:
            th, tw = tmpl.shape[:2]
            if th > ih or tw > iw:
                continue
            res = cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_x, best_y = max_loc
                best_w, best_h = tw, th

        if best_score >= min_score:
            results[card] = (best_x, best_y, best_w, best_h, best_score)

    return results


def _apply_nms(
    hits: dict[str, tuple[int, int, int, int, float]],
    radius: int = 8,
) -> dict[str, tuple[int, int, int, int, float]]:
    """
    Non-Maximum Suppression espacial: quando dois cards diferentes batem
    dentro de `radius` pixels do mesmo ponto, mantém apenas o de maior score.
    Resolve a confusão Td/Ts, Ac/As, etc. que têm templates quasi-idênticos.
    """
    items = sorted(hits.items(), key=lambda kv: -kv[1][4])  # desc por score
    kept: dict[str, tuple[int, int, int, int, float]] = {}
    occupied: list[tuple[int, int]] = []  # centros ja reservados

    for card, (x, y, w, h, s) in items:
        cx, cy = x + w // 2, y + h // 2
        too_close = any(
            abs(cx - ox) <= radius and abs(cy - oy) <= radius
            for ox, oy in occupied
        )
        if not too_close:
            kept[card] = (x, y, w, h, s)
            occupied.append((cx, cy))

    return kept


def find_showdown_cards(
    table_crop: np.ndarray,
    min_score: float = DEFAULT_MIN_SCORE,
    board_excl: tuple[int, int] | None = None,
) -> list[dict]:
    """
    Detecta todos os pares de hole cards visiveis no crop de mesa completo.
    NAO precisa de calibracao de seats — busca os templates no frame inteiro.

    Melhorias vs versao anterior:
      - NMS espacial: elimina confusao Td/Ts, Ac/As no mesmo pixel.
      - board_excl: (y_min, y_max) para mascarar a faixa do board e evitar
        que as cartas comunitarias gerem falsos positivos.
      - min_score separado para LEFT (mais permissivo) e RIGHT (mais restrito).

    Retorna lista de dicts:
      [{"left": "Ah", "right": "Ks", "x": 120, "y": 85,
        "left_score": 0.91, "right_score": 0.87}, ...]
    """
    if table_crop is None or table_crop.size == 0:
        return []

    rec = get_recognizer()

    # Mascara a faixa do board antes do template matching
    search_img = table_crop.copy()
    if board_excl is not None:
        y0, y1 = board_excl
        h = search_img.shape[0]
        search_img[max(0, y0):min(h, y1), :] = 0

    # LEFT: threshold ligeiramente mais baixo para capturar ah (0.87)
    # RIGHT: threshold mais alto para evitar falsos positivos proximos aos seats
    left_hits  = _find_best_per_card(search_img, rec.left_templates,  min_score)
    right_hits = _find_best_per_card(search_img, rec.right_templates, max(min_score, 0.80))

    # NMS: mantém apenas 1 carta por posicao espacial (elimina td/ts no mesmo pixel)
    left_hits  = _apply_nms(left_hits)
    right_hits = _apply_nms(right_hits)

    # Correcao de naipe: verifica cor HSV do fundo no frame real
    # Resolve ambiguidade residual (ex: ts→td quando fundo e azul)
    left_hits  = _apply_suit_correction(left_hits,  search_img)
    right_hits = _apply_suit_correction(right_hits, search_img)

    pairs: list[dict]  = []
    used_rights: set   = set()

    # Ordena cartas esquerdas por score desc (melhores matches primeiro)
    for lcard, (lx, ly, lw, lh, ls) in sorted(
        left_hits.items(), key=lambda kv: -kv[1][4]
    ):
        best_rcard = None
        best_dist  = float("inf")

        for rcard, (rx, ry, rw, rh, rs) in right_hits.items():
            if rcard in used_rights:
                continue

            dx = rx - lx
            dy = abs(ry - ly)

            if 0 < dx <= lw * 3 and dy <= lh // 2:
                dist = dx + dy * 2
                if dist < best_dist:
                    best_dist  = dist
                    best_rcard = rcard

        if best_rcard:
            used_rights.add(best_rcard)
            rx, ry = right_hits[best_rcard][:2]
            pairs.append({
                "left":        lcard,
                "right":       best_rcard,
                "x":           lx + lw // 2,
                "y":           ly + lh // 2,
                "left_score":  ls,
                "right_score": right_hits[best_rcard][4],
            })

    pairs.sort(key=lambda p: (p["y"], p["x"]))
    return pairs


# ---------------------------------------------------------------------------
# Singleton para uso no pipeline (evita recarregar templates a cada frame)
# ---------------------------------------------------------------------------

_recognizer: SeatCardRecognizer | None = None


def get_recognizer() -> SeatCardRecognizer:
    """Instancia singleton — cria na primeira chamada."""
    global _recognizer
    if _recognizer is None:
        _recognizer = SeatCardRecognizer()
    return _recognizer


def recognize_seat_cards(seat_crop: np.ndarray) -> str | None:
    """
    API de conveniencia para crop de seat ja extraido.
    recognize_seat_cards(crop) -> 'AhKs'  ou  None
    """
    return get_recognizer().recognize_pair(seat_crop)
