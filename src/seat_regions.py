"""
Coordenadas dos seats e do hero por mesa, em espaco de crop.

O crop de cada mesa tem tamanho (frame_w//2, frame_h//2).
Exemplo: frame 1920x1032 → crop 960x516.

Gerado por:  python calibrate_seats.py video.mp4
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Hero — area das hole cards do hero (ambas as cartas juntas)
# (x1, y1, x2, y2) em espaco de crop da mesa
# ---------------------------------------------------------------------------
HERO_REGIONS: dict[int, tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 0),
    2: (0, 0, 0, 0),
    3: (0, 0, 0, 0),
}

# ---------------------------------------------------------------------------
# Opponents — hole cards visiveis no showdown
# {table_idx: {seat_idx: (x1, y1, x2, y2)}}
# ---------------------------------------------------------------------------
SEAT_REGIONS: dict[int, dict[int, tuple[int, int, int, int]]] = {
    0: {2:(0,0,0,0), 3:(0,0,0,0), 4:(0,0,0,0), 5:(0,0,0,0), 6:(0,0,0,0), 7:(0,0,0,0), 8:(0,0,0,0)},
    1: {2:(0,0,0,0), 3:(0,0,0,0), 4:(0,0,0,0), 5:(0,0,0,0), 6:(0,0,0,0), 7:(0,0,0,0), 8:(0,0,0,0)},
    2: {2:(0,0,0,0), 3:(0,0,0,0), 4:(0,0,0,0), 5:(0,0,0,0), 6:(0,0,0,0), 7:(0,0,0,0), 8:(0,0,0,0)},
    3: {2:(0,0,0,0), 3:(0,0,0,0), 4:(0,0,0,0), 5:(0,0,0,0), 6:(0,0,0,0), 7:(0,0,0,0), 8:(0,0,0,0)},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid(r: tuple) -> bool:
    x1, y1, x2, y2 = r
    return x2 > x1 and y2 > y1


def is_calibrated(table_idx: int | None = None) -> bool:
    tables = [table_idx] if table_idx is not None else list(SEAT_REGIONS)
    for tid in tables:
        for r in SEAT_REGIONS.get(tid, {}).values():
            if _valid(r):
                return True
    return False


def get_hero_crop(table_crop: np.ndarray, table_idx: int) -> np.ndarray | None:
    r = HERO_REGIONS.get(table_idx, (0, 0, 0, 0))
    if not _valid(r):
        return None
    x1, y1, x2, y2 = r
    h, w = table_crop.shape[:2]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    c = table_crop[y1:y2, x1:x2]
    return c if c.size > 0 else None


def get_seat_crop(
    table_crop: np.ndarray,
    table_idx: int,
    seat_idx: int,
) -> np.ndarray | None:
    r = SEAT_REGIONS.get(table_idx, {}).get(seat_idx, (0, 0, 0, 0))
    if not _valid(r):
        return None
    x1, y1, x2, y2 = r
    h, w = table_crop.shape[:2]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    c = table_crop[y1:y2, x1:x2]
    return c if c.size > 0 else None
